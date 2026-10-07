"""Tests for scripts/run_datasets.py without any medical data.

End-to-end tests run the real script with a fake ``weavehr`` package
(tests/fake_weavehr) on PYTHONPATH; resource detection is tested against
synthetic cgroup trees.
"""

import importlib.util
import itertools
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "run_datasets.py"
FAKE_WEAVEHR = Path(__file__).resolve().parent / "fake_weavehr"
ALL = ["aumc", "eicu-crd", "hirid", "mimic-iv"]
GIB = 2**30

spec = importlib.util.spec_from_file_location("run_datasets", SCRIPT)
assert spec is not None and spec.loader is not None
runner = importlib.util.module_from_spec(spec)
sys.modules["run_datasets"] = runner  # dataclasses resolve the defining module
spec.loader.exec_module(runner)


# ---------------------------------------------------------------------------
# End-to-end helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def env(tmp_path: Path) -> dict:
    """Empty source directories (no data) and an output root outside the repo."""
    data_root = tmp_path / "physionet"
    registry = yaml.safe_load(
        (REPO / "config/ricu-validation/datasets.yml").read_text()
    )
    for entry in registry["datasets"].values():
        (data_root / entry["source"]).mkdir(parents=True)
    return {
        "data_root": data_root,
        "output_root": tmp_path / "out",
        "record": tmp_path / "events.jsonl",
    }


def run_cli(env: dict, *args: str, **fake: str) -> subprocess.CompletedProcess:
    process_env = {
        **{k: v for k, v in os.environ.items() if not k.startswith("FAKE_WEAVEHR")},
        "PYTHONPATH": str(FAKE_WEAVEHR),
        "FAKE_WEAVEHR_RECORD": str(env["record"]),
        **{f"FAKE_WEAVEHR_{k.upper()}": v for k, v in fake.items()},
    }
    process_env.pop("WEAVEHR_DATA_ROOT", None)
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--output-root",
            str(env["output_root"]),
            "--data-root",
            str(env["data_root"]),
            *args,
        ],
        env=process_env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def events(env: dict) -> list[dict]:
    if not env["record"].is_file():
        return []
    return [json.loads(line) for line in env["record"].read_text().splitlines()]


def run_json(env: dict, dataset: str) -> dict:
    return json.loads((env["output_root"] / dataset / "run.json").read_text())


def started(env: dict) -> list[str]:
    """Datasets whose extraction started, in order."""
    return [
        e["dataset"]
        for e in events(env)
        if e["event"] == "start" and e["step"] == "extraction"
    ]


# ---------------------------------------------------------------------------
# Selection, order, success
# ---------------------------------------------------------------------------


def test_single_dataset(env: dict) -> None:
    result = run_cli(env, "--dataset", "mimic-iv")
    assert result.returncode == 0, result.stderr
    assert started(env) == ["mimic-iv"]
    assert sorted(p.name for p in env["output_root"].iterdir()) == ["logs", "mimic-iv"]
    assert "[1/1] mimic-iv: extraction" in result.stdout
    assert "[1/1] mimic-iv: success" in result.stdout


def test_several_and_all_datasets(env: dict) -> None:
    assert run_cli(env, "--datasets", "hirid", "aumc").returncode == 0
    assert started(env) == ["hirid", "aumc"]

    result = run_cli(env, "--all", "--overwrite")
    assert result.returncode == 0, result.stderr
    assert started(env)[2:] == ALL


def test_datasets_run_strictly_sequentially(env: dict) -> None:
    assert run_cli(env, "--all").returncode == 0
    spans = {}
    for e in events(env):
        if e["event"] in {"start", "end"}:
            spans.setdefault(e["dataset"], []).append(e["t"])
    order = sorted(spans, key=lambda d: min(spans[d]))
    assert order == ALL
    for earlier, later in itertools.pairwise(order):
        assert max(spans[earlier]) <= min(spans[later])
    # Each dataset ran in its own worker process.
    pids = {e["dataset"]: e["pid"] for e in events(env) if e["event"] == "start"}
    assert len(set(pids.values())) == len(ALL)


def test_successful_run_json(env: dict) -> None:
    result = run_cli(env, "--dataset", "mimic-iv", "--threads", "3")
    assert result.returncode == 0, result.stderr
    record = run_json(env, "mimic-iv")

    assert record["schema_version"] == runner.SCHEMA_VERSION
    assert (record["dataset"], record["dataset_version"], record["status"]) == (
        "mimic-iv",
        "2.2",
        "success",
    )
    assert record["error"] is None
    assert record["duration_seconds"] >= 0 and record["finished_at"]
    for step in runner.STEPS:
        assert record["steps"][step]["status"] == "success"
        assert (
            record["steps"][step]["started_at"] and record["steps"][step]["finished_at"]
        )

    root = env["output_root"] / "mimic-iv"
    assert record["output"] == {
        "dataset_root": str(root),
        "project_path": str(root / "project"),
        "log_path": str(env["output_root"] / "logs" / "mimic-iv.log"),
    }
    resources = record["resources"]
    assert resources["threads"] == 3 and resources["threads_explicit"] is True
    assert resources["resource_fraction"] == 0.75
    assert resources["dataset_execution"] == "sequential"
    for key in ("cpus_available", "memory_capacity_bytes", "memory_budget_bytes"):
        assert isinstance(resources[key], int)
    assert "memory_hard_limit_enforced" in resources
    assert set(record["software"]) >= {
        "python_version",
        "weavehr_version",
        "weavehr_commit",
        "weavehr_example_commit",
    }
    files = record["configuration"]["config_files"]
    assert [Path(f["path"]).name for f in files] == [
        "datasets.yml",
        "extraction.yml",
        "concept.yml",
    ]
    assert all(len(f["sha256"]) == 64 for f in files)

    # Generated step configs use the current WeavEHR schema for this dataset.
    concept = yaml.safe_load((root / "config" / "concept.yml").read_text())
    mapping = concept["config"]["mapping_configs"][0]
    assert (mapping["name"], mapping["version"]) == ("mimic-iv", "2.2")
    assert mapping["extension_columns"]["dataset_version"] == 'col("version")'
    extraction = yaml.safe_load((root / "config" / "extraction.yml").read_text())
    assert extraction["config"]["data"][0]["path"] == str(
        (env["data_root"] / "physionet.org/files/mimiciv/2.2").resolve()
    )
    assert (env["output_root"] / "logs" / "mimic-iv.log").is_file()


def test_weavehr_logs_go_to_dataset_log_not_console(env: dict) -> None:
    result = run_cli(env, "--dataset", "aumc")
    assert result.returncode == 0, result.stderr
    assert "Loaded configuration" not in result.stdout
    assert (
        "Loaded configuration" in (env["output_root"] / "logs" / "aumc.log").read_text()
    )


def test_resource_report_is_printed(env: dict) -> None:
    result = run_cli(env, "--dataset", "aumc")
    assert "Resource configuration:" in result.stdout
    assert "Dataset execution:    sequential" in result.stdout
    assert "Hard memory enforced:" in result.stdout


# ---------------------------------------------------------------------------
# Failures
# ---------------------------------------------------------------------------


def test_failure_continues_and_exits_nonzero(env: dict) -> None:
    result = run_cli(env, "--all", fail="eicu-crd:concept")
    assert result.returncode == 1
    assert started(env) == ALL
    assert {d: run_json(env, d)["status"] for d in ALL} == {
        "aumc": "success",
        "eicu-crd": "failed",
        "hirid": "success",
        "mimic-iv": "success",
    }
    assert "[2/4] eicu-crd: FAILED (RuntimeError in concept" in result.stdout


def test_failed_run_json_is_safe(env: dict) -> None:
    run_cli(env, "--dataset", "eicu-crd", fail="eicu-crd:concept")
    record = run_json(env, "eicu-crd")
    assert record["status"] == "failed"
    assert record["error"] == {
        "type": "RuntimeError",
        "message": None,
        "message_redacted": True,
        "step": "concept",
    }
    assert record["steps"]["extraction"]["status"] == "success"
    assert record["steps"]["concept"]["status"] == "failed"

    # The data-like value from the error stays in the private log only.
    run_json_text = (env["output_root"] / "eicu-crd" / "run.json").read_text()
    assert "PATIENT-4711" not in run_json_text
    assert "Traceback" not in run_json_text
    assert "PATIENT-4711" in (env["output_root"] / "logs" / "eicu-crd.log").read_text()


def test_successful_run_json_contains_no_rows(env: dict) -> None:
    assert run_cli(env, "--dataset", "hirid").returncode == 0
    text = (env["output_root"] / "hirid" / "run.json").read_text()
    assert "PATIENT-4711" not in text


def test_stop_on_error(env: dict) -> None:
    result = run_cli(env, "--all", "--stop-on-error", fail="eicu-crd:extraction")
    assert result.returncode == 1
    assert started(env) == ["aumc", "eicu-crd"]
    assert not (env["output_root"] / "hirid").exists()
    assert "hirid      not started" in result.stdout


def test_killed_worker_is_recorded_and_others_continue(env: dict) -> None:
    result = run_cli(env, "--all", crash="hirid:extraction")
    assert result.returncode == 1
    record = run_json(env, "hirid")
    assert record["status"] == "failed"
    assert record["error"]["type"] == "WorkerTerminated"
    assert "SIGKILL" in record["error"]["message"]
    assert record["error"]["step"] == "extraction"
    assert run_json(env, "mimic-iv")["status"] == "success"


def test_missing_source_directory_fails_that_dataset_only(env: dict) -> None:
    (env["data_root"] / "physionet.org/files/hirid/1.1.1").rmdir()
    result = run_cli(env, "--datasets", "hirid", "aumc")
    assert result.returncode == 1
    error = run_json(env, "hirid")["error"]
    assert (
        error["type"] == "RunnerError"
        and "source directory not found" in error["message"]
    )
    assert run_json(env, "aumc")["status"] == "success"


def config_copy(tmp_path: Path, edit=None) -> Path:
    """Writable copy of config/ricu-validation; ``edit(registry)`` may change it."""
    target = tmp_path / "config"
    shutil.copytree(REPO / "config" / "ricu-validation", target)
    if edit is not None:
        path = target / "datasets.yml"
        registry = yaml.safe_load(path.read_text())
        edit(registry)
        path.write_text(yaml.safe_dump(registry, sort_keys=False))
    return target


# ---------------------------------------------------------------------------
# Output completeness
# ---------------------------------------------------------------------------


def test_empty_extraction_is_a_failure_not_a_success(env: dict) -> None:
    """WeavEHR succeeds without writing events when source files are missing."""
    result = run_cli(env, "--dataset", "mimic-iv", empty="mimic-iv:extraction")
    assert result.returncode == 1
    record = run_json(env, "mimic-iv")
    assert record["status"] == "failed"
    assert record["steps"]["extraction"]["status"] == "failed"
    assert record["error"]["type"] == "RunnerError"
    assert record["error"]["step"] == "extraction"
    assert "no output for configured table(s)" in record["error"]["message"]


def test_one_missing_configured_table_fails_extraction(env: dict) -> None:
    result = run_cli(env, "--dataset", "mimic-iv", skip_table="chartevents")
    assert result.returncode == 1
    error = run_json(env, "mimic-iv")["error"]
    assert error["step"] == "extraction"
    assert "['chartevents']" in error["message"]


def test_all_configured_tables_present_succeeds_with_counts(env: dict) -> None:
    assert run_cli(env, "--dataset", "aumc").returncode == 0
    assert run_json(env, "aumc")["outputs"] == {
        "extraction_tables_configured": 4,
        "extraction_tables_with_output": 4,
        "extraction_tables_missing_optional": [],
        "extraction_tables_empty": [],
        "extraction_event_files": 5,
        "extraction_rows": 15,
        "concept_files": 2,
        "concept_files_nonempty": 1,
        "concept_files_empty": 1,
        "concept_rows": 3,
    }


def test_optional_table_may_be_missing(env: dict, tmp_path: Path) -> None:
    def edit(registry: dict) -> None:
        registry["datasets"]["mimic-iv"]["optional_extraction_tables"] = ["chartevents"]

    config = config_copy(tmp_path, edit)
    result = run_cli(
        env,
        "--dataset",
        "mimic-iv",
        "--config-dir",
        str(config),
        skip_table="chartevents",
    )
    assert result.returncode == 0, result.stdout
    outputs = run_json(env, "mimic-iv")["outputs"]
    assert outputs["extraction_tables_missing_optional"] == ["chartevents"]


def test_required_event_with_zero_rows_fails(env: dict) -> None:
    result = run_cli(env, "--dataset", "aumc", zero_event="VISIT_START")
    assert result.returncode == 1
    error = run_json(env, "aumc")["error"]
    assert error["step"] == "extraction"
    assert "VISIT_START events with zero rows" in error["message"]


def test_zero_rows_in_a_non_required_table_is_reported_not_failed(env: dict) -> None:
    # CHART is not required for mimic-iv stay windows.
    assert run_cli(env, "--dataset", "mimic-iv", zero_event="CHART").returncode == 0
    assert run_json(env, "mimic-iv")["outputs"]["extraction_tables_empty"] == [
        "chartevents"
    ]


def test_missing_required_stay_event_fails(env: dict, tmp_path: Path) -> None:
    """Even an optional table cannot drop an event required for stay windows."""

    def edit(registry: dict) -> None:
        registry["datasets"]["hirid"]["optional_extraction_tables"] = ["observations"]

    config = config_copy(tmp_path, edit)
    result = run_cli(
        env,
        "--dataset",
        "hirid",
        "--config-dir",
        str(config),
        skip_table="observations",
    )
    assert result.returncode == 1
    error = run_json(env, "hirid")["error"]
    assert error["step"] == "extraction"
    assert "no OBSERVATION events" in error["message"]


def test_all_empty_concept_files_fail(env: dict) -> None:
    result = run_cli(env, "--dataset", "eicu-crd", zero_concepts="1")
    assert result.returncode == 1
    record = run_json(env, "eicu-crd")
    assert record["steps"]["extraction"]["status"] == "success"
    assert record["error"]["step"] == "concept"
    assert "all 2 concept files of eicu-crd are empty" in record["error"]["message"]


def test_legitimately_empty_concept_does_not_fail(env: dict) -> None:
    # The fake always writes one empty concept next to a non-empty one.
    assert run_cli(env, "--dataset", "hirid").returncode == 0
    outputs = run_json(env, "hirid")["outputs"]
    assert (outputs["concept_files_empty"], outputs["concept_files_nonempty"]) == (1, 1)


def test_empty_concept_step_is_a_failure(env: dict) -> None:
    result = run_cli(env, "--dataset", "eicu-crd", empty="eicu-crd:concept")
    assert result.returncode == 1
    record = run_json(env, "eicu-crd")
    assert record["steps"]["extraction"]["status"] == "success"
    assert record["error"]["step"] == "concept"
    assert "no concept files" in record["error"]["message"]


# ---------------------------------------------------------------------------
# Resume / overwrite
# ---------------------------------------------------------------------------


def test_resume_skips_successful_and_retries_failed(env: dict) -> None:
    run_cli(env, "--all", fail="eicu-crd:concept")
    first = len(started(env))

    result = run_cli(env, "--all", "--resume")
    assert result.returncode == 0, result.stderr
    assert started(env)[first:] == ["eicu-crd"]
    assert "aumc: skipped" in result.stdout
    assert run_json(env, "eicu-crd")["status"] == "success"


def test_resume_retries_interrupted_running_runs(env: dict) -> None:
    run_cli(env, "--dataset", "aumc")
    path = env["output_root"] / "aumc" / "run.json"
    record = json.loads(path.read_text())
    record["status"] = "running"  # e.g. the whole runner was killed mid-run
    path.write_text(json.dumps(record))

    result = run_cli(env, "--dataset", "aumc", "--resume")
    assert result.returncode == 0
    assert started(env) == ["aumc", "aumc"]
    assert run_json(env, "aumc")["status"] == "success"


def test_success_is_never_inferred_from_outputs(env: dict) -> None:
    project = env["output_root"] / "aumc" / "project"
    project.mkdir(parents=True)
    (project / "something.parquet").write_text("not a run record")

    assert run_cli(env, "--dataset", "aumc", "--resume").returncode == 0
    assert started(env) == ["aumc"]
    assert not (project / "something.parquet").exists()  # stale output discarded


def test_existing_output_requires_resume_or_overwrite(env: dict) -> None:
    run_cli(env, "--dataset", "aumc")
    result = run_cli(env, "--dataset", "aumc")
    assert result.returncode == 2
    assert "--resume" in result.stderr and "--overwrite" in result.stderr
    assert started(env) == ["aumc"]

    assert run_cli(env, "--dataset", "aumc", "--overwrite").returncode == 0
    assert started(env) == ["aumc", "aumc"]


def test_resume_and_overwrite_are_exclusive(env: dict) -> None:
    result = run_cli(env, "--dataset", "aumc", "--resume", "--overwrite")
    assert result.returncode == 2
    assert "not allowed with argument" in result.stderr


# ---------------------------------------------------------------------------
# Thread environment before WeavEHR/Polars imports
# ---------------------------------------------------------------------------


def test_thread_environment_is_set_before_weavehr_import(env: dict) -> None:
    assert run_cli(env, "--dataset", "aumc", "--threads", "5").returncode == 0
    imports = [e for e in events(env) if e["event"] == "import"]
    assert len(imports) == 1
    assert imports[0]["threads"] == {name: "5" for name in runner.THREAD_ENV_VARS}
    assert imports[0]["polars_loaded_before"] is False


def test_runner_module_does_not_import_polars() -> None:
    code = (
        "import importlib.util, sys;"
        f"s = importlib.util.spec_from_file_location('r', {str(SCRIPT)!r});"
        "m = importlib.util.module_from_spec(s); sys.modules['r'] = m;"
        "s.loader.exec_module(m);"
        "print('polars' in sys.modules, 'weavehr' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.split() == ["False", "False"]


# ---------------------------------------------------------------------------
# Resource detection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("cpus", "threads"), [(4, 3), (8, 6), (16, 12), (32, 24)])
def test_default_thread_count_is_75_percent(cpus: int, threads: int) -> None:
    assert runner.thread_count(cpus, 0.75) == threads


def test_thread_count_is_at_least_one() -> None:
    assert runner.thread_count(1, 0.75) == 1
    assert runner.thread_count(2, 0.25) == 1


def _cgroup_v2(tmp_path: Path, **files: str) -> tuple[Path, Path]:
    root = tmp_path / "cgroup"
    leaf = root / "user.slice" / "worker.scope"
    leaf.mkdir(parents=True)
    for name, value in files.items():
        (leaf / name.replace("_", ".")).write_text(value + "\n")
    proc = tmp_path / "proc_self_cgroup"
    proc.write_text("0::/user.slice/worker.scope\n")
    return root, proc


def test_cpu_detection_uses_process_affinity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner.os, "sched_getaffinity", lambda pid: set(range(6)))
    root, proc = _cgroup_v2(tmp_path)
    assert runner.detect_cpus(root, proc) == (6, "affinity")


def test_cpu_quota_caps_affinity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner.os, "sched_getaffinity", lambda pid: set(range(32)))
    root, proc = _cgroup_v2(tmp_path, cpu_max="200000 100000")
    assert runner.detect_cpus(root, proc) == (2, "cgroup_cpu_quota")


def test_cgroup_memory_limit_is_preferred_over_host_memory(tmp_path: Path) -> None:
    root, proc = _cgroup_v2(tmp_path, memory_max=str(8 * GIB))
    plan = runner.plan_resources(
        cgroup_root=root, proc_cgroup=proc, host_bytes=64 * GIB
    )
    assert (plan.memory_capacity_bytes, plan.memory_capacity_source) == (
        8 * GIB,
        "cgroup",
    )
    assert plan.memory_budget_bytes == 6 * GIB
    assert plan.memory_hard_limit_enforced is True
    assert plan.memory_hard_limit_bytes == 8 * GIB


def test_host_memory_without_cgroup_limit_is_not_reported_as_enforced(
    tmp_path: Path,
) -> None:
    root, proc = _cgroup_v2(tmp_path, memory_max="max")
    plan = runner.plan_resources(
        cgroup_root=root, proc_cgroup=proc, host_bytes=64 * GIB
    )
    assert (plan.memory_capacity_bytes, plan.memory_capacity_source) == (
        64 * GIB,
        "host",
    )
    assert plan.memory_budget_bytes == 48 * GIB
    assert plan.memory_hard_limit_enforced is False
    assert plan.memory_hard_limit_bytes is None


def test_cgroup_v1_unlimited_memory_is_ignored(tmp_path: Path) -> None:
    root = tmp_path / "cgroup"
    (root / "memory" / "docker").mkdir(parents=True)
    (root / "memory" / "docker" / "memory.limit_in_bytes").write_text(str(2**63 - 4096))
    proc = tmp_path / "proc_self_cgroup"
    proc.write_text("4:memory:/docker\n")
    assert runner.cgroup_memory_limit(root, proc) is None


def test_memory_and_cpu_fractions(tmp_path: Path) -> None:
    root, proc = _cgroup_v2(tmp_path)
    plan = runner.plan_resources(
        resource_fraction=0.5,
        memory_fraction=0.25,
        cgroup_root=root,
        proc_cgroup=proc,
        host_bytes=64 * GIB,
    )
    assert (plan.cpu_fraction, plan.memory_fraction) == (0.5, 0.25)
    assert plan.memory_budget_bytes == 16 * GIB
    assert plan.threads == runner.thread_count(plan.cpus_available, 0.5)


def test_explicit_threads_override_fractions(tmp_path: Path) -> None:
    root, proc = _cgroup_v2(tmp_path)
    plan = runner.plan_resources(
        threads=8, cpu_fraction=0.1, cgroup_root=root, proc_cgroup=proc, host_bytes=GIB
    )
    assert (plan.threads, plan.threads_explicit) == (8, True)


@pytest.mark.parametrize("value", [0.0, -0.5, 1.5, math.nan])
def test_resource_fraction_validation(value: float) -> None:
    with pytest.raises(runner.RunnerError, match="must be > 0 and <= 1"):
        runner.plan_resources(resource_fraction=value)
    with pytest.raises(runner.RunnerError):
        runner.plan_resources(memory_fraction=value)


def test_invalid_cli_resources_fail_before_running(env: dict) -> None:
    result = run_cli(env, "--dataset", "aumc", "--resource-fraction", "1.2")
    assert result.returncode == 2 and "--resource-fraction must be" in result.stderr
    result = run_cli(env, "--dataset", "aumc", "--threads", "0")
    assert result.returncode == 2 and "--threads must be >= 1" in result.stderr
    assert started(env) == []


def test_enforced_memory_wraps_worker_in_systemd_scope(tmp_path: Path) -> None:
    root, proc = _cgroup_v2(tmp_path)
    plan = runner.plan_resources(
        enforce_memory=True, cgroup_root=root, proc_cgroup=proc, host_bytes=64 * GIB
    )
    command = runner.worker_command(
        dataset="aumc", output_root=tmp_path, position="1/1", plan=plan
    )
    assert command[:5] == ["systemd-run", "--user", "--scope", "--quiet", "-p"]
    assert command[5] == f"MemoryMax={48 * GIB}"
    # Requested, but only reported as enforced once the worker observes the limit.
    assert plan.memory_hard_limit_enforced is False


def test_enforce_memory_without_systemd_run_fails(
    env: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner.shutil, "which", lambda name: None)
    args = runner.build_parser().parse_args(
        [
            "--dataset",
            "aumc",
            "--enforce-memory",
            "--output-root",
            str(env["output_root"]),
        ]
    )
    with pytest.raises(runner.RunnerError, match="needs systemd-run"):
        runner.run(args)


def test_output_root_inside_repository_is_rejected(env: dict) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--dataset",
            "aumc",
            "--output-root",
            str(REPO / "out"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2 and "outside the repository" in result.stderr
    assert not (REPO / "out").exists()


# ---------------------------------------------------------------------------
# Overwrite invalidation and setup failures
# ---------------------------------------------------------------------------


def _broken_templates(tmp_path: Path) -> Path:
    """Config whose concept template makes setup fail after outputs exist."""
    config = config_copy(tmp_path)
    (config / "concept.yml").write_text("name: Concept\nversion: 1.0.0\n")
    return config


def test_failed_overwrite_never_leaves_stale_success(env: dict, tmp_path: Path) -> None:
    assert run_cli(env, "--dataset", "aumc").returncode == 0
    assert run_json(env, "aumc")["status"] == "success"

    result = run_cli(
        env,
        "--dataset",
        "aumc",
        "--overwrite",
        "--config-dir",
        str(_broken_templates(tmp_path)),
    )
    assert result.returncode == 1
    record = run_json(env, "aumc")
    assert record["status"] == "failed"
    assert record["error"]["step"] == "setup"
    assert record["error"]["type"] == "KeyError" and record["error"]["message"] is None

    # A later --resume retries instead of skipping.
    result = run_cli(env, "--dataset", "aumc", "--resume")
    assert result.returncode == 0
    assert "skipped" not in result.stdout
    assert started(env) == ["aumc", "aumc"]
    assert run_json(env, "aumc")["rerun"]["previous_status"] == "failed"


def test_overwrite_invalidates_before_deleting_outputs(env: dict) -> None:
    assert run_cli(env, "--dataset", "aumc").returncode == 0
    path = env["output_root"] / "aumc" / "run.json"
    seen = []

    def fake_invalidate(run_json: Path, previous: dict, reason: str) -> None:
        original_invalidate(run_json, previous, reason)
        # At this point the old outputs still exist, but success is gone.
        seen.append(
            (json.loads(path.read_text())["status"], (path.parent / "project").exists())
        )
        raise KeyboardInterrupt  # interrupted right after invalidation

    original_invalidate = runner._invalidate
    args = runner.build_parser().parse_args(
        [
            "--dataset",
            "aumc",
            "--overwrite",
            "--output-root",
            str(env["output_root"]),
            "--data-root",
            str(env["data_root"]),
        ]
    )
    runner._invalidate = fake_invalidate
    try:
        with pytest.raises(KeyboardInterrupt):
            runner.run(args, launcher=lambda command, environment: 0)
    finally:
        runner._invalidate = original_invalidate
    assert seen == [("invalidated", True)]
    assert json.loads(path.read_text())["status"] == "invalidated"


def test_setup_exception_does_not_stop_later_datasets(
    env: dict, tmp_path: Path
) -> None:
    def edit(registry: dict) -> None:
        registry["datasets"]["aumc"]["extension_columns"] = ["not", "a", "mapping"]

    config = config_copy(tmp_path, edit)
    result = run_cli(env, "--datasets", "aumc", "hirid", "--config-dir", str(config))
    assert result.returncode == 1
    assert run_json(env, "aumc")["status"] == "failed"
    assert run_json(env, "aumc")["error"]["step"] == "setup"
    assert run_json(env, "hirid")["status"] == "success"

    result = run_cli(
        env,
        "--datasets",
        "aumc",
        "hirid",
        "--config-dir",
        str(config),
        "--overwrite",
        "--stop-on-error",
    )
    assert result.returncode == 1
    assert "hirid      not started" in result.stdout


# ---------------------------------------------------------------------------
# --resume compares the run inputs
# ---------------------------------------------------------------------------


def _resume(env: dict, config: Path, *extra: str) -> subprocess.CompletedProcess:
    return run_cli(
        env, "--dataset", "mimic-iv", "--resume", "--config-dir", str(config), *extra
    )


def test_resume_skips_unchanged_configuration(env: dict, tmp_path: Path) -> None:
    config = config_copy(tmp_path)
    assert _resume(env, config).returncode == 0
    result = _resume(env, config)
    assert "mimic-iv: skipped" in result.stdout
    assert started(env) == ["mimic-iv"]


def test_resume_ignores_unrelated_run_metadata(env: dict, tmp_path: Path) -> None:
    config = config_copy(tmp_path)
    assert _resume(env, config).returncode == 0
    path = env["output_root"] / "mimic-iv" / "run.json"
    record = json.loads(path.read_text())
    record.update(started_at="2000-01-01T00:00:00+00:00", duration_seconds=1.0)
    record["resources"]["threads"] = 99
    record["software"]["weavehr_example_commit"] = "something-else"
    path.write_text(json.dumps(record))

    assert "mimic-iv: skipped" in _resume(env, config).stdout
    assert started(env) == ["mimic-iv"]


def _changed_and_rerun(env: dict, tmp_path: Path, change, *extra: str) -> dict:
    config = config_copy(tmp_path)
    assert _resume(env, config).returncode == 0
    change(config)
    result = _resume(env, config, *extra)
    assert result.returncode == 0, result.stdout
    assert "previous successful run configuration changed" in result.stdout
    assert started(env) == ["mimic-iv", "mimic-iv"]
    rerun = run_json(env, "mimic-iv")["rerun"]
    assert rerun["reason"] == "configuration changed"
    assert rerun["previous_status"] == "success"
    return rerun


def _edit_registry(config: Path, key: str, value: object) -> None:
    path = config / "datasets.yml"
    registry = yaml.safe_load(path.read_text())
    registry["datasets"]["mimic-iv"][key] = value
    path.write_text(yaml.safe_dump(registry, sort_keys=False))


def test_resume_reruns_when_version_changes(env: dict, tmp_path: Path) -> None:
    rerun = _changed_and_rerun(
        env, tmp_path, lambda config: _edit_registry(config, "version", "3.1")
    )
    assert "dataset_version" in rerun["changed_inputs"]
    assert rerun["previous_inputs"]["dataset_version"] == "2.2"
    assert run_json(env, "mimic-iv")["dataset_version"] == "3.1"


def test_resume_reruns_when_source_path_changes(env: dict, tmp_path: Path) -> None:
    other = tmp_path / "other-data"
    shutil.copytree(env["data_root"], other)
    config = config_copy(tmp_path)
    assert _resume(env, config).returncode == 0
    result = _resume(env, config, "--data-root", str(other))
    assert "configuration changed (source_path" in result.stdout
    assert run_json(env, "mimic-iv")["rerun"]["changed_inputs"] == [
        "source_path",
        "extraction_config_sha256",
    ]


def test_resume_reruns_when_extraction_config_changes(
    env: dict, tmp_path: Path
) -> None:
    def change(config: Path) -> None:
        template = config / "extraction.yml"
        template.write_text(template.read_text() + "overwrite: true\n")

    rerun = _changed_and_rerun(env, tmp_path, change)
    assert rerun["changed_inputs"] == ["extraction_config_sha256"]


def test_resume_reruns_when_concept_config_changes(env: dict, tmp_path: Path) -> None:
    rerun = _changed_and_rerun(
        env,
        tmp_path,
        lambda config: _edit_registry(
            config, "extension_columns", {"dataset_version": 'col("version")'}
        ),
    )
    assert rerun["changed_inputs"] == ["concept_config_sha256"]


# ---------------------------------------------------------------------------
# Small safety checks
# ---------------------------------------------------------------------------


def test_output_root_overlapping_source_data_is_rejected(env: dict) -> None:
    inside = run_cli(
        env, "--dataset", "aumc", "--output-root", str(env["data_root"] / "out")
    )
    assert inside.returncode == 2 and "overlaps source data" in inside.stderr
    around = run_cli(
        env, "--dataset", "aumc", "--output-root", str(env["data_root"].parent)
    )
    assert around.returncode == 2 and "overlaps source data" in around.stderr
    assert started(env) == []


def test_unquoted_versions_are_rejected(env: dict, tmp_path: Path) -> None:
    config = config_copy(tmp_path)
    path = config / "datasets.yml"
    path.write_text(path.read_text().replace('version: "2.2"', "version: 2.2"))
    result = run_cli(env, "--dataset", "mimic-iv", "--config-dir", str(config))
    assert result.returncode == 2
    assert "must be a quoted string" in result.stderr
