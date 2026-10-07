#!/usr/bin/env python
"""Run the WeavEHR pipeline (extraction -> concepts) for several datasets.

Datasets run strictly one after another. Each dataset runs in its own worker
process (started only after the previous one has finished), so its memory is
released completely afterwards and a crash or out-of-memory kill only fails
that dataset. Every dataset gets an isolated output directory with a
``run.json`` (machine-readable, free of patient-level data) and its own log.

This script runs WeavEHR only. RICU/YAIB validation (weavehr-yaib) and concept
coverage updates are separate, later steps.

Thread limits must be in place before Polars is imported, so this module
never imports Polars or WeavEHR at module level; the worker imports WeavEHR
only after its environment carries the thread settings.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import math
import os
import shutil
import signal
import subprocess
import sys
import traceback
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import yaml

SCHEMA_VERSION = "1.1"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_DIR = REPO_ROOT / "config" / "ricu-validation"
DEFAULT_OUTPUT_ROOT = Path("~/output/WeavEHR.example/ricu-validation")
VALIDATION_DATASETS = ("aumc", "eicu-crd", "hirid", "mimic-iv")
DEFAULT_RESOURCE_FRACTION = 0.75
THREAD_ENV_VARS = (
    "POLARS_MAX_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
# WeavEHR steps run per dataset, named as in the step configs (lowercased).
STEPS = ("extraction", "concept")
# Name of WeavEHR's package logger (weavehr.logging.LOGGER_NAME).
WEAVEHR_LOGGER = "weavehr"
# cgroup v1 reports "unlimited" as a huge number.
_CGROUP_V1_UNLIMITED = 2**60

logger = logging.getLogger("run_datasets")


class RunnerError(Exception):
    """A runner error whose message never contains source data."""


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------


def validate_fraction(name: str, value: float) -> float:
    """Fractions must lie in (0, 1]."""
    if not (0.0 < value <= 1.0) or math.isnan(value):
        raise RunnerError(f"{name} must be > 0 and <= 1, got {value}")
    return value


def _cgroup_dirs(
    controller: str, cgroup_root: Path, proc_cgroup: Path
) -> list[tuple[Path, int]]:
    """cgroup directories of this process and its ancestors (with cgroup version)."""
    try:
        lines = proc_cgroup.read_text().splitlines()
    except OSError:
        return []
    found: list[tuple[Path, int]] = []
    for line in lines:
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        hierarchy, controllers, rel = parts
        if hierarchy == "0" and controllers == "":
            base, version = cgroup_root, 2
        elif controller in controllers.split(","):
            base, version = cgroup_root / controllers, 1
        else:
            continue
        current = base / rel.lstrip("/")
        while True:
            if current.is_dir():
                found.append((current, version))
            if current == base:
                break
            current = current.parent
    return found


def _read(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def cgroup_memory_limit(
    cgroup_root: Path = Path("/sys/fs/cgroup"),
    proc_cgroup: Path = Path("/proc/self/cgroup"),
) -> int | None:
    """Smallest finite cgroup memory limit applying to this process, if any."""
    limits: list[int] = []
    for directory, version in _cgroup_dirs("memory", cgroup_root, proc_cgroup):
        name = "memory.max" if version == 2 else "memory.limit_in_bytes"
        value = _read(directory / name)
        if value is None or value == "max" or not value.isdigit():
            continue
        limit = int(value)
        if version == 1 and limit >= _CGROUP_V1_UNLIMITED:
            continue
        limits.append(limit)
    return min(limits) if limits else None


def cgroup_cpu_quota(
    cgroup_root: Path = Path("/sys/fs/cgroup"),
    proc_cgroup: Path = Path("/proc/self/cgroup"),
) -> float | None:
    """Smallest CPU quota (in CPUs) applying to this process, if any."""
    quotas: list[float] = []
    for directory, version in _cgroup_dirs("cpu", cgroup_root, proc_cgroup):
        if version == 2:
            value = _read(directory / "cpu.max")
            if not value:
                continue
            quota, _, period = value.partition(" ")
            if quota == "max" or not quota.isdigit() or not period.isdigit():
                continue
            quotas.append(int(quota) / int(period))
        else:
            quota_us = _read(directory / "cpu.cfs_quota_us")
            period_us = _read(directory / "cpu.cfs_period_us")
            if quota_us is None or period_us is None or quota_us.startswith("-"):
                continue
            quotas.append(int(quota_us) / int(period_us))
    return min(quotas) if quotas else None


def detect_cpus(
    cgroup_root: Path = Path("/sys/fs/cgroup"),
    proc_cgroup: Path = Path("/proc/self/cgroup"),
) -> tuple[int, str]:
    """CPUs available to this process and how they were determined.

    Order: process affinity, ``os.process_cpu_count``, ``os.cpu_count``; a
    smaller cgroup CPU quota caps the result.
    """
    if hasattr(os, "sched_getaffinity"):
        cpus, source = len(os.sched_getaffinity(0)), "affinity"
    elif getattr(os, "process_cpu_count", None) and os.process_cpu_count():
        cpus, source = os.process_cpu_count() or 1, "process_cpu_count"
    else:
        cpus, source = os.cpu_count() or 1, "cpu_count"
    quota = cgroup_cpu_quota(cgroup_root, proc_cgroup)
    if quota is not None and quota < cpus:
        cpus, source = max(1, math.floor(quota)), "cgroup_cpu_quota"
    return max(1, cpus), source


def host_memory_bytes() -> int:
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")


def detect_memory(
    cgroup_root: Path = Path("/sys/fs/cgroup"),
    proc_cgroup: Path = Path("/proc/self/cgroup"),
    host_bytes: int | None = None,
) -> tuple[int, str, int | None]:
    """Memory capacity, its source and the finite cgroup limit (if any).

    A finite cgroup/container limit below physical RAM is the capacity;
    otherwise physical RAM is. Instantaneous MemAvailable is not used.
    """
    host = host_bytes if host_bytes is not None else host_memory_bytes()
    limit = cgroup_memory_limit(cgroup_root, proc_cgroup)
    if limit is not None and limit < host:
        return limit, "cgroup", limit
    return host, "host", limit


def thread_count(cpus: int, fraction: float) -> int:
    return max(1, math.floor(cpus * fraction))


@dataclass(frozen=True)
class ResourcePlan:
    resource_fraction: float
    cpu_fraction: float
    memory_fraction: float
    cpus_available: int
    cpus_source: str
    threads: int
    threads_explicit: bool
    memory_capacity_bytes: int
    memory_capacity_source: str
    memory_budget_bytes: int
    # Finite cgroup limit actually in force for this process (container or
    # systemd scope); None means the budget is only reported, not enforced.
    memory_hard_limit_bytes: int | None
    memory_hard_limit_enforced: bool
    # Requested per-worker hard limit via systemd-run (verified in the worker).
    memory_hard_limit_requested: str | None
    dataset_execution: str = "sequential"


def plan_resources(
    *,
    resource_fraction: float = DEFAULT_RESOURCE_FRACTION,
    cpu_fraction: float | None = None,
    memory_fraction: float | None = None,
    threads: int | None = None,
    enforce_memory: bool = False,
    cgroup_root: Path = Path("/sys/fs/cgroup"),
    proc_cgroup: Path = Path("/proc/self/cgroup"),
    host_bytes: int | None = None,
) -> ResourcePlan:
    resource_fraction = validate_fraction("--resource-fraction", resource_fraction)
    cpu_fraction = validate_fraction(
        "--cpu-fraction",
        cpu_fraction if cpu_fraction is not None else resource_fraction,
    )
    memory_fraction = validate_fraction(
        "--memory-fraction",
        memory_fraction if memory_fraction is not None else resource_fraction,
    )
    if threads is not None and threads < 1:
        raise RunnerError(f"--threads must be >= 1, got {threads}")

    cpus, cpus_source = detect_cpus(cgroup_root, proc_cgroup)
    capacity, capacity_source, limit = detect_memory(
        cgroup_root, proc_cgroup, host_bytes
    )
    budget = int(capacity * memory_fraction)
    return ResourcePlan(
        resource_fraction=resource_fraction,
        cpu_fraction=cpu_fraction,
        memory_fraction=memory_fraction,
        cpus_available=cpus,
        cpus_source=cpus_source,
        threads=threads if threads is not None else thread_count(cpus, cpu_fraction),
        threads_explicit=threads is not None,
        memory_capacity_bytes=capacity,
        memory_capacity_source=capacity_source,
        memory_budget_bytes=budget,
        memory_hard_limit_bytes=limit,
        memory_hard_limit_enforced=limit is not None,
        memory_hard_limit_requested=(
            f"systemd-run --user --scope MemoryMax={budget}" if enforce_memory else None
        ),
    )


def thread_environment(threads: int) -> dict[str, str]:
    return {name: str(threads) for name in THREAD_ENV_VARS}


def _gib(n: int) -> str:
    return f"{n / 2**30:.1f} GiB"


def format_resources(plan: ResourcePlan) -> str:
    threads = f"{plan.threads}" + (
        " (explicit --threads)" if plan.threads_explicit else ""
    )
    if plan.memory_hard_limit_requested:
        enforced = (
            f"requested per dataset worker (systemd-run MemoryMax="
            f"{_gib(plan.memory_budget_bytes)}), verified in run.json"
        )
    elif plan.memory_hard_limit_enforced:
        assert plan.memory_hard_limit_bytes is not None
        enforced = f"yes (cgroup limit {_gib(plan.memory_hard_limit_bytes)})"
    else:
        enforced = "no (budget is reported, not enforced)"
    return "\n".join(
        [
            "Resource configuration:",
            f"  CPUs available:       {plan.cpus_available} ({plan.cpus_source})",
            f"  Threads selected:     {threads}",
            f"  CPU fraction:         {plan.cpu_fraction:.0%}",
            (
                f"  Memory capacity:      {_gib(plan.memory_capacity_bytes)}"
                f" ({plan.memory_capacity_source})"
            ),
            (
                f"  Memory budget:        {_gib(plan.memory_budget_bytes)}"
                f" ({plan.memory_fraction:.0%})"
            ),
            f"  Hard memory enforced: {enforced}",
            f"  Dataset execution:    {plan.dataset_execution}",
        ]
    )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def load_dataset_registry(config_dir: Path) -> dict:
    path = config_dir / "datasets.yml"
    try:
        data = yaml.safe_load(path.read_text()) or {}
    except OSError as exc:
        raise RunnerError(f"cannot read dataset configuration {path}: {exc}") from exc
    datasets = data.get("datasets")
    if not isinstance(datasets, dict) or not datasets:
        raise RunnerError(f"{path} defines no datasets")
    for name, entry in datasets.items():
        for key in ("version", "source"):
            if not isinstance(entry, dict) or not entry.get(key):
                raise RunnerError(f"{path}: dataset {name!r} needs '{key}'")
        # Unquoted YAML versions become numbers (1.10 -> 1.1): require strings.
        if not isinstance(entry["version"], str):
            raise RunnerError(
                f"{path}: version of {name!r} must be a quoted string, "
                f'e.g. version: "{entry["version"]}"'
            )
        for key in ("required_extraction_events", "optional_extraction_tables"):
            values = entry.get(key) or []
            if not isinstance(values, list) or not all(
                isinstance(v, str) for v in values
            ):
                raise RunnerError(f"{path}: {key} of {name!r} must be a list of names")
    return data


def resolve_data_root(cli_value: str | None, registry: dict) -> Path:
    value = (
        cli_value or os.environ.get("WEAVEHR_DATA_ROOT") or registry.get("data_root")
    )
    if not value:
        raise RunnerError(
            "No source data root configured: pass --data-root, set WEAVEHR_DATA_ROOT "
            "or set data_root in datasets.yml"
        )
    return Path(value).expanduser()


def source_path(data_root: Path, entry: dict) -> Path:
    return (data_root / Path(str(entry["source"])).expanduser()).resolve()


def render_step_configs(
    dataset: str, entry: dict, source: Path, config_dir: Path
) -> dict[str, str]:
    """Fill the WeavEHR step templates for one dataset (YAML text, deterministic)."""
    version = entry["version"]

    extraction = yaml.safe_load((config_dir / "extraction.yml").read_text())
    extraction["config"]["data"] = [
        {"name": dataset, "version": version, "path": str(source)}
    ]
    concept = yaml.safe_load((config_dir / "concept.yml").read_text())
    concept["config"]["mapping_configs"] = [
        {
            "name": dataset,
            "version": version,
            "extension_columns": dict(entry.get("extension_columns") or {}),
        }
    ]
    return {
        "extraction": yaml.safe_dump(extraction, sort_keys=False),
        "concept": yaml.safe_dump(concept, sort_keys=False),
    }


def write_step_configs(rendered: dict[str, str], target_dir: Path) -> dict[str, Path]:
    target_dir.mkdir(parents=True, exist_ok=True)
    paths = {step: target_dir / f"{step}.yml" for step in rendered}
    for step, text in rendered.items():
        paths[step].write_text(text)
    return paths


def run_inputs(
    entry: dict, source: Path, rendered: dict[str, str]
) -> dict[str, object]:
    """Everything that determines a dataset run's result (compared on --resume)."""
    return {
        "dataset_version": entry["version"],
        "source_path": str(source),
        "extraction_config_sha256": hashlib.sha256(
            rendered["extraction"].encode()
        ).hexdigest(),
        "concept_config_sha256": hashlib.sha256(
            rendered["concept"].encode()
        ).hexdigest(),
        "required_extraction_events": sorted(
            entry.get("required_extraction_events") or []
        ),
        "optional_extraction_tables": sorted(
            entry.get("optional_extraction_tables") or []
        ),
    }


def changed_inputs(previous: object, current: dict[str, object]) -> list[str]:
    """Names of run inputs that differ from a previous run (all if unknown)."""
    if not isinstance(previous, dict):
        return ["unrecorded inputs"]
    return [key for key, value in current.items() if previous.get(key) != value]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# run.json
# ---------------------------------------------------------------------------


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def read_run_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {"status": "invalid"}
    return data if isinstance(data, dict) else {"status": "invalid"}


def write_run_json(path: Path, data: dict) -> None:
    """Atomic write so an interrupted run never leaves a half-written file."""
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(tmp, path)


def _git(*args: str, cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def software_info() -> dict:
    """Versions without importing WeavEHR (or Polars)."""
    import importlib.metadata
    import importlib.util

    try:
        weavehr_version = importlib.metadata.version("weavehr")
    except importlib.metadata.PackageNotFoundError:
        weavehr_version = None
    spec = importlib.util.find_spec("weavehr")
    weavehr_dir = Path(spec.origin).parent if spec and spec.origin else None
    status = _git("status", "--porcelain", cwd=REPO_ROOT)
    return {
        "python_version": sys.version.split()[0],
        "weavehr_version": weavehr_version,
        "weavehr_commit": _git("rev-parse", "HEAD", cwd=weavehr_dir)
        if weavehr_dir
        else None,
        "weavehr_example_commit": _git("rev-parse", "HEAD", cwd=REPO_ROOT),
        "weavehr_example_dirty": None if status is None else bool(status),
    }


# Exceptions whose messages are known not to contain source data.
_SAFE_MESSAGE_TYPES: tuple[type[BaseException], ...] = (RunnerError, OSError)


def safe_error(exc: BaseException, step: str | None) -> dict:
    """Error summary for run.json: no traceback and no data-bearing messages.

    Messages of library errors (e.g. Polars) can quote source values, so they
    are replaced by a pointer to the private dataset log.
    """
    safe = isinstance(exc, _SAFE_MESSAGE_TYPES)
    return {
        "type": type(exc).__name__,
        "message": str(exc)[:500] if safe else None,
        "message_redacted": not safe,
        "step": step,
    }


def new_run_record(
    *,
    dataset: str,
    entry: dict,
    dataset_root: Path,
    plan: ResourcePlan,
    config_files: Sequence[Path],
    source: Path,
    inputs: dict[str, object],
    rerun: dict[str, object] | None,
) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "runner": "scripts/run_datasets.py",
        "dataset": dataset,
        "dataset_version": str(entry["version"]),
        "status": "running",
        "started_at": now(),
        "finished_at": None,
        "duration_seconds": None,
        "output": {
            "dataset_root": str(dataset_root),
            "project_path": str(dataset_root / "project"),
            "log_path": str(dataset_root.parent / "logs" / f"{dataset}.log"),
        },
        "resources": asdict(plan),
        "steps": {
            step: {"status": "pending", "started_at": None, "finished_at": None}
            for step in STEPS
        },
        "software": software_info(),
        # Counts and table/concept names only; no rows or values.
        "outputs": {},
        # Compared with the current configuration before --resume skips a run.
        "inputs": inputs,
        # Why an earlier run of this dataset was replaced (None for a first run).
        "rerun": rerun,
        "configuration": {
            "source_path": str(source),
            "required_extraction_events": list(
                entry.get("required_extraction_events") or []
            ),
            "optional_extraction_tables": list(
                entry.get("optional_extraction_tables") or []
            ),
            "config_files": [
                {"path": str(p), "sha256": _sha256(p)} for p in config_files
            ],
        },
        "error": None,
    }


def _finish(record: dict, status: str, error: dict | None) -> None:
    record["status"] = status
    record["finished_at"] = now()
    started = datetime.fromisoformat(record["started_at"])
    record["duration_seconds"] = round(
        (datetime.fromisoformat(record["finished_at"]) - started).total_seconds(), 1
    )
    record["error"] = error


def _running_step(record: dict) -> str | None:
    return next(
        (s for s, v in record["steps"].items() if v["status"] == "running"), None
    )


def parquet_rows(path: Path) -> int:
    """Row count from the Parquet footer; no rows are read."""
    import pyarrow.parquet as pq

    try:
        return pq.read_metadata(path).num_rows
    except Exception as exc:
        raise RunnerError(
            f"unreadable Parquet file {path} ({type(exc).__name__})"
        ) from exc


def expected_extraction_tables(dataset: str, version: str) -> dict[str, list[str]]:
    """Tables (and their events) WeavEHR extracts for this dataset version.

    The same registry query as WeavEHR's ExtractionStep (no includes/excludes
    in the generated config). Imported lazily: worker process only.
    """
    from weavehr import dataset_config_registry

    return {
        table.name: [event.name for event in table.events]
        for table in dataset_config_registry.filter(dataset, version)
    }


def check_extraction_output(
    base: Path,
    dataset: str,
    version: str,
    expected: dict[str, list[str]],
    required_events: Sequence[str],
    optional_tables: Sequence[str],
) -> dict[str, object]:
    """Every configured table must have produced output; required events need rows.

    WeavEHR logs "Skipping table" and continues when a table's source files are
    missing, so a missing table directory means missing source coverage. Tables
    listed in ``optional_extraction_tables`` may be absent.
    """
    if not expected:
        raise RunnerError(
            f"WeavEHR configures no extraction tables for {dataset} {version}"
        )
    produced = {
        table: sorted((base / table).glob("*.parquet"))
        for table in expected
        if (base / table).is_dir()
    }
    produced = {table: files for table, files in produced.items() if files}
    missing = sorted(
        t for t in expected if t not in produced and t not in optional_tables
    )
    if missing:
        raise RunnerError(
            f"extraction produced no output for configured table(s) {missing} of "
            f"{dataset} {version}; their source files are probably missing "
            "(see 'Skipping table' in the dataset log) or mark them optional"
        )

    table_rows: dict[str, int] = {}
    event_rows: dict[str, int] = {}
    for table, files in produced.items():
        for file in files:
            rows = parquet_rows(file)
            table_rows[table] = table_rows.get(table, 0) + rows
            event_rows[file.stem] = event_rows.get(file.stem, 0) + rows

    for event in required_events:
        if event not in event_rows:
            raise RunnerError(
                f"extraction produced no {event} events for {dataset} {version}; "
                "they are required downstream (stay windows)"
            )
        if event_rows[event] == 0:
            raise RunnerError(
                f"extraction produced {event} events with zero rows for {dataset} "
                f"{version}; they are required downstream (stay windows)"
            )
    total = sum(table_rows.values())
    if total == 0:
        raise RunnerError(f"extraction produced only empty event files for {dataset}")
    empty_tables = sorted(t for t, rows in table_rows.items() if rows == 0)
    if empty_tables:
        logger.warning("extraction tables without rows: %s", ", ".join(empty_tables))
    return {
        "extraction_tables_configured": len(expected),
        "extraction_tables_with_output": len(produced),
        "extraction_tables_missing_optional": sorted(set(expected) - set(produced)),
        "extraction_tables_empty": empty_tables,
        "extraction_event_files": sum(len(files) for files in produced.values()),
        "extraction_rows": total,
    }


def check_concept_output(concept_root: Path, dataset: str) -> dict[str, object]:
    """Concept files may be empty individually (a concept can be absent from a
    dataset), but a run whose concept files are all empty failed."""
    files = sorted(concept_root.glob(f"*/*/{dataset}.parquet"))
    if not files:
        raise RunnerError(f"concept generation produced no concept files for {dataset}")
    rows = [parquet_rows(file) for file in files]
    nonempty = sum(1 for n in rows if n > 0)
    if nonempty == 0:
        raise RunnerError(
            f"all {len(files)} concept files of {dataset} are empty; no concept "
            "mapping matched the extracted data"
        )
    return {
        "concept_files": len(files),
        "concept_files_nonempty": nonempty,
        "concept_files_empty": len(files) - nonempty,
        "concept_rows": sum(rows),
    }


# ---------------------------------------------------------------------------
# Worker (one dataset, own process)
# ---------------------------------------------------------------------------

Pipeline = Callable[[Path, dict[str, Path], Callable[[str], None]], None]


def run_weavehr_pipeline(
    project_path: Path, step_configs: dict[str, Path], on_step: Callable[[str], None]
) -> None:
    """Extraction -> concepts with the current WeavEHR API.

    Imported here, after the thread environment is set. Sharding is not run:
    weavehr-yaib reads the extraction and concept outputs only.
    """
    from weavehr import ConceptStep, ExtractionStep, WeavEHRProject

    with WeavEHRProject(project_path) as project:
        on_step("extraction")
        ExtractionStep.load(project, step_configs["extraction"]).run()
        on_step("concept")
        ConceptStep.load(project, step_configs["concept"]).run()


def worker_main(
    dataset: str,
    output_root: Path,
    position: str,
    pipeline: Pipeline = run_weavehr_pipeline,
    expected_tables: Callable[[str, str], dict[str, list[str]]] = (
        expected_extraction_tables
    ),
) -> int:
    dataset_root = output_root / dataset
    run_json = dataset_root / "run.json"
    record = read_run_json(run_json)
    if not record or record.get("status") != "running":
        raise RunnerError(f"{run_json} was not prepared by the runner")

    handler = logging.FileHandler(Path(record["output"]["log_path"]), encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    # WeavEHR logs to its own non-propagating "weavehr" logger and adds a
    # stdout handler on first use unless one exists. Attaching the dataset log
    # here, before WeavEHR is imported, keeps its output in logs/<dataset>.log
    # and the console concise.
    weavehr_logger = logging.getLogger(WEAVEHR_LOGGER)
    weavehr_logger.addHandler(handler)
    weavehr_logger.setLevel(logging.INFO)
    weavehr_logger.propagate = False

    # Record the hard limit actually in force for this worker process.
    _, _, limit = detect_memory()
    record["resources"]["memory_hard_limit_bytes"] = limit
    record["resources"]["memory_hard_limit_enforced"] = limit is not None
    write_run_json(run_json, record)

    project_path = dataset_root / "project"

    version = record["dataset_version"]
    configuration = record["configuration"]

    def finish_step(step: str) -> None:
        # Raises RunnerError while ``step`` is still running, so it is the
        # step recorded as failed.
        workspace = project_path / "workspace"
        if step == "extraction":
            record["outputs"].update(
                check_extraction_output(
                    workspace / "extraction" / dataset / version,
                    dataset,
                    version,
                    expected_tables(dataset, version),
                    configuration["required_extraction_events"],
                    configuration["optional_extraction_tables"],
                )
            )
        elif step == "concept":
            record["outputs"].update(
                check_concept_output(workspace / "concept", dataset)
            )
        record["steps"][step].update(status="success", finished_at=now())

    def on_step(step: str) -> None:
        previous = _running_step(record)
        if previous is not None:
            finish_step(previous)
        record["steps"][step].update(status="running", started_at=now())
        write_run_json(run_json, record)
        logger.info("dataset %s: step %s started", dataset, step)
        print(f"[{position}] {dataset}: {step}", flush=True)

    config_dir = dataset_root / "config"
    step_configs = {step: config_dir / f"{step}.yml" for step in STEPS}
    try:
        pipeline(project_path, step_configs, on_step)
        last = _running_step(record)
        if last is not None:
            finish_step(last)
    except BaseException as exc:  # noqa: BLE001 - recorded, then re-signalled by exit code
        step = _running_step(record)
        if step is not None:
            record["steps"][step].update(status="failed", finished_at=now())
        logger.error(
            "dataset %s failed in step %s\n%s", dataset, step, traceback.format_exc()
        )
        _finish(record, "failed", safe_error(exc, step))
        write_run_json(run_json, record)
        return 1
    finally:
        gc.collect()
        root.removeHandler(handler)
        weavehr_logger.removeHandler(handler)
        handler.close()

    if any(v["status"] != "success" for v in record["steps"].values()):
        _finish(
            record,
            "failed",
            {
                "type": "IncompletePipeline",
                "message": "not all steps ran",
                "message_redacted": False,
                "step": None,
            },
        )
        write_run_json(run_json, record)
        return 1
    _finish(record, "success", None)
    write_run_json(run_json, record)
    logger.info("dataset %s finished successfully", dataset)
    return 0


# ---------------------------------------------------------------------------
# Parent: sequential driver
# ---------------------------------------------------------------------------

Launcher = Callable[[list[str], dict[str, str]], int]


def launch_worker(command: list[str], env: dict[str, str]) -> int:
    """Run one dataset worker and wait for it (strictly sequential)."""
    return subprocess.run(command, env=env, check=False).returncode


def worker_command(
    *, dataset: str, output_root: Path, position: str, plan: ResourcePlan
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--dataset",
        dataset,
        "--output-root",
        str(output_root),
        "--position",
        position,
    ]
    if plan.memory_hard_limit_requested:
        command = [
            "systemd-run",
            "--user",
            "--scope",
            "--quiet",
            "-p",
            f"MemoryMax={plan.memory_budget_bytes}",
            *command,
        ]
    return command


def _fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return "?"
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m {secs}s" if hours else f"{minutes}m {secs}s"


def _inside(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def select_datasets(args: argparse.Namespace, registry: dict) -> list[str]:
    if args.all:
        selected = list(VALIDATION_DATASETS)
    elif args.datasets:
        selected = list(dict.fromkeys(args.datasets))
    else:
        selected = [args.dataset]
    unknown = [d for d in selected if d not in registry["datasets"]]
    if unknown:
        raise RunnerError(
            f"unknown dataset(s) {unknown}; configured: {sorted(registry['datasets'])}"
        )
    return selected


def _finalize_crashed_worker(run_json: Path, returncode: int) -> None:
    """Mark a run the worker could not finish itself (killed, crashed)."""
    record = read_run_json(run_json)
    if record is None or record.get("status") != "running":
        return
    step = _running_step(record)
    if step is not None:
        record["steps"][step].update(status="failed", finished_at=now())
    if returncode < 0:
        name = signal.Signals(-returncode).name
        message = f"dataset worker terminated by {name}"
        if name == "SIGKILL":
            message += " (possibly the out-of-memory killer)"
    else:
        message = f"dataset worker exited with code {returncode} without finishing"
    _finish(
        record,
        "failed",
        {
            "type": "WorkerTerminated",
            "message": message,
            "message_redacted": False,
            "step": step,
        },
    )
    write_run_json(run_json, record)


def _record_setup_failure(run_json: Path, dataset: str, exc: Exception) -> None:
    """Safe failed record for a setup error; never leaves an earlier success."""
    record = read_run_json(run_json)
    if not record or "steps" not in record:
        record = {
            "schema_version": SCHEMA_VERSION,
            "runner": "scripts/run_datasets.py",
            "dataset": dataset,
            "status": "running",
            "started_at": now(),
            "steps": {},
        }
    for step in record["steps"].values():
        if step.get("status") == "running":
            step.update(status="failed", finished_at=now())
    if "started_at" not in record or record.get("status") != "running":
        record["started_at"] = now()
    _finish(record, "failed", safe_error(exc, "setup"))
    run_json.parent.mkdir(parents=True, exist_ok=True)
    write_run_json(run_json, record)


def _invalidate(run_json: Path, previous: dict, reason: str) -> None:
    """Replace an earlier run record before its outputs are touched.

    Written atomically, so an interruption at any later point leaves a
    non-successful record and --resume reruns the dataset.
    """
    write_run_json(
        run_json,
        {
            "schema_version": SCHEMA_VERSION,
            "runner": "scripts/run_datasets.py",
            "dataset": previous.get("dataset"),
            "status": "invalidated",
            "invalidated_at": now(),
            "reason": reason,
            "previous_status": previous.get("status"),
        },
    )


def _process_dataset(
    *,
    dataset: str,
    position: str,
    entry: dict,
    args: argparse.Namespace,
    plan: ResourcePlan,
    thread_env: dict[str, str],
    output_root: Path,
    data_root: Path,
    config_dir: Path,
    launcher: Launcher,
) -> str:
    """Run (or skip) one dataset; returns its final status."""
    dataset_root = output_root / dataset
    run_json = dataset_root / "run.json"
    previous = read_run_json(run_json)

    source = source_path(data_root, entry)
    rendered = render_step_configs(dataset, entry, source, config_dir)
    inputs = run_inputs(entry, source, rendered)

    rerun: dict[str, object] | None = None
    if previous is not None:
        previous_status = previous.get("status")
        changed: list[str] = []
        if args.overwrite:
            reason = "overwrite requested"
        elif previous_status == "success":
            changed = changed_inputs(previous.get("inputs"), inputs)
            if not changed:
                print(
                    f"[{position}] {dataset}: skipped (successful run in run.json)",
                    flush=True,
                )
                return "skipped"
            reason = "configuration changed"
            print(
                f"[{position}] {dataset}: previous successful run configuration "
                f"changed ({', '.join(changed)}); rerunning",
                flush=True,
            )
        else:
            reason = f"previous run {previous_status}"
        rerun = {
            "reason": reason,
            "changed_inputs": changed,
            "previous_status": previous_status,
            "previous_started_at": previous.get("started_at"),
            "previous_inputs": previous.get("inputs"),
        }
        # Invalidate first: from here on run.json never claims success.
        _invalidate(run_json, previous, reason)

    for owned in ("project", "config"):
        shutil.rmtree(dataset_root / owned, ignore_errors=True)
    dataset_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "logs" / f"{dataset}.log").open("a", encoding="utf-8") as log:
        log.write(f"\n===== run started {now()} =====\n")

    step_configs = write_step_configs(rendered, dataset_root / "config")
    record = new_run_record(
        dataset=dataset,
        entry=entry,
        dataset_root=dataset_root,
        plan=plan,
        config_files=[config_dir / "datasets.yml", *step_configs.values()],
        source=source,
        inputs=inputs,
        rerun=rerun,
    )
    if not source.is_dir():
        _finish(
            record,
            "failed",
            safe_error(RunnerError(f"source directory not found: {source}"), None),
        )
        write_run_json(run_json, record)
        return "failed"

    write_run_json(run_json, record)
    returncode = launcher(
        worker_command(
            dataset=dataset, output_root=output_root, position=position, plan=plan
        ),
        {**os.environ, **thread_env},
    )
    _finalize_crashed_worker(run_json, returncode)
    gc.collect()
    final = read_run_json(run_json) or {}
    return "success" if final.get("status") == "success" else "failed"


def _check_no_overlap(output_root: Path, roots: Sequence[Path]) -> None:
    """The output root must neither contain nor lie inside source data."""
    for root in roots:
        if _inside(output_root, root) or _inside(root, output_root):
            raise RunnerError(
                f"--output-root {output_root} overlaps source data {root}; "
                "use a separate output directory"
            )


def run(args: argparse.Namespace, launcher: Launcher = launch_worker) -> int:
    plan = plan_resources(
        resource_fraction=args.resource_fraction,
        cpu_fraction=args.cpu_fraction,
        memory_fraction=args.memory_fraction,
        threads=args.threads,
        enforce_memory=args.enforce_memory,
    )
    if plan.memory_hard_limit_requested and shutil.which("systemd-run") is None:
        raise RunnerError("--enforce-memory needs systemd-run, which is not available")
    thread_env = thread_environment(plan.threads)
    os.environ.update(thread_env)

    config_dir = Path(args.config_dir).expanduser().resolve()
    registry = load_dataset_registry(config_dir)
    datasets = select_datasets(args, registry)
    output_root = Path(args.output_root).expanduser().resolve()
    if _inside(output_root, REPO_ROOT):
        raise RunnerError(f"--output-root must be outside the repository {REPO_ROOT}")
    data_root = resolve_data_root(args.data_root, registry)
    _check_no_overlap(
        output_root,
        [
            data_root.resolve(),
            *(source_path(data_root, registry["datasets"][d]) for d in datasets),
        ],
    )

    # Without --resume/--overwrite, never touch earlier outputs.
    if not (args.resume or args.overwrite):
        existing = [d for d in datasets if (output_root / d).exists()]
        if existing:
            raise RunnerError(
                f"output already exists for {existing} below {output_root}; "
                "use --resume to skip successful datasets or --overwrite to rerun them"
            )

    print(format_resources(plan))
    print(f"Datasets: {' '.join(datasets)}")
    print(f"Output root: {output_root}", flush=True)
    (output_root / "logs").mkdir(parents=True, exist_ok=True)

    results: dict[str, str] = {}
    for index, dataset in enumerate(datasets, start=1):
        position = f"{index}/{len(datasets)}"
        log_path = output_root / "logs" / f"{dataset}.log"
        try:
            status = _process_dataset(
                dataset=dataset,
                position=position,
                entry=registry["datasets"][dataset],
                args=args,
                plan=plan,
                thread_env=thread_env,
                output_root=output_root,
                data_root=data_root,
                config_dir=config_dir,
                launcher=launcher,
            )
        except Exception as exc:
            # Unexpected parent-side setup failure: record it safely, go on.
            logger.exception("setup of dataset %s failed", dataset)
            _record_setup_failure(output_root / dataset / "run.json", dataset, exc)
            status = "failed"
        results[dataset] = status
        if status in {"success", "skipped"}:
            if status == "success":
                final = read_run_json(output_root / dataset / "run.json") or {}
                duration = _fmt_duration(final.get("duration_seconds"))
                print(f"[{position}] {dataset}: success ({duration})", flush=True)
            continue
        final = read_run_json(output_root / dataset / "run.json") or {}
        error = final.get("error") or {}
        where = f" in {error['step']}" if error.get("step") else ""
        print(
            f"[{position}] {dataset}: FAILED ({error.get('type', 'unknown')}{where};"
            f" see {log_path})",
            flush=True,
        )
        if args.stop_on_error:
            for later in datasets[index:]:
                results[later] = "not started"
            print("Stopping after the first failure (--stop-on-error).", flush=True)
            break

    print("Summary:")
    for dataset in datasets:
        print(f"  {dataset:<10} {results.get(dataset, 'not started')}")
    failed = [d for d in datasets if results.get(d) not in {"success", "skipped"}]
    return 1 if failed else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run WeavEHR extraction and concept generation for several datasets, "
            "sequentially. RICU/YAIB validation is not part of this script."
        )
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--dataset", help="run one dataset")
    selection.add_argument(
        "--datasets", nargs="+", help="run these datasets, in this order"
    )
    selection.add_argument(
        "--all",
        action="store_true",
        help=f"run all validation datasets ({' '.join(VALIDATION_DATASETS)})",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="output directory, outside the repository (default: %(default)s)",
    )
    parser.add_argument(
        "--data-root",
        help="directory containing the source datasets (or WEAVEHR_DATA_ROOT, or datasets.yml)",
    )
    parser.add_argument(
        "--config-dir",
        default=str(DEFAULT_CONFIG_DIR),
        help="directory with datasets.yml and step templates (default: %(default)s)",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--resume",
        action="store_true",
        help="skip datasets whose run.json records success; rerun all others from scratch",
    )
    mode.add_argument(
        "--overwrite",
        action="store_true",
        help="rerun every selected dataset from scratch, discarding earlier outputs",
    )
    parser.add_argument(
        "--stop-on-error", action="store_true", help="stop after the first failure"
    )
    parser.add_argument(
        "--resource-fraction",
        type=float,
        default=DEFAULT_RESOURCE_FRACTION,
        help="fraction of CPUs and memory to use (default: %(default)s)",
    )
    parser.add_argument("--cpu-fraction", type=float, help="override the CPU fraction")
    parser.add_argument(
        "--memory-fraction", type=float, help="override the memory fraction"
    )
    parser.add_argument(
        "--threads", type=int, help="explicit thread count (overrides fractions)"
    )
    parser.add_argument(
        "--enforce-memory",
        action="store_true",
        help="run each dataset worker in a systemd-run --user --scope with MemoryMax=budget",
    )
    # Internal: one dataset worker, started by the runner itself.
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--position", default="1/1", help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None, launcher: Launcher = launch_worker) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.worker:
            if not args.dataset:
                raise RunnerError("--worker needs --dataset")
            return worker_main(
                args.dataset,
                Path(args.output_root).expanduser().resolve(),
                args.position,
            )
        return run(args, launcher)
    except RunnerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrupted; rerun with --resume to continue.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
