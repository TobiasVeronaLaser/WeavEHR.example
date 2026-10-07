# WeavEHR.example

Execution and use-case repository for [WeavEHR](https://github.com/aidh-ms/WeavEHR).
WeavEHR is the pipeline library; this repository runs it on concrete datasets.

> `pipeline.py`, `pipeline.ipynb` and `config/{extraction,concept,sharding}.yml`
> still use the legacy OpenICU API (`open_icu`) and do not run against current
> WeavEHR. Use `scripts/run_datasets.py` for dataset runs.

## Running datasets: `scripts/run_datasets.py`

Runs the WeavEHR pipeline for several datasets unattended, replacing long
multi-dataset notebook runs. Per dataset it runs the current WeavEHR steps

1. **Extraction** (`ExtractionStep`)
2. **Concept generation** (`ConceptStep`)

and produces the WeavEHR project that the `weavehr-yaib` RICU/YAIB validation
reads later. Sharding is not run: `weavehr-yaib` uses only the extraction and
concept outputs.

**Not part of this script:** RICU, YAIB, the `weavehr-yaib` comparison and
concept coverage updates. Those are later, separate steps.

### Setup

Run with a Python environment in which WeavEHR is installed (Python ≥ 3.13),
from the root of this repository. Source data is not part of this repository;
point the runner at it with `--data-root` (or `WEAVEHR_DATA_ROOT`, or
`data_root` in `config/ricu-validation/datasets.yml`).

Dataset versions and source directories (relative to the data root) are in
[`config/ricu-validation/datasets.yml`](config/ricu-validation/datasets.yml):

| Dataset | WeavEHR version | Default source (below the data root) |
|---|---|---|
| `aumc` | 1.5.0 | `aumc/1.5.0` (AMSTEL OMOP export; adjust) |
| `eicu-crd` | 2.0 | `physionet.org/files/eicu-crd/2.0` |
| `hirid` | 1.1.1 | `physionet.org/files/hirid/1.1.1` |
| `mimic-iv` | 2.2 | `physionet.org/files/mimiciv/2.2` |

Each source directory must have the layout expected by the WeavEHR table
configs of that dataset version. Versions must be quoted strings (`"1.10"`, not
`1.10`). Tables that may legitimately be absent can be listed per dataset under
`optional_extraction_tables` (none by default). The same file sets the concept-step
`extension_columns` that `weavehr-yaib` needs (`dataset_version`,
`source_code`, AUMC `visit_occurrence_id`) and the extraction events required
for stay windows (AUMC `VISIT_START`/`VISIT_END`, HiRID
`ICU_ADMISSION`/`OBSERVATION`). The step configs in the same directory are
templates in the current WeavEHR schema; the runner fills in the dataset and
writes them to `<dataset>/config/`.

### Examples

Single dataset:

```bash
python scripts/run_datasets.py \
    --dataset mimic-iv \
    --data-root ~/physionet \
    --output-root ~/output/WeavEHR.example/ricu-validation
```

Multiple datasets (all four: `--all`):

```bash
python scripts/run_datasets.py \
    --datasets aumc eicu-crd hirid mimic-iv \
    --data-root ~/physionet \
    --output-root ~/output/WeavEHR.example/ricu-validation \
    --resume
```

Custom resource fraction:

```bash
python scripts/run_datasets.py \
    --datasets aumc eicu-crd hirid mimic-iv \
    --data-root ~/physionet \
    --resource-fraction 0.60
```

Explicit threads:

```bash
python scripts/run_datasets.py \
    --dataset mimic-iv \
    --data-root ~/physionet \
    --threads 8
```

`--output-root` defaults to `~/output/WeavEHR.example/ricu-validation`. It must
be outside this repository and must neither lie inside nor contain the data root
or any selected dataset's source directory.

### Sequential execution

Datasets run strictly one after another, never concurrently. Each dataset runs
in its own worker process, started only after the previous one has finished:
its memory is released completely, and a crash or out-of-memory kill fails only
that dataset.

### Resources

By default the runner uses about **75%** of the resources available to the
process:

- **CPU:** CPUs from the process affinity (`os.sched_getaffinity`), capped by a
  cgroup CPU quota if one exists; fallback `os.process_cpu_count()` /
  `os.cpu_count()`. Threads = `max(1, floor(cpus × fraction))`, e.g. 32 → 24.
  `POLARS_MAX_THREADS`, `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
  `MKL_NUM_THREADS` and `NUMEXPR_NUM_THREADS` are set for the worker before
  WeavEHR/Polars are imported.
- **Memory:** the capacity is a finite cgroup/container memory limit if one
  applies, otherwise physical RAM (never the fluctuating `MemAvailable`). The
  budget is capacity × fraction.

Options: `--resource-fraction` (default 0.75), `--cpu-fraction` and
`--memory-fraction` (override one side), `--threads N` (explicit, overrides the
CPU fraction). Fractions must be in (0, 1].

The memory budget is **reported, not enforced**, unless a hard limit actually
applies. The startup report and `run.json` distinguish:

- `memory_capacity_bytes`: detected capacity;
- `memory_budget_bytes`: calculated budget;
- `memory_hard_limit_bytes` / `memory_hard_limit_enforced`: a cgroup limit
  actually in force for the dataset worker (observed in the worker).

Hard limits are best enforced by cgroups/systemd. `--enforce-memory` runs each
dataset worker in `systemd-run --user --scope -p MemoryMax=<budget>`; whether
the limit took effect is recorded per dataset in `run.json`. Alternatively,
limit the whole runner externally, choosing values for the machine at hand:

```bash
systemd-run --user --scope \
    -p MemoryMax=<limit> \
    -p CPUQuota=<percent> \
    python scripts/run_datasets.py --all --data-root ~/physionet
```

Startup report:

```text
Resource configuration:
  CPUs available:       32 (affinity)
  Threads selected:     24
  CPU fraction:         75%
  Memory capacity:      64.0 GiB (host)
  Memory budget:        48.0 GiB (75%)
  Hard memory enforced: no (budget is reported, not enforced)
  Dataset execution:    sequential
```

### Outputs

```text
~/output/WeavEHR.example/ricu-validation/
    aumc/
        project/        # WeavEHR project (workspace/, datasets/, configs/)
        config/         # step configs used for this dataset
        run.json
    eicu-crd/ ...
    hirid/ ...
    mimic-iv/ ...
    logs/
        aumc.log  eicu-crd.log  hirid.log  mimic-iv.log
```

The console stays concise (`[1/4] aumc: extraction`, `[1/4] aumc: success (12m 31s)`);
WeavEHR's own logging and full tracebacks go to `logs/<dataset>.log`. These logs
stay in the private execution environment.

### Resume, overwrite, failures

- A dataset is complete **only** if its `run.json` records `"status": "success"`;
  existing directories or parquet files are never taken as success.
- Without `--resume`/`--overwrite`, existing output for a selected dataset is an
  error (nothing is overwritten).
- `--resume`: skip a dataset only if its `run.json` records success **and** the
  recorded `inputs` equal the current ones: `dataset_version`, resolved
  `source_path`, SHA-256 of the generated extraction and concept configs,
  `required_extraction_events` and `optional_extraction_tables`. Otherwise
  (configuration changed, failed, interrupted, invalidated or unrecorded runs)
  the dataset is rerun from scratch, e.g.
  `mimic-iv: previous successful run configuration changed (dataset_version, ...); rerunning`.
  Timestamps, resources and software versions are not compared.
- `--overwrite`: rerun every selected dataset from scratch, discarding its
  earlier output. `--resume` and `--overwrite` are mutually exclusive.
- A rerun first replaces the earlier `run.json` atomically by an
  `"invalidated"` record, and only then discards `project/` and `config/`; an
  interruption or setup error afterwards can never leave a stale `success`. The
  new `run.json` records why under `rerun` (reason, changed inputs, previous
  status and inputs). The log is appended.
- An unexpected error while preparing a dataset is recorded as a failed
  `run.json` (step `setup`) and the next dataset runs.
- A failing dataset is recorded as failed and the next dataset runs;
  `--stop-on-error` stops after the first failure.
- Output completeness is checked, because WeavEHR skips tables whose source
  files are missing ("Skipping table" in the log) and still finishes:
  - after extraction, every table that WeavEHR configures for the dataset
    version must have produced output (except `optional_extraction_tables`);
    the required stay events must exist and contain rows; tables without rows
    are reported in `run.json`;
  - after concept generation, concept files may be empty individually (a
    concept can be absent from a dataset), but the run fails if there are none
    or all of them are empty.

  Row counts come from the Parquet footers (`pyarrow.parquet.read_metadata`);
  no rows are read.
- The exit code is non-zero if any selected dataset failed.

### `run.json`

Machine-readable metadata, safe to copy to other environments: it contains no
patient-level data, no rows, no tracebacks. Fields: `schema_version`,
`dataset`, `dataset_version`, `status` (`running`/`success`/`failed`; an
earlier run being replaced is briefly `invalidated`),
`started_at`, `finished_at`, `duration_seconds`, `output`, `resources`,
`steps` (`extraction`, `concept` with status and timestamps), `outputs` (file,
table and row counts; table names), `inputs` (compared by `--resume`), `rerun`,
`software` (Python, WeavEHR version/commit, this repository's commit),
`configuration` (source path, config files with SHA-256) and `error`. Error
messages of library exceptions can quote source values, so they are redacted
(`message_redacted: true`) and only the exception type and step are kept; the
details are in the dataset log.

### Tests

The tests use a fake `weavehr` package and need no medical data:

```bash
python -m pytest tests
```
