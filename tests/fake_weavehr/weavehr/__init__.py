"""Minimal stand-in for the WeavEHR API used by scripts/run_datasets.py (tests only).

Controlled through environment variables:

- FAKE_WEAVEHR_RECORD: JSON-lines file receiving import and step events.
- FAKE_WEAVEHR_FAIL: "<dataset>:<step>" raises an error with a data-like message.
- FAKE_WEAVEHR_CRASH: "<dataset>:<step>" kills the worker process (like the OOM killer).
- FAKE_WEAVEHR_EMPTY: "<dataset>:<step>" writes no output (like missing source files).
- FAKE_WEAVEHR_SKIP_TABLE: extraction table that produces no output.
- FAKE_WEAVEHR_ZERO_EVENT: extraction event written with zero rows.
- FAKE_WEAVEHR_ZERO_CONCEPTS: "1" writes every concept file with zero rows.

Outputs mimic WeavEHR's layout with synthetic, non-medical content:
workspace/extraction/<dataset>/<version>/<table>/<EVENT>.parquet and
workspace/concept/<concept>/<version>/<dataset>.parquet.
"""

import json
import logging
import os
import signal
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Self

import yaml

THREAD_ENV_VARS = (
    "POLARS_MAX_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
# A value standing in for patient-level source data; must never reach run.json.
FAKE_SOURCE_VALUE = "PATIENT-4711"
# Tables (and events) "configured" for every dataset version.
TABLES = {
    "chartevents": ["CHART"],
    "visit_occurrence": ["VISIT_START", "VISIT_END"],
    "general": ["ICU_ADMISSION"],
    "observations": ["OBSERVATION"],
}
ROWS = 3


def _record(event: dict) -> None:
    path = os.environ.get("FAKE_WEAVEHR_RECORD")
    if path:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(
                json.dumps({**event, "pid": os.getpid(), "t": time.time()}) + "\n"
            )


# Like weavehr.logging.get_logger: a stdout handler on the non-propagating
# package logger, unless the package logger already has a handler.
_logger = logging.getLogger("weavehr")
if not _logger.handlers:
    _console = logging.StreamHandler(sys.stdout)
    _logger.addHandler(_console)
    _logger.setLevel(logging.INFO)
    _logger.propagate = False
logging.getLogger("weavehr.config.registry").info("Loaded configuration: fake.concept")

_record(
    {
        "event": "import",
        "threads": {name: os.environ.get(name) for name in THREAD_ENV_VARS},
        "polars_loaded_before": "polars" in sys.modules,
    }
)

# Imported only after recording the import-time state above.
import pyarrow as pa
import pyarrow.parquet as pq


def _write(path: Path, rows: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"x": pa.array(range(rows), type=pa.int64())}), path)


class _Registry:
    def filter(self, dataset: str, version: str, **_: object) -> list:
        return [
            SimpleNamespace(
                name=table, events=[SimpleNamespace(name=e) for e in events]
            )
            for table, events in TABLES.items()
        ]


dataset_config_registry = _Registry()


class WeavEHRProject:
    def __init__(self, path: Path, overwrite: bool = False) -> None:
        self.path = Path(path)
        for sub in ("datasets", "workspace", "configs"):
            (self.path / sub).mkdir(parents=True, exist_ok=True)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


class _Step:
    step = ""

    def __init__(self, project: WeavEHRProject, config: dict) -> None:
        self.project = project
        self.config = config

    @classmethod
    def load(cls, project: WeavEHRProject, config_path: Path) -> "_Step":
        return cls(project, yaml.safe_load(Path(config_path).read_text()))

    def _entry(self) -> dict:
        entries = (
            self.config["config"].get("data")
            or self.config["config"]["mapping_configs"]
        )
        return entries[0]

    def _write_outputs(self, dataset: str, version: str) -> None:
        workspace = self.project.path / "workspace"
        if self.step == "extraction":
            skip = os.environ.get("FAKE_WEAVEHR_SKIP_TABLE")
            zero = os.environ.get("FAKE_WEAVEHR_ZERO_EVENT")
            for table, events in TABLES.items():
                if table == skip:
                    continue
                for event in events:
                    path = workspace / "extraction" / dataset / version / table
                    _write(path / f"{event}.parquet", 0 if event == zero else ROWS)
        else:
            rows = 0 if os.environ.get("FAKE_WEAVEHR_ZERO_CONCEPTS") == "1" else ROWS
            concept = workspace / "concept"
            _write(concept / "heart_rate" / "1.0.0" / f"{dataset}.parquet", rows)
            # A concept that is legitimately empty for this dataset.
            _write(concept / "rare_concept" / "1.0.0" / f"{dataset}.parquet", 0)

    def run(self) -> None:
        entry = self._entry()
        dataset, version = entry["name"], str(entry["version"])
        _record({"event": "start", "dataset": dataset, "step": self.step})
        target = f"{dataset}:{self.step}"
        if os.environ.get("FAKE_WEAVEHR_CRASH") == target:
            os.kill(os.getpid(), signal.SIGKILL)
        if os.environ.get("FAKE_WEAVEHR_FAIL") == target:
            raise RuntimeError(
                f"could not parse value '{FAKE_SOURCE_VALUE}' in column hr"
            )
        if os.environ.get("FAKE_WEAVEHR_EMPTY") != target:
            self._write_outputs(dataset, version)
            rows = self.project.path / "workspace" / f"{self.step}-{dataset}-rows.txt"
            rows.write_text(f"row with {FAKE_SOURCE_VALUE}\n")
        time.sleep(0.05)
        _record({"event": "end", "dataset": dataset, "step": self.step})


class ExtractionStep(_Step):
    step = "extraction"


class ConceptStep(_Step):
    step = "concept"
