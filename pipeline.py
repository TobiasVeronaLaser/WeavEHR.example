from pathlib import Path

from weavehr import WeavEHRProject, ExtractionStep, ConceptStep, ShardingStep
from weavehr import PersistenceStep
from weavehr.logging import configure_logging

configure_logging(level="DEBUG")

config_path: Path = Path.cwd() / "config"
project_path = Path.cwd() / "output" / "project"

with WeavEHRProject(project_path) as project:
    # Optional: persist the source columns selected by extraction.yml as Parquet files.
    # These files are independent artifacts and are not used by the ExtractionStep.
    persistence_step = PersistenceStep.load(project, config_path / "extraction.yml")
    persistence_step.run()

    extraction_step = ExtractionStep.load(project, config_path / "extraction.yml")
    extraction_step.run()

    concept_step = ConceptStep.load(project, config_path / "concept.yml")
    concept_step.run()

    sharding_step = ShardingStep.load(project, config_path / "sharding.yml")
    sharding_step.run()