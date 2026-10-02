from pathlib import Path

from weavehr import ExtractionStep, WeavEHRProject
from weavehr.logging import configure_logging

configure_logging(level="DEBUG")

config_path = Path.cwd() / "config" 
project_path = Path.cwd() / "output" / "project"
with WeavEHRProject(project_path, overwrite=True) as project:

    extraction_step = ExtractionStep.load(project, config_path / "extraction.yml")
    extraction_step.run()

    # concept_step = ConceptStep.load(project, config_path / "concept.yml")
    # concept_step.run()

    # sharding_step = ShardingStep.load(project, config_path / "sharding.yml")
    # sharding_step.run()