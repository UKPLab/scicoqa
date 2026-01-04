import logging
from dataclasses import dataclass
from logging.config import fileConfig
from pathlib import Path

import simple_parsing
from tqdm.contrib.logging import logging_redirect_tqdm

from scicodeqa.core import (
    InferenceArgs,
    ReproducibilityReportDiscrepancyVerificationExperiment,
)

fileConfig("logging.ini")
logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class ReproducibilityReportDiscrepancyVerificationArgs(InferenceArgs):
    predictions_and_classifications_file: Path = (
        Path("out")
        / "data_collection"
        / "reproducibility_extraction"
        / "reproducibility_reports-012"
        / "predictions_and_classifications.jsonl"
    )
    data_config_file: Path = Path("config") / "data.yaml"
    prompt: str = "reproducibility_report_discrepancy_verification"
    dir_prefix: str = "reproducibility_report_discrepancy_verification"
    output_base_dir: Path = (
        Path("out") / "data_collection" / "reproducibility_validation"
    )

    def __post_init__(self):
        super().__post_init__()

        exp_dir_name = self.get_exp_dir_name(self.output_base_dir)
        self.output_dir = self.output_base_dir / exp_dir_name
        logger.info(f"Output dir: {self.output_dir}")

        self.llm_kwargs = {}
        self.iterator_kwargs = {}


if __name__ == "__main__":
    with logging_redirect_tqdm(loggers=[logging.root, logging.getLogger("scicodeqa")]):
        args, unknown_args = simple_parsing.parse_known_args(
            ReproducibilityReportDiscrepancyVerificationArgs
        )
        logger.info(f"ReproducibilityReportDiscrepancyVerificationArgs: {args}")
        if unknown_args:
            logger.warning(f"Unknown args: {unknown_args}")
        ReproducibilityReportDiscrepancyVerificationExperiment(args)()
