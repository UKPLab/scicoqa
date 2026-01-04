import logging
from dataclasses import dataclass
from logging.config import fileConfig
from pathlib import Path
from typing import Literal

import simple_parsing
from tqdm.contrib.logging import logging_redirect_tqdm

from scicodeqa.core import InferenceArgs, ReproducibilityReportExperiment

fileConfig("logging.ini")
logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class ReproducibilityReportsInferenceArgs(InferenceArgs):
    prompt: str = "reproducibility_report_discrepancy_extraction"
    data_config_file: Path = Path("config") / "data.yaml"
    output_base_dir: Path = (
        Path("out") / "data_collection" / "reproducibility_extraction"
    )
    dir_prefix: str = "reproducibility_reports"
    iterate_over: Literal["paper", "reproducibility_paper"] = "paper"

    def __post_init__(self):
        super().__post_init__()

        exp_dir_name = self.get_exp_dir_name(self.output_base_dir)
        self.output_dir = self.output_base_dir / exp_dir_name

        self.llm_kwargs = {}
        self.iterator_kwargs = {}


if __name__ == "__main__":
    with logging_redirect_tqdm(loggers=[logging.root, logging.getLogger("scicodeqa")]):
        args, unknown_args = simple_parsing.parse_known_args(
            ReproducibilityReportsInferenceArgs
        )
        logger.info(f"ReproducibilityReportsInferenceArgs: {args}")

        if unknown_args:
            logger.warning(f"Unknown args: {unknown_args}")

        ReproducibilityReportExperiment(args)()
