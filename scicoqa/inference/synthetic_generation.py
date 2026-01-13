import logging
from dataclasses import dataclass
from logging.config import fileConfig
from pathlib import Path

import simple_parsing
from tqdm.contrib.logging import logging_redirect_tqdm

from scicoqa.core import InferenceArgs, SyntheticDiscrepancyExperiment

fileConfig("logging.ini")
logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class SyntheticDiscrepancyInferenceArgs(InferenceArgs):
    prompt: str = "synthetic_discrepancy_generation_v2"
    data_config_file: Path = Path("config") / "data.yaml"
    output_base_dir: Path = Path("out") / "data_collection" / "synthetic_generation"
    dir_prefix: str = "synthetic_discrepancy"
    dir_suffix: str | None = None
    num_discrepancies: int | None = None
    data_config_section: str = "synthetic_discrepancies"
    paper_url_field: str = "arxiv_url"

    def __post_init__(self):
        super().__post_init__()

        exp_dir_name = self.get_exp_dir_name(self.output_base_dir)
        self.output_dir = self.output_base_dir / exp_dir_name

        self.llm_kwargs = {}
        self.iterator_kwargs = {}
        if self.num_discrepancies is not None:
            self.iterator_kwargs["num_discrepancies"] = self.num_discrepancies
        self.iterator_kwargs["data_config_section"] = self.data_config_section
        self.iterator_kwargs["paper_url_field"] = self.paper_url_field


if __name__ == "__main__":
    with logging_redirect_tqdm(loggers=[logging.root, logging.getLogger("scicoqa")]):
        args, unknown_args = simple_parsing.parse_known_args(
            SyntheticDiscrepancyInferenceArgs
        )
        logger.info(f"SyntheticDiscrepancyInferenceArgs: {args}")

        if unknown_args:
            logger.warning(f"Unknown args: {unknown_args}")

        SyntheticDiscrepancyExperiment(args)()
