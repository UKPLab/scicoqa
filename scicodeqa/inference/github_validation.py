import logging
from dataclasses import dataclass
from logging.config import fileConfig
from pathlib import Path

import simple_parsing
from tqdm.contrib.logging import logging_redirect_tqdm

from scicodeqa.core import DiscrepancyExperiment, InferenceArgs

fileConfig("logging.ini")
logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class DiscrepancyValidationArgs(InferenceArgs):
    discrepancy_file: Path
    db_path: Path = Path("out") / "data_collection" / "github_crawl" / "ric.json"
    prompt: str = "github_issue_classification_discrapency"
    dir_prefix: str = "dis_val"
    output_base_dir: Path = Path("out") / "data_collection" / "github_validation"
    replace_images_in_issue_text: bool = True
    add_comments: bool = False

    def __post_init__(self):
        super().__post_init__()

        exp_dir_name = self.get_exp_dir_name(self.output_base_dir)
        self.output_dir = self.output_base_dir / exp_dir_name
        logger.info(f"Output dir: {self.output_dir}")

        self.llm_kwargs = {}
        self.iterator_kwargs = {
            "replace_images": self.replace_images_in_issue_text,
            "add_comments": self.add_comments,
        }


if __name__ == "__main__":
    with logging_redirect_tqdm(loggers=[logging.root, logging.getLogger("scicodeqa")]):
        args, unknown_args = simple_parsing.parse_known_args(DiscrepancyValidationArgs)
        logger.info(f"DiscrepancyValidationArgs: {args}")
        if unknown_args:
            logger.warning(f"Unknown args: {unknown_args}")
        DiscrepancyExperiment(args)()
