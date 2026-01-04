import json
import logging
from dataclasses import dataclass
from logging.config import fileConfig
from pathlib import Path

import simple_parsing
from tqdm.contrib.logging import logging_redirect_tqdm

from scicodeqa.core import GitHubIssueExperiment, InferenceArgs

fileConfig("logging.ini")
logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class IssueClassificationArgs(InferenceArgs):
    db_path: Path = Path("out") / "data_collection" / "github_crawl" / "ric.json"
    issue_ids_from_file: Path | None = None
    prompt: str = "github_issue_classification_discrapency"
    dir_prefix: str = "github_issues"
    output_base_dir: Path = Path("out") / "data_collection" / "github_classification"
    shuffle: bool = False
    seed: int | None = None
    issue_id: int | None = None
    add_comments: bool = False
    add_contributors: bool = False
    slice_size: int | None = None
    slice_index: int | None = None

    def __post_init__(self):
        super().__post_init__()

        exp_dir_name = self.get_exp_dir_name(self.output_base_dir)
        self.output_dir = self.output_base_dir / exp_dir_name
        logger.info(f"Output dir: {self.output_dir}")

        self.llm_kwargs = {}
        if self.shuffle and self.seed is None:
            raise ValueError("seed is required when shuffle is True")

        if self.issue_ids_from_file is not None:
            logger.info(f"Loading issue ids from file: {self.issue_ids_from_file}")
            if self.issue_ids_from_file.suffix == ".json":
                with open(self.issue_ids_from_file, "r") as f:
                    self.issue_ids = list(map(int, json.load(f).keys()))
            elif self.issue_ids_from_file.suffix == ".jsonl":
                self.issue_ids = []
                with open(self.issue_ids_from_file, "r") as f:
                    for line in f:
                        issue_id = int(json.loads(line)["issue_id"])
                        self.issue_ids.append(issue_id)
            else:
                raise ValueError(f"Unsupported file type: {self.issue_ids_from_file}")
        else:
            self.issue_ids = None

        self.iterator_kwargs = {
            "issue_ids": self.issue_ids,
            "shuffle": self.shuffle,
            "seed": self.seed,
            "add_comments": self.add_comments,
            "add_contributors": self.add_contributors,
            "slice_size": self.slice_size,
            "slice_index": self.slice_index,
        }


if __name__ == "__main__":
    with logging_redirect_tqdm(loggers=[logging.root, logging.getLogger("scicodeqa")]):
        args, unknown_args = simple_parsing.parse_known_args(IssueClassificationArgs)
        logger.info(f"IssueClassificationArgs: {args}")
        if unknown_args:
            logger.warning(f"Unknown args: {unknown_args}")
        GitHubIssueExperiment(args)()
