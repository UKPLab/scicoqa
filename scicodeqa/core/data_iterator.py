import json
import logging
import random
from datetime import date, datetime
from logging.config import fileConfig
from pathlib import Path
from typing import Generator, Literal

import yaml
from tqdm import tqdm

from scicodeqa.core import CodeLoader
from scicodeqa.core.mistral_ocr import MistralOCR
from scicodeqa.core.prompt import Prompt
from scicodeqa.core.token_counter import TokenCounter
from scicodeqa.github.db import GitHubDB
from scicodeqa.github.utils import comments_to_text, issue_to_text

fileConfig("logging.ini")
logger = logging.getLogger(__name__)


# TODO: read this from config
MAX_TOKENS_BUFFER = 0.9


class BaseIterator:
    def __init__(
        self,
        prompt: Prompt,
        model: str,
        tokenizer: str,
        model_max_tokens: int,
        debug: bool = False,
        debug_break_after: int = 1,
    ):
        self.prompt = prompt
        self.model = model
        self.model_max_tokens = model_max_tokens
        self.debug = debug
        self.debug_break_after = debug_break_after
        self.token_counter = TokenCounter(model=tokenizer)

        self.mistral_ocr = MistralOCR()

    def __call__(self, **kwargs):
        return self.__iter__(**kwargs)

    def __iter__(self) -> Generator[tuple[dict, str], None, None]:
        raise NotImplementedError()

    def prepare_prompt(self, paper: dict, **kwargs) -> tuple[dict, dict]:
        """
        Prepare prompt kwargs from paper data and additional kwargs.

        Args:
            paper: Dictionary with paper metadata (id, arxiv_url, github_url, etc.)
            **kwargs: Additional arguments including:
                - target_date: Date for arxiv version and code base checkout
                - paper_url_field: Field name for paper URL (default: "arxiv_url")
                - github_url_field: Field name for github URL (default: "github_url")
                - Other prompt variables to be passed through

        Returns:
            Dictionary of kwargs ready to be passed to the prompt template
        """
        prompt_kwargs = {}

        # Internal kwargs that shouldn't be passed to prompt or generate warnings
        internal_kwargs = {
            "target_date",
            "paper_url_field",
            "github_url_field",
            "replace_images",
            "add_comment",
            "add_contributors",
        }

        target_date = kwargs.get("target_date")
        # Convert datetime.date objects to ISO string format for compatibility
        if isinstance(target_date, (date, datetime)):
            target_date = target_date.isoformat()

        if "paper" in self.prompt.prompt_vars:
            paper_url_field = kwargs.get("paper_url_field", "arxiv_url")
            paper_text, paper_url = self.mistral_ocr(
                paper[paper_url_field], target_date=target_date
            )
            if "appendix" in paper:
                appendix_text, appendix_url = self.mistral_ocr(paper["appendix"])
                paper_text += "\n\n" + appendix_text
            else:
                appendix_url = None
            prompt_kwargs["paper"] = paper_text

        if "code" in self.prompt.prompt_vars:
            intermediate_prompt = self.prompt(**prompt_kwargs)
            tokens_intermediate_prompt = self.token_counter(intermediate_prompt)
            remaining_code_tokens = (
                self.model_max_tokens * MAX_TOKENS_BUFFER - tokens_intermediate_prompt
            )
            github_url_field = kwargs.get("github_url_field", "github_url")
            code_loader = CodeLoader(paper[github_url_field], target_date=target_date)
            prompt_kwargs["code"] = code_loader.get_code_prompt(
                token_counter=self.token_counter, max_tokens=remaining_code_tokens
            )

        # Add other kwargs that are prompt variables
        for k, v in kwargs.items():
            # Skip internal kwargs and those starting with "_"
            if k.startswith("_") or k in internal_kwargs:
                continue
            if k in self.prompt.prompt_vars:
                prompt_kwargs[k] = v
            else:
                logger.warning(
                    f"'{k}' provided to fill prompt, but not in prompt variables"
                )

        metadata = {
            "paper_url": paper_url,
            "appendix_url": appendix_url,
            "repo_permalink": code_loader.repo_permalink,
        }

        return prompt_kwargs, metadata

    def should_break_debug(self, processed_count: int) -> bool:
        """Check if we should break due to debug mode limits."""
        if self.debug and processed_count >= self.debug_break_after:
            logger.info(f"Breaking after {processed_count} items in debug mode")
            return True
        return False

class SyntheticDiscrepancyIterator(BaseIterator):
    def __init__(
        self,
        data_config_path: Path,
        output_dir: Path,
        prompt: Prompt,
        model: str,
        tokenizer: str,
        model_max_tokens: int,
        debug: bool = False,
        debug_break_after: int = 1,
        data_config_section: str = "synthetic_discrepancies",
    ):
        super().__init__(
            prompt,
            model,
            tokenizer,
            model_max_tokens,
            debug,
            debug_break_after,
        )
        self.data_config_path = data_config_path
        with open(self.data_config_path, "r") as f:
            self.data_config = yaml.safe_load(f)

        self.data_config_section = data_config_section
        self.output_dir = output_dir
        self.already_processed = set()
        if (self.output_dir / "generations.jsonl").exists():
            with open(self.output_dir / "generations.jsonl", "r") as f:
                for line in f:
                    generation = json.loads(line)
                    self.already_processed.add(generation["paper"]["id"])

    @property
    def data_config_path(self):
        return self._data_config_path

    @data_config_path.setter
    def data_config_path(self, value: Path):
        if not value.exists():
            raise FileNotFoundError(f"Data config file {value} not found")
        self._data_config_path = value

    def __iter__(
        self, show_progress: bool = True, **kwargs
    ) -> Generator[tuple[dict, Prompt, tqdm], None, None]:
        # Allow kwargs to override the data_config_section if needed
        data_config_section = kwargs.pop(
            "data_config_section", self.data_config_section
        )

        pbar = tqdm(
            total=len(self.data_config[data_config_section]),
            ncols=100,
            disable=not show_progress,
        )
        processed_count = 0
        for paper in self.data_config[data_config_section].values():
            if paper["id"] in self.already_processed:
                logger.info(f"Skipping {paper['id']} because it already exists")
                pbar.update(1)
                continue
            # Extract num_discrepancies and target_date from paper config
            # Allow kwargs to override paper config for num_discrepancies
            num_discrepancies = kwargs.get("num_discrepancies")
            if num_discrepancies is None:
                num_discrepancies = paper.get("num_discrepancies", 3)

            # Prepare kwargs for prepare_prompt (exclude num_discrepancies to
            # avoid duplication)
            prepare_kwargs = {
                k: v for k, v in kwargs.items() if k != "num_discrepancies"
            }
            prepare_kwargs["target_date"] = paper.get("target_date")

            prompt_kwargs, prompt_metadata = self.prepare_prompt(
                paper, num_discrepancies=num_discrepancies, **prepare_kwargs
            )
            prompt = self.prompt(**prompt_kwargs)
            paper["prompt_metadata"] = prompt_metadata
            yield paper, prompt, pbar
            pbar.update(1)
            processed_count += 1

            if self.should_break_debug(processed_count):
                break
        pbar.close()
class DiscrepancyIterator(BaseIterator):
    def __init__(
        self,
        discrepancy_file: Path,
        output_dir: Path,
        prompt: Prompt,
        model: str,
        tokenizer: str,
        model_max_tokens: int,
        db_path: Path,
        debug: bool = False,
        debug_break_after: int = 1,
    ):
        super().__init__(
            prompt,
            model,
            tokenizer,
            model_max_tokens,
            debug,
            debug_break_after,
        )
        self.discrepancy_file = discrepancy_file
        self.output_dir = output_dir
        logger.debug(f"DiscrepancyIterator output_dir: {self.output_dir}")

        # Initialize GitHub database (required)
        self.db = GitHubDB(db_path)

        # Load discrepancy issues from JSONL file
        self.discrepancy_issues = []
        with open(self.discrepancy_file, "r") as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    issue = json.loads(line)
                    self.discrepancy_issues.append(issue)

        logger.info(f"Loaded {len(self.discrepancy_issues)} discrepancy issues")

        # Track already processed issues
        self.already_processed = set()
        generations_file = self.output_dir / "generations.jsonl"
        if generations_file.exists():
            logger.info(f"Loading already processed issues from {generations_file}")
            with open(generations_file, "r") as f:
                for line in f:
                    generation = json.loads(line)
                    self.already_processed.add(
                        generation["discrepancy_issue"]["issue_id"]
                    )
            logger.info(f"Found {len(self.already_processed)} already processed issues")

    @property
    def discrepancy_file(self):
        return self._discrepancy_file

    @discrepancy_file.setter
    def discrepancy_file(self, value: Path):
        if not value.exists():
            raise FileNotFoundError(f"Discrepancy file {value} not found")
        self._discrepancy_file = value

    def __iter__(
        self,
        show_progress: bool = True,
        **kwargs,
    ) -> Generator[tuple[dict, str, tqdm], None, None]:
        # Process all issues from the discrepancy file
        issues_to_process = self.discrepancy_issues
        logger.info(f"Found {len(issues_to_process)} discrepancy issues to process")

        pbar = tqdm(total=len(issues_to_process), ncols=100, disable=not show_progress)
        processed_count = 0

        for discrepancy_issue in issues_to_process:
            issue_id = discrepancy_issue["issue_id"]

            if issue_id in self.already_processed:
                logger.info(
                    f"Skipping discrepancy issue {issue_id} "
                    "because it has already been processed"
                )
                pbar.update(1)
                continue

            paper_data = {
                "id": str(discrepancy_issue["issue_id"]),
                "arxiv_url": discrepancy_issue["paper_url"],
                "github_url": discrepancy_issue["github_url"],
            }

            # Load GitHub issue content and repository data from databas
            issue_data = self.db.get_issue_by_id(discrepancy_issue["issue_id"])
            issue_content = issue_to_text(
                issue_data, replace_images=kwargs.get("replace_images", False)
            )
            # Use the issue's created_at timestamp as target_date for repository
            # checkout
            issue_target_date = issue_data.get("created_at")
            if not issue_target_date:
                raise ValueError(
                    f"Issue {discrepancy_issue['issue_id']} has no created_at timestamp"
                )

            # Get the authoritative repository URL from the database
            repository_id = issue_data.get("repository_id")
            repo_data = self.db.get_repo_by_id(repository_id)
            github_url = repo_data.get("html_url")
            if not github_url:
                raise ValueError(f"Repository {repository_id} has no html_url")
            logger.debug(f"Using repository URL from database: {github_url}")

            logger.debug(
                f"Loaded issue content for {discrepancy_issue['issue_id']}, "
                f"created_at: {issue_target_date}"
            )

            paper_data["github_url"] = github_url
            kwargs["issue"] = issue_content
            kwargs["target_date"] = issue_target_date

            prompt_kwargs, prompt_metadata = self.prepare_prompt(paper_data, **kwargs)
            prompt = self.prompt(**prompt_kwargs)
            discrepancy_issue["prompt_metadata"] = prompt_metadata
            yield discrepancy_issue, prompt, pbar
            pbar.update(1)
            processed_count += 1

            if self.should_break_debug(processed_count):
                break

        pbar.close()


class GitHubIssueIterator(BaseIterator):
    def __init__(
        self,
        db_path: Path,
        output_dir: Path,
        prompt: Prompt,
        model: str,
        tokenizer: str,
        model_max_tokens: int,
        debug: bool = False,
        debug_break_after: int = 1,
    ):
        super().__init__(
            prompt,
            model,
            tokenizer,
            model_max_tokens,
            debug,
            debug_break_after,
        )
        self.db = GitHubDB(db_path)
        self.output_dir = output_dir
        logger.debug(f"GitHubIssueIterator output_dir: {self.output_dir}")

        self.already_processed = set()
        generations_file = self.output_dir / "generations.jsonl"
        if generations_file.exists():
            logger.info(f"Loading already processed issues from {generations_file}")
            with open(generations_file, "r") as f:
                for line in f:
                    generation = json.loads(line)
                    self.already_processed.add(generation["issue"]["id"])
            logger.info(f"Found {len(self.already_processed)} already processed issues")

    def __iter__(
        self,
        show_progress: bool = True,
        issue_ids: list[int] | None = None,
        shuffle: bool = False,
        seed: int | None = None,
        slice_size: int | None = None,
        slice_index: int | None = None,
        **kwargs,
    ) -> Generator[tuple[dict, str, tqdm], None, None]:
        if issue_ids is not None:
            issues = self.db.get_issues_by_ids(issue_ids)
        else:
            issues = self.db.get_all_issues()

        if slice_size is not None and slice_index is not None:
            issues = issues[slice_index * slice_size : (slice_index + 1) * slice_size]

        logger.info(f"Processing {len(issues)} issues")

        if shuffle and len(issues) > 1:
            logger.info(f"Shuffling {len(issues)} issues with {seed=}")
            rng = random.Random(seed)
            rng.shuffle(issues)

        pbar = tqdm(total=len(issues), ncols=100, disable=not show_progress)
        processed_count = 0

        for issue in issues:
            if issue["id"] in self.already_processed:
                logger.info(
                    f"Skipping issue {issue['id']} because has already been run"
                )
                pbar.update(1)
                continue

            issue_text = issue_to_text(issue, truncate_after=1024)
            prompt_kwargs = {"issue": issue_text}
            if kwargs.get("add_comment", False):
                comments = self.db.get_comments_by_issue_id(issue_id=issue["id"])
                if kwargs.get("add_contributors", False):
                    contributors = self.db.get_contributors(
                        repository_id=issue["repository_id"]
                    )
                else:
                    contributors = None
                comments = comments_to_text(comments, contributors=contributors)
                prompt_kwargs["comments"] = comments

            prompt = self.prompt(**prompt_kwargs)
            yield issue, prompt, pbar
            pbar.update(1)
            processed_count += 1

            if self.should_break_debug(processed_count):
                break
        pbar.close()


class ReproducibilityReportIterator(BaseIterator):
    def __init__(
        self,
        data_config_path: Path,
        output_dir: Path,
        prompt: Prompt,
        model: str,
        tokenizer: str,
        model_max_tokens: int,
        debug: bool = False,
        debug_break_after: int = 1,
        iterate_over: Literal["paper", "reproducibility_paper"] = "paper",
    ):
        super().__init__(
            prompt,
            model,
            tokenizer,
            model_max_tokens,
            debug,
            debug_break_after,
        )
        self.data_config_path = data_config_path
        with open(self.data_config_path, "r") as f:
            self.data_config = yaml.safe_load(f)

        self.output_dir = output_dir
        self.already_processed = set()
        if (self.output_dir / "generations.jsonl").exists():
            with open(self.output_dir / "generations.jsonl", "r") as f:
                for line in f:
                    generation = json.loads(line)
                    self.already_processed.add(generation["paper"]["id"])
        self.iterate_over = iterate_over

    def __iter__(
        self, show_progress: bool = True, **kwargs
    ) -> Generator[tuple[dict, Prompt, tqdm], None, None]:
        pbar = tqdm(
            total=len(self.data_config["reproducibility_reports"]),
            ncols=100,
            disable=not show_progress,
        )
        kwargs["paper_url_field"] = kwargs.get("paper_url_field", self.iterate_over)
        if (
            "code" in self.prompt.prompt_vars
            and self.iterate_over == "reproducibility_report"
        ):
            kwargs["github_url_field"] = kwargs.get(
                "github_url_field", "reproducibility_code"
            )
        else:
            kwargs["github_url_field"] = kwargs.get("github_url_field", "code")

        processed_count = 0
        for paper in self.data_config["reproducibility_reports"].values():
            if paper["id"] in self.already_processed:
                logger.info(f"Skipping {paper['id']} because it already exists")
                pbar.update(1)
                continue
            kwargs["target_date"] = paper.get("reproducibility_paper_date")
            prompt_kwargs, prompt_metadata = self.prepare_prompt(paper, **kwargs)
            prompt = self.prompt(**prompt_kwargs)
            paper["prompt_metadata"] = prompt_metadata
            yield paper, prompt, pbar
            pbar.update(1)
            processed_count += 1

            if self.should_break_debug(processed_count):
                break
        pbar.close()


class ReproducibilityReportDiscrepancyVerificationIterator(BaseIterator):
    def __init__(
        self,
        predictions_and_classifications_file: Path,
        data_config_path: Path,
        output_dir: Path,
        prompt: Prompt,
        model: str,
        tokenizer: str,
        model_max_tokens: int,
        debug: bool = False,
        debug_break_after: int = 1,
    ):
        super().__init__(
            prompt,
            model,
            tokenizer,
            model_max_tokens,
            debug,
            debug_break_after,
        )
        self.output_dir = output_dir

        # Load data config to get reproducibility_paper_date
        with open(data_config_path) as f:
            self.data_config = yaml.safe_load(f)

        # Validate and load predictions and classifications from merged file
        if not predictions_and_classifications_file.exists():
            raise FileNotFoundError(
                "Predictions and classifications file not found: "
                f"{predictions_and_classifications_file}"
            )

        # Load all entries and filter for "Actual discrepancy" classifications
        self.actual_discrepancies = []
        with open(predictions_and_classifications_file, "r") as f:
            for line in f:
                if line := line.strip():
                    entry = json.loads(line)
                    if entry.get("classification") != "Actual discrepancy":
                        continue
                    self.actual_discrepancies.append(entry)

        logger.info(
            f"Found {len(self.actual_discrepancies)} actual discrepancies to verify"
        )

        # Track already processed items
        self.already_processed = set()
        generations_file = output_dir / "generations.jsonl"
        if generations_file.exists():
            with open(generations_file, "r") as f:
                for line in f:
                    generation = json.loads(line)
                    self.already_processed.add(generation["classification_key"])
            logger.info(f"Found {len(self.already_processed)} already processed items")

    def __iter__(
        self,
        show_progress: bool = True,
        **kwargs,
    ) -> Generator[tuple[dict, str, tqdm], None, None]:
        # Process all actual discrepancies
        items_to_process = self.actual_discrepancies
        logger.info(f"Processing {len(items_to_process)} actual discrepancies")

        pbar = tqdm(total=len(items_to_process), ncols=100, disable=not show_progress)
        processed_count = 0

        for entry in items_to_process:
            # Create unique key for this discrepancy
            classification_key = f"{entry['id']}_discrepancy_{entry['discrepancy_id']}"

            if classification_key in self.already_processed:
                logger.info(
                    f"Skipping {classification_key} because it has already been "
                    "processed"
                )
                pbar.update(1)
                continue

            # Extract prediction_id from the entry
            prediction_id = entry["id"]

            # Find the corresponding paper in data_config to get date
            paper_entry = None
            for paper_id, paper_info in self.data_config.get(
                "reproducibility_reports", {}
            ).items():
                # Match by prediction_id (could be string or the id field)
                if str(paper_info.get("id")) == str(prediction_id):
                    paper_entry = paper_info
                    break

            if not paper_entry:
                logger.warning(
                    f"No paper found in data_config for prediction_id: {prediction_id}"
                )
                pbar.update(1)
                continue

            # Prepare the discrepancy text for the prompt
            discrepancy_text = f"{entry['discrepancy']}\n\n"
            if entry.get("evidence"):
                discrepancy_text += "**Evidence from Reproducibility Report:**\n"
                for i, evidence in enumerate(entry["evidence"], 1):
                    discrepancy_text += f"{i}. {evidence}\n\n"

            # Prepare paper data with fields expected by base class
            paper_data = {
                "id": str(prediction_id),
                "arxiv_url": entry.get("paper") or entry.get("arxiv_url"),
                "github_url": entry["code"],
            }

            # Add discrepancy_in_report to kwargs
            kwargs["discrepancy_in_report"] = discrepancy_text

            # Add target_date for CodeLoader to checkout code at reproducibility paper
            # date
            if "reproducibility_paper_date" in paper_entry:
                # Convert date object to string if needed
                target_date = paper_entry["reproducibility_paper_date"]
                if hasattr(target_date, "isoformat"):
                    target_date = target_date.isoformat()
                kwargs["target_date"] = target_date
                logger.debug(
                    f"Using reproducibility_paper_date: {kwargs['target_date']}"
                )

            prompt_kwargs, prompt_metadata = self.prepare_prompt(paper_data, **kwargs)
            prompt = self.prompt(**prompt_kwargs)

            # Create context with all relevant info
            verification_item = {
                "classification_key": classification_key,
                "prediction_id": prediction_id,
                "entry": entry,
                "prompt_metadata": prompt_metadata,
            }

            yield verification_item, prompt, pbar
            pbar.update(1)
            processed_count += 1

            if self.should_break_debug(processed_count):
                break

        pbar.close()
