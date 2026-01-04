import json
import logging
from dataclasses import dataclass
from logging.config import fileConfig
from pathlib import Path

import simple_parsing
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from scicodeqa.core import InferenceArgs
from scicodeqa.core.code_loader import CodeLoader
from scicodeqa.core.experiment import BaseExperiment
from scicodeqa.core.mistral_ocr import MistralOCR
from scicodeqa.core.token_counter import TokenCounter

fileConfig("logging.ini")
logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class DiscrepancyDatasetInferenceArgs(InferenceArgs):
    prompt: str = "discrepancy_generation"
    discrepancies_file: Path = Path("data") / "scicoqa-real.jsonl"
    output_base_dir: Path = (
        Path("out") / "inference" / "discrepancy_detection" / "real" / "full"
    )
    dir_prefix: str = "discrepancy_gen"

    def __post_init__(self):
        super().__post_init__()

        # Determine dataset type (real vs synthetic)
        is_synthetic = "synthetic" in self.discrepancies_file.name
        
        # Determine if code-only mode from prompt
        is_code_only = "code_only" in self.prompt
        
        # Build the path based on dataset and mode
        dataset_dir = "synthetic" if is_synthetic else "real"
        mode_dir = "code_only" if is_code_only else "full"
        
        self.output_base_dir = (
            Path("out") / "inference" / "discrepancy_detection" / dataset_dir / mode_dir
        )
        
        # Adjust dir_prefix for synthetic
        if is_synthetic:
            self.dir_prefix = self.dir_prefix + "_synthetic"
        
        self.apply_code_changes = is_synthetic
        
        if is_synthetic:
            logger.info("Using synthetic discrepancies dataset")
        if is_code_only:
            logger.info("Using code-only mode (ablation)")

        exp_dir_name = self.get_exp_dir_name(self.output_base_dir)
        self.output_dir = self.output_base_dir / exp_dir_name

        # No structured output - model will generate YAML as instructed in prompt
        self.llm_kwargs = {}
        self.iterator_kwargs = {}


class DiscrepancyDatasetIterator:
    """Iterator that reads unique papers from discrepancies dataset."""

    def __init__(
        self,
        discrepancies_file: Path,
        output_dir: Path,
        prompt,
        model: str,
        tokenizer: str,
        model_max_tokens: int,
        debug: bool = False,
        debug_break_after: int = 1,
        apply_code_changes: bool = False,
    ):
        self.discrepancies_file = discrepancies_file
        self.output_dir = output_dir
        self.prompt = prompt
        self.model = model
        self.model_max_tokens = model_max_tokens
        self.debug = debug
        self.debug_break_after = debug_break_after
        self.apply_code_changes = apply_code_changes
        self.token_counter = TokenCounter(model=tokenizer)

        # Initialize OCR for papers
        self.mistral_ocr = MistralOCR()

        # Load unique papers
        self.unique_papers = self._load_unique_papers()
        logger.info(
            f"Found {len(self.unique_papers)} unique papers in discrepancies dataset"
        )

        # Track already processed papers
        self.already_processed = set()
        generations_file = self.output_dir / "generations.jsonl"
        if generations_file.exists():
            logger.info(f"Loading already processed papers from {generations_file}")
            with open(generations_file, "r") as f:
                for line in f:
                    generation = json.loads(line)
                    self.already_processed.add(generation["paper"]["id"])
            logger.info(f"Found {len(self.already_processed)} already processed papers")

    def _load_unique_papers(self) -> list[dict]:
        """Load unique papers from discrepancies JSONL file."""
        if not self.discrepancies_file.exists():
            raise FileNotFoundError(
                f"Discrepancies file not found: {self.discrepancies_file}"
            )

        papers_dict = {}
        with open(self.discrepancies_file, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                entry = json.loads(line)
                paper_url = entry.get("paper_url_versioned")
                code_url = entry.get("code_url")
                target_date = entry.get("discrepancy_date")

                # Create paper dict in format expected by iterator
                if paper_url not in papers_dict.keys():
                    papers_dict[paper_url] = {
                        "id": paper_url.split("/")[
                            -1
                        ],  # Use arxiv ID or last part of URL
                        "arxiv_url": paper_url,
                        "github_url": code_url,
                        "target_date": target_date,
                    }

                if self.apply_code_changes:
                    if "changed_code_files" not in papers_dict[paper_url].keys():
                        papers_dict[paper_url]["changed_code_files"] = []
                    papers_dict[paper_url]["changed_code_files"].extend(
                        entry["changed_code_files"]
                    )

        return list(papers_dict.values())

    def __call__(self, **kwargs):
        return self.__iter__(**kwargs)

    def __iter__(self, show_progress: bool = True, **kwargs):
        """Iterate over unique papers from discrepancies dataset."""
        MAX_TOKENS_BUFFER = 0.9  # Same as BaseIterator

        papers_to_process = [
            p for p in self.unique_papers if p["id"] not in self.already_processed
        ]

        if not papers_to_process:
            logger.info("All papers already processed")
            return

        logger.info(f"Processing {len(papers_to_process)} papers")
        pbar = tqdm(total=len(papers_to_process), ncols=100, disable=not show_progress)

        processed_count = 0
        for paper in papers_to_process:
            if self.debug and processed_count >= self.debug_break_after:
                logger.info(f"Debug mode: stopping after {processed_count} papers")
                break

            # Load paper using MistralOCR
            try:
                paper_text, _ = self.mistral_ocr(paper["arxiv_url"])
            except Exception as e:
                logger.error(f"Error loading paper for {paper['id']}: {e}")
                raise e

            # Calculate remaining tokens for code
            prompt_kwargs = {"paper": paper_text}
            intermediate_prompt = self.prompt(**prompt_kwargs)
            tokens_intermediate_prompt = self.token_counter(intermediate_prompt)
            remaining_code_tokens = int(
                self.model_max_tokens * MAX_TOKENS_BUFFER - tokens_intermediate_prompt
            )
            logger.info(f"Tokens intermediate prompt: {tokens_intermediate_prompt}")
            logger.info(f"Remaining code tokens: {remaining_code_tokens}")

            # Load code with target_date from discrepancy
            try:
                target_date = paper.get("target_date")
                if target_date:
                    logger.debug(
                        f"Loading code for {paper['id']} at target_date: {target_date}"
                    )
                code_loader = CodeLoader(paper["github_url"], target_date=target_date)
                code_text = code_loader.get_code_prompt(
                    token_counter=self.token_counter,
                    max_tokens=remaining_code_tokens,
                    code_changes=paper.get("changed_code_files"),
                )
                prompt_kwargs["code"] = code_text
            except Exception as e:
                logger.error(f"Error loading code for {paper['id']}: {e}")
                pbar.update(1)
                raise e

            # Format final prompt
            prompt = self.prompt(**prompt_kwargs)
            logger.info(f"Tokens in final prompt: {self.token_counter(prompt)}")

            yield paper, prompt, pbar
            pbar.update(1)
            processed_count += 1

        pbar.close()


class DiscrepancyDatasetExperiment(BaseExperiment):
    """Experiment that reads unique papers from discrepancies dataset."""

    def __init__(self, args: DiscrepancyDatasetInferenceArgs):
        super().__init__(args)
        self.iterator = DiscrepancyDatasetIterator(
            discrepancies_file=args.discrepancies_file,
            output_dir=args.output_dir,
            prompt=self.prompt,
            model=args.model,
            tokenizer=self.get_tokenizer(),
            model_max_tokens=self.llm.model_config["max_context_size"],
            debug=args.debug,
            debug_break_after=args.debug_break_after,
            apply_code_changes=args.apply_code_changes,
        )

    def _iter_items(self):
        """Iterate over unique papers from discrepancies dataset."""
        for paper, prompt, pbar in self.iterator(**self.args.iterator_kwargs):
            context = {"paper": paper}
            yield {
                "prompt": prompt,
                "pbar": pbar,
                "context": context,
                "log_label": paper["id"],
            }


if __name__ == "__main__":
    with logging_redirect_tqdm(loggers=[logging.root, logging.getLogger("scicodeqa")]):
        args, unknown_args = simple_parsing.parse_known_args(
            DiscrepancyDatasetInferenceArgs
        )
        logger.info(f"DiscrepancyDatasetInferenceArgs: {args}")

        if unknown_args:
            logger.warning(f"Unknown args: {unknown_args}")

        DiscrepancyDatasetExperiment(args)()
