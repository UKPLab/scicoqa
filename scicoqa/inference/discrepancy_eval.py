import asyncio
import json
import logging
import re
from dataclasses import dataclass
from logging.config import fileConfig
from pathlib import Path

import pandas as pd
import simple_parsing
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from scicoqa.core import InferenceArgs
from scicoqa.core.dataset import load_scicoqa
from scicoqa.core.llm import AsyncVLLM
from scicoqa.core.prompt import Prompt

fileConfig("logging.ini")
logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class DiscrepancyEvalInferenceArgs(InferenceArgs):
    generations_dir: Path
    model: str | None = None  # Override to make optional
    similarity_model: str = "google/embeddinggemma-300m"
    vllm_server_url: str = "http://localhost:11435/v1"
    prompt: str = "discrepancy_evaluation_v2"
    # Dataset split: 'real' or 'synthetic'
    dataset_split: str = "real"
    # Use local JSONL files instead of HuggingFace Hub
    use_local: bool = False
    # Path to local JSONL file (only used if use_local=True)
    local_path: Path | None = None
    dir_prefix: str = "eval_v2"
    concurrency: int = 128
    seed: int = 42
    answer_field: str = "match"
    similarity_only: bool = False

    def __post_init__(self):
        # If similarity_only is True, model defaults to sentence transformer model
        if self.similarity_only and self.model is None:
            self.model = self.similarity_model

        # If similarity_only is False, model is required for LLM
        if not self.similarity_only and self.model is None:
            raise ValueError("--model is required when similarity_only is False")

        super().__post_init__()

        # Generations file is in the provided directory
        self.generations_file = self.generations_dir / "generations.jsonl"
        if not self.generations_file.exists():
            raise FileNotFoundError(
                f"Generations file not found: {self.generations_file}"
            )

        # Output directory is a subdirectory of the generations directory
        self.output_dir = self.generations_dir / self.dir_prefix

        if self.use_local:
            logger.info(f"Using local dataset files: {self.local_path or 'default'}")
        else:
            logger.info("Loading dataset from HuggingFace Hub: UKPLab/scicoqa")


class DiscrepancyEvalIterator:
    """Iterator that yields (reference_discrepancy, predicted_discrepancy) pairs."""

    def __init__(
        self,
        generations_file: Path,
        output_dir: Path,
        prompt: Prompt,
        dataset_split: str = "real",
        use_local: bool = False,
        local_path: Path | None = None,
        debug: bool = False,
        debug_break_after: int = 10,
    ):
        self.generations_file = generations_file
        self.output_dir = output_dir
        self.prompt = prompt
        self.dataset_split = dataset_split
        self.use_local = use_local
        self.local_path = local_path
        self.debug = debug
        self.debug_break_after = debug_break_after

        # Load and merge data
        discrepancy_df = self._load_discrepancies_df()
        generations_df = self._load_generations_df(self.generations_file)

        # Merge on normalized paper URLs
        self.df = pd.merge(
            discrepancy_df,
            generations_df[
                ["arxiv_url_normalized", "arxiv_url", "generation", "model"]
            ],
            left_on="paper_url_normalized",
            right_on="arxiv_url_normalized",
            how="inner",
        )

        logger.info(
            f"Merged {len(self.df)} pairs from {len(discrepancy_df)} discrepancies "
            f"and {len(generations_df)} generations"
        )

        # Track already processed
        self.already_processed = set()
        generations_file_out = self.output_dir / "generations.jsonl"
        if generations_file_out.exists():
            logger.info(f"Loading already processed from {generations_file_out}")
            with open(generations_file_out, "r") as f:
                for line in f:
                    gen = json.loads(line)
                    key = f"{gen['discrepancy_id']}_{gen['generation'][:50]}"
                    self.already_processed.add(key)
            logger.info(f"Found {len(self.already_processed)} already processed")

    def _load_discrepancies_df(self) -> pd.DataFrame:
        df = load_scicoqa(
            split=self.dataset_split,
            use_local=self.use_local,
            local_path=self.local_path,
        )
        # Normalize URLs for matching
        df["paper_url_normalized"] = (
            df["paper_url_versioned"]
            .str.replace(".pdf", "", regex=False)
            .str.replace("/abs/", "/pdf/", regex=False)
        )
        return df

    def _load_generations_df(self, generations_file: Path) -> pd.DataFrame:
        rows = []
        with open(generations_file) as f:
            for idx, line in enumerate(f):
                try:
                    generation = json.loads(line)
                    output = generation["response"]["choices"][0]["message"]["content"]

                    # Extract list items from output
                    discrepancies = self._extract_list_items(output)

                    arxiv_url = generation["paper"]["arxiv_url"]
                    # Normalize URL for matching
                    arxiv_url_normalized = arxiv_url.replace(".pdf", "").replace(
                        "/abs/", "/pdf/"
                    )

                    rows.append(
                        {
                            "arxiv_url": arxiv_url,
                            "arxiv_url_normalized": arxiv_url_normalized,
                            "generation": discrepancies,
                            "model": generation.get("model", "unknown"),
                        }
                    )
                except Exception as e:
                    logger.error(f"Error processing line {idx}: {e}")
                    raise

        df = pd.DataFrame(rows)
        df = df.explode("generation")
        df = df.dropna(subset=["generation"])
        return df

    def _extract_list_items(self, text: str) -> list[str] | None:
        """Extract list items from model output."""
        if not text:
            return None

        if "</think>" in text:
            text = text.split("</think>")[1]

        if "```yaml\ndiscrepancies:" in text:
            text = text.split("```yaml\ndiscrepancies:")[-1]
            yaml_or_dashed = True
        elif "```yaml" in text:
            text = text.split("```yaml")[-1]
            yaml_or_dashed = True
        elif "discrepancies:" in text:
            text = text.split("discrepancies:")[1]
            yaml_or_dashed = True
        elif re.search(r"# Discrepancies[\s\r\n]*-", text, re.IGNORECASE):
            text = re.split(
                r"# Discrepancies[\s\r\n]*-", text, maxsplit=1, flags=re.IGNORECASE
            )[1]
            text = "- " + text
            yaml_or_dashed = True
        else:
            yaml_or_dashed = False

        if yaml_or_dashed:
            text = text.strip("\n").strip().strip("```yaml").strip("```").strip("\n")
            text = (
                text.strip("discrepancies:").strip("discrepancies").strip("\n").strip()
            )

            pattern = r"\n\s{0,2}-\s+"
            parts = re.split(pattern, text)

            items = []
            for part in parts:
                cleaned = " ".join(part.split())
                if cleaned and not cleaned.startswith("discrepancies:"):
                    cleaned = cleaned.strip().strip("-").strip()
                    cleaned = cleaned.strip().strip("-").strip()
                    cleaned = cleaned.strip().strip("|").strip()
                    cleaned = cleaned.strip().strip(">-").strip()
                    cleaned = cleaned.strip().strip(">").strip()
                    cleaned = cleaned.strip().strip('"').strip()
                    cleaned = cleaned.strip().strip("'").strip()
                    cleaned = cleaned.strip("summary: |\n")
                    cleaned = cleaned.strip("summary: ")
                    cleaned = cleaned.strip("|")
                    cleaned = cleaned.strip("\n").strip()
                    cleaned = re.sub(r"^[0-9]+[\.\)]\s*", "", cleaned)
                    items.append(cleaned)
        else:
            items = None

        if items and len(items) == 1 and items[0].strip() == "[]":
            items = None

        return items if items else None

    def get_items(self) -> list[dict]:
        """Get all items to process."""
        items = []
        for row in self.df.itertuples():
            key = f"{row.discrepancy_id}_{row.generation[:50]}"
            if key in self.already_processed:
                continue

            items.append(
                {
                    "discrepancy_id": row.discrepancy_id,
                    "discrepancy_description": row.discrepancy_description,
                    "generation": row.generation,
                    "paper_url": getattr(row, "paper_url_versioned", None),
                    "origin": getattr(row, "origin", None),
                    "model": getattr(row, "model", None),
                }
            )

            if self.debug and len(items) >= self.debug_break_after:
                break

        return items

    def get_all_items(self) -> list[dict]:
        """Get all items including already processed ones (for sim. computation)."""
        items = []
        for row in self.df.itertuples():
            items.append(
                {
                    "discrepancy_id": row.discrepancy_id,
                    "discrepancy_description": row.discrepancy_description,
                    "generation": row.generation,
                    "paper_url": getattr(row, "paper_url_versioned", None),
                    "origin": getattr(row, "origin", None),
                    "model": getattr(row, "model", None),
                }
            )

            if self.debug and len(items) >= self.debug_break_after:
                break

        return items

    def format_prompts(self, items: list[dict]) -> list[str]:
        """Format prompts for all items."""
        return [
            self.prompt(
                reference_discrepancy=item["discrepancy_description"],
                predicted_discrepancy=item["generation"],
            )
            for item in items
        ]


def extract_answer(content: str, answer_field: str = "answer") -> str:
    """Extract yes/no answer from model output."""
    if not content:
        return "no"

    try:
        # Try for explicit '{answer_field}:' at end
        field_regex = rf'{answer_field}:\s*(["\']?\s*(yes|no)\s*["\']?)\s*$'
        match = re.search(field_regex, content, re.IGNORECASE)
        if match:
            return match.group(2).lower()
    except Exception as e:
        logger.error(f"Error extracting answer: {e}")

    # Look for any '{answer_field}:' line, grab last if multiple
    matches = list(
        re.finditer(rf"{answer_field}:\s*([\s\S]+?)(?:$|\n)", content, re.IGNORECASE)
    )
    if matches:
        last = matches[-1].group(1).strip()
        val_match = re.match(r'["\']?\s*(yes|no)\s*["\']?', last, re.IGNORECASE)
        if val_match:
            return val_match.group(1).lower()

    # Try to find 'yes' or 'no' at end
    val_match = re.search(r'(yes|no)\s*["\']*\s*$', content, re.IGNORECASE)
    if val_match:
        return val_match.group(1).lower()

    return "no"


class DiscrepancyEvalExperiment:
    """Experiment for evaluating discrepancy predictions using async vLLM."""

    def __init__(self, args: DiscrepancyEvalInferenceArgs):
        self.args = args
        self.output_dir = args.output_dir
        self.generations_file = args.output_dir / "generations.jsonl"
        self.evaluations_file = args.output_dir / "evaluations.jsonl"

        # Similarities directory and file
        self.similarities_dir = args.generations_dir / "similarities"
        self.similarities_file = self.similarities_dir / "similarities.jsonl"

        # Track already computed similarities
        self.already_computed_similarities = set()
        if self.similarities_file.exists():
            logger.info(
                f"Loading already computed similarities from {self.similarities_file}"
            )
            with open(self.similarities_file, "r") as f:
                for line in f:
                    similarity_record = json.loads(line)
                    key = (
                        f"{similarity_record['discrepancy_id']}_"
                        f"{similarity_record['generation'][:50]}"
                    )
                    self.already_computed_similarities.add(key)
            logger.info(
                f"Found {len(self.already_computed_similarities)} "
                "already computed similarities"
            )

        # Initialize prompt
        self.prompt = Prompt(args.prompt)

        # Initialize async vLLM client (only if not similarity_only)
        if not args.similarity_only:
            self.llm = AsyncVLLM(
                model=args.model,
                decoding_settings=args.decoding_config,
                output_file=args.output_dir / "llm.json",
                vllm_server_url=args.vllm_server_url,
                concurrency=args.concurrency,
                seed=args.seed,
            )
        else:
            self.llm = None

        # Initialize sentence transformer for similarity computation
        # Always use the similarity_model parameter
        self.similarity_model = SentenceTransformer(
            args.similarity_model, trust_remote_code=True
        )

        # Initialize iterator
        self.iterator = DiscrepancyEvalIterator(
            generations_file=args.generations_file,
            output_dir=args.output_dir,
            prompt=self.prompt,
            dataset_split=args.dataset_split,
            use_local=args.use_local,
            local_path=args.local_path,
            debug=args.debug,
            debug_break_after=args.debug_break_after,
        )

        self.answer_field = args.answer_field

    def save_generation(self, item: dict, response: dict):
        """Save a single generation to generations.jsonl."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        generation = {
            **item,
            "response": response,
        }
        with open(self.generations_file, "a") as f:
            f.write(json.dumps(generation) + "\n")

    def save_evaluation(self, item: dict, response: dict, answer: str):
        """Save a single evaluation to evaluations.jsonl."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        evaluation = {
            "discrepancy_id": item["discrepancy_id"],
            "generation": item["generation"],
            "answer": answer,
            "reasoning": response.get("choices", [{}])[0]
            .get("message", {})
            .get("content", ""),
        }
        with open(self.evaluations_file, "a") as f:
            f.write(json.dumps(evaluation) + "\n")

    def compute_similarities_batch(self, items: list[dict]) -> list[float]:
        """Compute similarities for a batch of items."""
        if not items:
            return []

        # Extract all texts
        discrepancy_descriptions = [item["discrepancy_description"] for item in items]
        generations = [item["generation"] for item in items]

        # Batch encode all texts
        discrepancy_embeddings = self.similarity_model.encode(
            discrepancy_descriptions,
            batch_size=32,
            show_progress_bar=False,
            convert_to_tensor=True,
        )
        generation_embeddings = self.similarity_model.encode(
            generations,
            batch_size=32,
            show_progress_bar=False,
            convert_to_tensor=True,
        )

        # Compute pairwise similarities
        similarities = self.similarity_model.similarity(
            discrepancy_embeddings, generation_embeddings
        )

        # Extract diagonal (pairwise similarities)
        # similarities is a matrix, we want the diagonal for pairwise comparisons
        # Convert to numpy if tensor, moving to CPU first if needed
        if hasattr(similarities, "cpu"):
            # Move to CPU first (handles MPS, CUDA, etc.), then convert to numpy
            similarities = similarities.cpu().numpy()
        elif hasattr(similarities, "numpy"):
            # Already on CPU, just convert to numpy
            similarities = similarities.numpy()

        # Extract diagonal elements (pairwise similarities)
        similarity_scores = [float(similarities[i, i]) for i in range(len(items))]

        return similarity_scores

    def save_similarity(self, item: dict, similarity: float):
        """Save a single similarity to similarities.jsonl."""
        self.similarities_dir.mkdir(parents=True, exist_ok=True)
        similarity_record = {
            "discrepancy_id": item["discrepancy_id"],
            "generation": item["generation"],
            "similarity": similarity,
        }
        with open(self.similarities_file, "a") as f:
            f.write(json.dumps(similarity_record) + "\n")

    def save_config(self):
        """Save experiment configuration."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.args.save(self.output_dir / "args.json")
        if self.llm is not None:
            self.llm.save()
        self.prompt.save(self.output_dir / "prompt.json")

    async def run_async(self):
        """Run the evaluation experiment asynchronously."""
        if self.args.similarity_only:
            # Get all items for similarity computation (including already processed)
            all_items = self.iterator.get_all_items()
            if not all_items:
                logger.info("No items to process")
                return

            logger.info(f"Computing similarities for {len(all_items)} items")

            # Compute and save similarities for all items
            items_to_compute_similarity = []
            for item in all_items:
                key = f"{item['discrepancy_id']}_{item['generation'][:50]}"
                if key not in self.already_computed_similarities:
                    items_to_compute_similarity.append(item)

            if items_to_compute_similarity:
                logger.info(
                    "Computing similarities for "
                    f"{len(items_to_compute_similarity)} items"
                )
                # Compute similarities in batch
                similarities = self.compute_similarities_batch(
                    items_to_compute_similarity
                )

                # Save all similarities
                for item, similarity in tqdm(
                    zip(items_to_compute_similarity, similarities),
                    total=len(items_to_compute_similarity),
                    desc="Saving similarities",
                ):
                    self.save_similarity(item, similarity)
            else:
                logger.info("All similarities already computed")

            logger.info(f"Completed similarity computation for {len(all_items)} items")
            return

        # Get all items to process
        items = self.iterator.get_items()
        if not items:
            logger.info("No items to process")
            return

        logger.info(f"Processing {len(items)} items")

        # Format all prompts
        prompts = self.iterator.format_prompts(items)

        # Run batch completion
        responses = await self.llm.batch_completion(prompts, show_progress=True)

        # Compute and save similarities for all items
        logger.info("Computing similarities...")
        items_to_compute_similarity = []
        for item in items:
            key = f"{item['discrepancy_id']}_{item['generation'][:50]}"
            if key not in self.already_computed_similarities:
                items_to_compute_similarity.append(item)

        if items_to_compute_similarity:
            logger.info(
                f"Computing similarities for {len(items_to_compute_similarity)} items"
            )
            # Compute similarities in batch
            similarities = self.compute_similarities_batch(items_to_compute_similarity)

            # Save all similarities
            for item, similarity in tqdm(
                zip(items_to_compute_similarity, similarities),
                total=len(items_to_compute_similarity),
                desc="Saving similarities",
            ):
                self.save_similarity(item, similarity)
        else:
            logger.info("All similarities already computed")

        # Process and save results
        for item, response in tqdm(
            zip(items, responses), total=len(items), desc="Saving results"
        ):
            # Extract answer
            content = (
                response.get("choices", [{}])[0].get("message", {}).get("content", "")
            )
            answer = extract_answer(content, self.answer_field)

            # Save both generation and evaluation
            self.save_generation(item, response)
            self.save_evaluation(item, response, answer)

        # Save config
        self.save_config()

        logger.info(f"Completed evaluation of {len(items)} items")
        logger.info(f"Results saved to {self.output_dir}")

    def __call__(self):
        """Run the experiment."""
        asyncio.run(self.run_async())


if __name__ == "__main__":
    with logging_redirect_tqdm(loggers=[logging.root, logging.getLogger("scicoqa")]):
        args, unknown_args = simple_parsing.parse_known_args(
            DiscrepancyEvalInferenceArgs
        )
        logger.info(f"DiscrepancyEvalInferenceArgs: {args}")

        if unknown_args:
            logger.warning(f"Unknown args: {unknown_args}")

        DiscrepancyEvalExperiment(args)()
