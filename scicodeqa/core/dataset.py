"""Dataset loading utilities for SciCoQA.

This module provides utilities for loading the SciCoQA dataset from HuggingFace
or local JSONL files.
"""

import json
import logging
from pathlib import Path

import pandas as pd
from datasets import load_dataset

logger = logging.getLogger(__name__)

# HuggingFace dataset identifier
HF_DATASET_ID = "UKPLab/scicoqa"


def load_scicoqa(
    split: str = "real",
    use_local: bool = False,
    local_path: Path | None = None,
) -> pd.DataFrame:
    """Load SciCoQA dataset from HuggingFace or local JSONL file.

    Args:
        split: Dataset split to load ('real' or 'synthetic').
        use_local: If True, load from local JSONL files instead of HuggingFace.
        local_path: Path to local JSONL file. If None and use_local=True,
            defaults to 'data/scicoqa-{split}.jsonl'.

    Returns:
        DataFrame containing the dataset.
    """
    if use_local:
        # Load from local JSONL file
        if local_path is None:
            local_path = Path("data") / f"scicoqa-{split}.jsonl"

        if not local_path.exists():
            raise FileNotFoundError(f"Local dataset file not found: {local_path}")

        logger.info(f"Loading SciCoQA {split} split from local file: {local_path}")
        df = pd.read_json(local_path, lines=True)
    else:
        # Load from HuggingFace Hub
        logger.info(f"Loading SciCoQA {split} split from HuggingFace: {HF_DATASET_ID}")
        dataset = load_dataset(HF_DATASET_ID, split=split)
        df = dataset.to_pandas()

        # Convert struct columns back to list-of-dicts format for compatibility
        # HuggingFace stores them as dict-of-lists (columnar format)
        if "changed_code_files" in df.columns:
            df["changed_code_files"] = df["changed_code_files"].apply(
                _columnar_to_row_format, keys=["file_name", "discrepancy_code"]
            )
        if "changed_code_snippets" in df.columns:
            df["changed_code_snippets"] = df["changed_code_snippets"].apply(
                _columnar_to_row_format,
                keys=["original_file", "original_code", "changed_code"],
            )

    logger.info(f"Loaded {len(df)} entries from SciCoQA {split} split")
    return df


def _columnar_to_row_format(col_dict: dict, keys: list[str]) -> list[dict]:
    """Convert columnar format (dict of lists) to row format (list of dicts).

    HuggingFace stores nested structures as dict-of-lists, but our code expects
    list-of-dicts format.

    Args:
        col_dict: Dictionary with lists as values.
        keys: Keys to extract from the dictionary.

    Returns:
        List of dictionaries.
    """
    if col_dict is None or not isinstance(col_dict, dict):
        return []

    # Get the length from the first key
    first_key = keys[0]
    if first_key not in col_dict:
        return []

    first_value = col_dict[first_key]
    # Handle numpy arrays, lists, or other sequences
    try:
        length = len(first_value)
    except TypeError:
        return []

    if length == 0:
        return []

    result = []
    for i in range(length):
        row = {}
        for k in keys:
            val = col_dict.get(k, [])
            try:
                row[k] = val[i] if i < len(val) else ""
            except (TypeError, IndexError):
                row[k] = ""
        result.append(row)
    return result


def load_scicoqa_as_records(
    split: str = "real",
    use_local: bool = False,
    local_path: Path | None = None,
) -> list[dict]:
    """Load SciCoQA dataset as a list of records.

    This is a convenience function that returns the dataset as a list of
    dictionaries, which is the format expected by some parts of the codebase.

    Args:
        split: Dataset split to load ('real' or 'synthetic').
        use_local: If True, load from local JSONL files instead of HuggingFace.
        local_path: Path to local JSONL file.

    Returns:
        List of dictionaries containing the dataset records.
    """
    df = load_scicoqa(split=split, use_local=use_local, local_path=local_path)
    return df.to_dict(orient="records")


def get_unique_papers(
    split: str = "real",
    use_local: bool = False,
    local_path: Path | None = None,
    apply_code_changes: bool = False,
) -> list[dict]:
    """Get unique papers from the dataset.

    This function extracts unique papers from the discrepancies dataset,
    deduplicating by paper URL and optionally aggregating code changes
    for synthetic data.

    Args:
        split: Dataset split to load ('real' or 'synthetic').
        use_local: If True, load from local JSONL files instead of HuggingFace.
        local_path: Path to local JSONL file.
        apply_code_changes: If True, aggregate code changes for each paper.

    Returns:
        List of unique paper dictionaries.
    """
    df = load_scicoqa(split=split, use_local=use_local, local_path=local_path)

    papers_dict = {}
    for _, entry in df.iterrows():
        paper_url = entry.get("paper_url_versioned")
        code_url = entry.get("code_url")
        target_date = entry.get("discrepancy_date")

        if paper_url not in papers_dict:
            papers_dict[paper_url] = {
                "id": paper_url.split("/")[-1] if paper_url else None,
                "arxiv_url": paper_url,
                "github_url": code_url,
                "target_date": target_date,
            }

        if apply_code_changes:
            if "changed_code_files" not in papers_dict[paper_url]:
                papers_dict[paper_url]["changed_code_files"] = []

            changed_files = entry.get("changed_code_files", [])
            if changed_files:
                papers_dict[paper_url]["changed_code_files"].extend(changed_files)

    return list(papers_dict.values())

