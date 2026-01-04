#!/usr/bin/env python3
"""
Compute Mean Recall for Discrepancy Detection Models

This script loads evaluation results from the discrepancy dataset experiments
and computes mean recall metrics for each model. It provides a simplified
alternative to the full discrepancy_analysis.ipynb notebook.

Usage:
    # Display results to console
    python -m scicodeqa.evaluation.compute_recall
    # Save results to CSV
    python -m scicodeqa.evaluation.compute_recall -o results.csv
    # Use top-5 predictions
    python -m scicodeqa.evaluation.compute_recall --max-rank 5
    # Use different eval type
    python -m scicodeqa.evaluation.compute_recall --eval-type eval

Output:
    - Recall Overall: Mean recall across all discrepancies (real + synthetic)
    - Recall Real: Mean recall on real discrepancies only
    - Recall Synthetic: Mean recall on synthetic discrepancies only

Note: NaN values indicate that a model was not evaluated on that specific dataset.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_evaluations(base_dirs: list[Path]) -> pd.DataFrame:
    """
    Load all evaluations from the given base directories.
    """
    records = []
    eval_patterns = ["eval", "eval_v2", "eval_v2-gpt-oss-120b", "evaluation"]

    for base_dir in base_dirs:
        for eval_pattern in eval_patterns:
            # Use rglob for recursive search
            for eval_file in base_dir.rglob(f"{eval_pattern}/evaluations.jsonl"):
                run_dir = eval_file.parent.parent
                run_name = run_dir.name
                eval_type = eval_file.parent.name

                # Load generator model from parent args.json
                generator_args_file = run_dir / "args.json"
                if generator_args_file.exists():
                    with open(generator_args_file) as f:
                        generator_args = json.load(f)
                        model = generator_args.get("model", "unknown")
                else:
                    model = "unknown"

                # Check for GPT-OSS high/mid variant
                if (
                    "gpt-oss" in model.lower()
                    and not model.endswith("-high")
                    and not model.endswith("-mid")
                ):
                    llm_file = run_dir / "llm.json"
                    think_level = "mid"  # default
                    if llm_file.exists():
                        try:
                            with open(llm_file) as f:
                                llm_config = json.load(f)
                                think_param = (
                                    llm_config.get("model_config", {})
                                    .get("request_params", {})
                                    .get("extra_body", {})
                                    .get("think")
                                )
                                if think_param == "high":
                                    think_level = "high"
                        except (json.JSONDecodeError, KeyError):
                            pass
                    model = f"{model}-{think_level}"

                # Determine origin and code_only from path
                # New structure: out/inference/discrepancy_detection/
                # real/full or synthetic/code_only
                path_str = str(eval_file)
                origin = "synthetic" if "/synthetic/" in path_str else "real"
                code_only = "/code_only/" in path_str

                # Load evaluations
                with open(eval_file) as f:
                    for line in f:
                        try:
                            eval_data = json.loads(line)
                            records.append(
                                {
                                    "run_name": run_name,
                                    "model": model,
                                    "eval_type": eval_type,
                                    "origin": origin,
                                    "code_only": code_only,
                                    "discrepancy_id": eval_data.get("discrepancy_id"),
                                    "generation": eval_data.get("generation"),
                                    "answer": eval_data.get("answer"),
                                }
                            )
                        except json.JSONDecodeError as e:
                            print(f"Error parsing line in {eval_file}: {e}")

    return pd.DataFrame(records)


# Model name mappings for pretty display
MODEL_TO_PRETTY = {
    "gemini-2.5-flash-lite": "Gemini 2.5 Flash Lite",
    "gemini-2.5-flash": "Gemini 2.5 Flash",
    "gemini-2.5-pro": "Gemini 2.5 Pro",
    "gpt-5-2025-08-07": "GPT-5",
    "gpt-5": "GPT-5",
    "gpt-5-flex": "GPT-5",
    "gpt-5-codex": "GPT-5 Codex",
    "gpt-5-mini-2025-08-07": "GPT-5 Mini",
    "gpt-5-mini": "GPT-5 Mini",
    "gpt-5-mini-flex": "GPT-5 Mini",
    "gpt-5-nano-2025-08-07": "GPT-5 Nano",
    "gpt-5-nano": "GPT-5 Nano",
    "gpt-5-nano-flex": "GPT-5 Nano",
    "gpt-oss-120b-mid": "GPT-OSS 120B-Mid",
    "gpt-oss-20b-mid": "GPT-OSS 20B-Mid",
    "gpt-oss-120b-high": "GPT-OSS 120B",
    "gpt-oss-20b-high": "GPT-OSS 20B",
    "ollama_chat/gpt-oss:120b-mid": "GPT-OSS 120B-Mid",
    "ollama_chat/gpt-oss:20b-mid": "GPT-OSS 20B-Mid",
    "ollama_chat/gpt-oss:120b-high": "GPT-OSS 120B",
    "ollama_chat/gpt-oss:20b-high": "GPT-OSS 20B",
    "hosted_vllm/nvidia/Llama-3_3-Nemotron-Super-49B-v1_5": "Nemotron Super 49B v1.5",
    "vllm-llama-3-3-nemotron-super-49b-v1_5": "Nemotron Super 49B v1.5",
    "hosted_vllm/nvidia/NVIDIA-Nemotron-Nano-9B-v2": "Nemotron Nano 9B v2",
    "vllm-nemotron-nano-9b-v2": "Nemotron Nano 9B v2",
    "ollama_chat/deepseek-coder-v2:16b-lite-instruct-fp16": "DeepSeek Coder 16B V2",
    "deepseek-coder-v2-16b-lite": "DeepSeek Coder 16B V2",
    "DeepSeek-R1-Distill-Llama-70B": "DeepSeek R1 70B",
    "ollama_chat/deepseek-r1:32b-qwen-distill-fp16": "DeepSeek R1 32B",
    "deepseek-r1-32b": "DeepSeek R1 32B",
    "ollama_chat/deepseek-r1:8b-0528-qwen3-fp16": "DeepSeek R1 8B",
    "deepseek-r1-8b": "DeepSeek R1 8B",
    "ollama_chat/qwen3-coder:30b-a3b-fp16": "Qwen3 30B Coder",
    "qwen-3-coder-30b-a3b": "Qwen3 30B Coder",
    "ollama_chat/qwen3:30b-a3b-instruct-2507-fp16": "Qwen3 30B Inst.",
    "qwen-3-30b-a3b-instruct": "Qwen3 30B Inst.",
    "ollama_chat/qwen3:30b-a3b-thinking-2507-fp16": "Qwen3 30B Think.",
    "qwen-3-30b-a3b-thinking": "Qwen3 30B Think.",
    "ollama_chat/qwen3:4b-instruct-2507-fp16": "Qwen3 4B Inst.",
    "qwen-3-4b-instruct": "Qwen3 4B Inst.",
    "ollama_chat/qwen3:4b-thinking-2507-fp16": "Qwen3 4B Think.",
    "qwen-3-4b-thinking": "Qwen3 4B Think.",
    "Devstral-24B-Small-v0.1": "Devstral 24B Small",
    "devstral-24b-small-vllm": "Devstral 24B Small",
    "ollama_chat/devstral:24b-small": "Devstral 24B Small",
    "Magistral-24B-Small-v0.1": "Magistral 24B Small",
    "magistral-24b-small-vllm": "Magistral 24B Small",
    "ollama_chat/magistral:24b-small": "Magistral 24B Small",
}


def compute_recall(
    df: pd.DataFrame, eval_type: str = "eval_v2", max_rank: int = None
) -> pd.DataFrame:
    """
    Compute recall (hit rate) per model.
    
    Recall is computed as:
    (# discrepancies recalled) / (total # discrepancies in dataset)
    This matches the methodology in discrepancy_analysis.ipynb
    (the big LaTeX table).
    
    Note: By default (max_rank=None), considers ALL predictions for
    each discrepancy. This matches the notebook's main results table.
    Set max_rank=3 to only consider top-3 predictions.
    
    Returns a dataframe with columns: model, model_pretty,
    recall_overall, recall_real, recall_synthetic
    """
    # Dataset sizes (from the notebook)
    REAL_DATASET_SIZE = 81
    SYNTHETIC_DATASET_SIZE = 530
    TOTAL_DATASET_SIZE = REAL_DATASET_SIZE + SYNTHETIC_DATASET_SIZE
    
    # Filter to specific eval_type and non-code-only
    df = df[(df["eval_type"] == eval_type) & ~df["code_only"]].copy()
    
    # Optionally filter by rank
    if max_rank is not None:
        # Add rank column if not present
        # (rank predictions by generation number)
        if "rank" not in df.columns:
            df["rank"] = df.groupby(
                ["model", "origin", "discrepancy_id"]
            )["generation"].rank(method="first")
        
        # Filter to max_rank
        df = df[df["rank"] <= max_rank]
    
    # Mark matches
    df["is_match"] = df["answer"].str.lower() == "yes"
    
    # Add pretty names (this combines models like gpt-5 and gpt-5-flex into "GPT-5")
    df["model_pretty"] = df["model"].map(MODEL_TO_PRETTY).fillna(df["model"])
    
    # Compute recall: for each discrepancy, did any prediction match?
    # Group by model_pretty (not raw model) to combine variants
    # like gpt-5 and gpt-5-flex
    discrepancy_hits = (
        df.groupby(["model_pretty", "origin", "discrepancy_id"])
        .agg(has_match=("is_match", "any"))
        .reset_index()
    )
    
    # Count discrepancies recalled per model and origin
    recalls_by_origin = (
        discrepancy_hits
        .groupby(["model_pretty", "origin"])
        .agg(discrepancies_recalled=("has_match", "sum"))
        .reset_index()
    )
    
    # Compute recall by dividing by total dataset size (not by # predicted on)
    recalls_by_origin["recall"] = recalls_by_origin.apply(
        lambda row: row["discrepancies_recalled"] / (
            REAL_DATASET_SIZE if row["origin"] == "real" else SYNTHETIC_DATASET_SIZE
        ),
        axis=1
    )
    
    # Pivot to get recall_real and recall_synthetic columns
    recalls_pivot = recalls_by_origin.pivot_table(
        index="model_pretty",
        columns="origin",
        values="recall",
        aggfunc="first"
    ).reset_index()
    
    # Rename columns
    recalls_pivot.columns.name = None
    if "real" in recalls_pivot.columns:
        recalls_pivot.rename(columns={"real": "recall_real"}, inplace=True)
    else:
        recalls_pivot["recall_real"] = np.nan
        
    if "synthetic" in recalls_pivot.columns:
        recalls_pivot.rename(columns={"synthetic": "recall_synthetic"}, inplace=True)
    else:
        recalls_pivot["recall_synthetic"] = np.nan
    
    # Compute overall recall (across both real and synthetic)
    overall_recalled = (
        discrepancy_hits
        .groupby("model_pretty")
        .agg(discrepancies_recalled=("has_match", "sum"))
        .reset_index()
    )
    overall_recalled["recall_overall"] = (
        overall_recalled["discrepancies_recalled"] / TOTAL_DATASET_SIZE
    )
    
    # Merge all metrics
    result = overall_recalled[["model_pretty", "recall_overall"]].merge(
        recalls_pivot[["model_pretty", "recall_real", "recall_synthetic"]],
        on="model_pretty",
        how="left"
    )
    
    # Sort by overall recall descending
    result = result.sort_values("recall_overall", ascending=False)
    
    return result


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description=(
            "Compute mean recall for each model from "
            "discrepancy evaluation results."
        )
    )
    parser.add_argument(
        "--output", "-o", type=str, help="Output CSV file path (optional)"
    )
    parser.add_argument(
        "--eval-type",
        type=str,
        default="eval_v2",
        help="Evaluation type to use (default: eval_v2)",
    )
    parser.add_argument(
        "--max-rank",
        type=int,
        default=None,
        help="Maximum rank to consider (default: None = all predictions)",
    )
    args = parser.parse_args()
    
    print("=" * 80)
    print("Computing Mean Recall for Each Model")
    print("=" * 80)
    
    # Define base directories for evaluations (relative to project root)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    
    # New directory structure
    base_dirs = [
        project_root / "out/inference/discrepancy_detection/real/full",
        project_root / "out/inference/discrepancy_detection/real/code_only",
        project_root / "out/inference/discrepancy_detection/synthetic/full",
        project_root / "out/inference/discrepancy_detection/synthetic/code_only",
    ]
    
    # Also check old structure for backwards compatibility
    old_base_dirs = [
        project_root / "out/discrepancy_dataset",
        project_root / "out/discrepancy_dataset_synthetic",
        project_root / "out/discrepancy_dataset_code_only",
        project_root / "out/discrepancy_dataset_code_only_synthetic",
    ]
    
    # Filter to only existing directories
    base_dirs = [d for d in base_dirs + old_base_dirs if d.exists()]
    print(f"\nFound {len(base_dirs)} base directories:")
    for d in base_dirs:
        print(f"  - {d}")
    
    # Load all evaluations
    print("\nLoading evaluations...")
    df_evals = load_evaluations(base_dirs)
    print(f"Total evaluations loaded: {len(df_evals):,}")
    
    if len(df_evals) == 0:
        print("\n⚠️  No evaluation files found!")
        print("Make sure you have run discrepancy_eval.py on your inference results.")
        print("Evaluation files should be in: <run_dir>/evaluation/evaluations.jsonl")
        return
    
    # Exclude mid-tier models as per the notebook
    df_evals = df_evals[
        ~df_evals["model"].isin(["gpt-oss-20b-mid", "gpt-oss-120b-mid"])
    ]
    
    # Compute recall metrics
    rank_desc = (
        f"max_rank={args.max_rank}" if args.max_rank else "all predictions"
    )
    print(
        f"\nComputing recall metrics "
        f"(eval_type={args.eval_type}, {rank_desc})..."
    )
    df_recall = compute_recall(
        df_evals, eval_type=args.eval_type, max_rank=args.max_rank
    )
    
    # Format for display (convert to percentages)
    df_display = df_recall.copy()
    df_display["recall_overall"] = (df_display["recall_overall"] * 100).round(1)
    df_display["recall_real"] = (df_display["recall_real"] * 100).round(1)
    df_display["recall_synthetic"] = (df_display["recall_synthetic"] * 100).round(1)
    
    # Display results
    print("\n" + "=" * 80)
    rank_label = f"Top-{args.max_rank}" if args.max_rank else "All"
    print(
        f"RESULTS: Mean Recall by Model "
        f"({rank_label} predictions, {args.eval_type}, Paper+Code)"
    )
    print("=" * 80)
    print()
    
    # Only show relevant columns
    output_df = df_display[
        ["model_pretty", "recall_overall", "recall_real", "recall_synthetic"]
    ]
    output_df.columns = [
        "Model",
        "Recall Overall (%)",
        "Recall Real (%)",
        "Recall Synthetic (%)",
    ]
    
    print(output_df.to_string(index=False))
    print()
    
    # Summary statistics
    print("=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print(f"Number of models: {len(df_recall)}")
    print(f"Mean recall (overall): {df_recall['recall_overall'].mean() * 100:.1f}%")
    print(f"Mean recall (real): {df_recall['recall_real'].mean() * 100:.1f}%")
    print(f"Mean recall (synthetic): {df_recall['recall_synthetic'].mean() * 100:.1f}%")
    print()
    
    # Save to CSV if requested
    if args.output:
        output_df.to_csv(args.output, index=False)
        print(f"Results saved to: {args.output}")
        print()


if __name__ == "__main__":
    main()

