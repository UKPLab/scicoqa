#!/usr/bin/env python3
"""
Compute Mean Recall for Discrepancy Detection Models

This script loads evaluation results from the discrepancy dataset experiments
and computes mean recall metrics for each model. It provides a simplified
alternative to the full discrepancy_analysis.ipynb notebook.

Usage:
    # Display results to console (top-3 most similar, matching paper methodology)
    python -m scicoqa.evaluation.compute_recall --eval-type eval-gpt-oss-20b
    # Save results to CSV
    python -m scicoqa.evaluation.compute_recall --eval-type eval-gpt-oss-20b -o results.csv
    # Use all predictions instead of top-3
    python -m scicoqa.evaluation.compute_recall --eval-type eval-gpt-oss-20b --top-k 0

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
    Load all evaluations and their similarity scores from the given base
    directories.
    """
    records = []
    eval_patterns = ["eval", "eval-gpt-oss-20b", "eval-qwen-3-32b"]

    for base_dir in base_dirs:
        for eval_pattern in eval_patterns:
            # Use rglob for recursive search
            for eval_file in base_dir.rglob(f"{eval_pattern}/evaluations.jsonl"):
                run_dir = eval_file.parent.parent
                run_name = run_dir.name
                eval_type = eval_file.parent.name

                # Model name is the run directory name
                model = run_dir.name

                # Determine origin and code_only from path
                path_str = str(eval_file)
                origin = "synthetic" if "/synthetic/" in path_str else "real"
                code_only = "/code_only/" in path_str

                # Load similarities for this run (keyed by
                # (discrepancy_id, generation, gt_source))
                # Also keep gt_source-agnostic entries as fallback
                similarities = {}
                similarities_fallback = {}
                sim_file = run_dir / "similarities" / "similarities.jsonl"
                if sim_file.exists():
                    with open(sim_file) as f:
                        for line in f:
                            try:
                                sim_data = json.loads(line)
                                gt_source = sim_data.get("gt_source")
                                if gt_source:
                                    key = (
                                        sim_data["discrepancy_id"],
                                        sim_data["generation"],
                                        gt_source,
                                    )
                                    similarities[key] = sim_data["similarity"]
                                else:
                                    # Fallback: no gt_source
                                    fb_key = (
                                        sim_data["discrepancy_id"],
                                        sim_data["generation"],
                                    )
                                    similarities_fallback[fb_key] = (
                                        sim_data["similarity"]
                                    )
                            except json.JSONDecodeError:
                                pass

                # Load evaluations
                with open(eval_file) as f:
                    for line in f:
                        try:
                            eval_data = json.loads(line)
                            disc_id = eval_data.get("discrepancy_id")
                            generation = eval_data.get("generation")
                            gt_source = eval_data.get("gt_source")

                            # Look up similarity score (with fallback)
                            sim_key = (disc_id, generation, gt_source)
                            similarity = similarities.get(sim_key)
                            if similarity is None:
                                fb_key = (disc_id, generation)
                                similarity = similarities_fallback.get(fb_key)

                            records.append(
                                {
                                    "run_name": run_name,
                                    "model": model,
                                    "eval_type": eval_type,
                                    "origin": origin,
                                    "code_only": code_only,
                                    "discrepancy_id": disc_id,
                                    "gt_source": gt_source,
                                    "generation": generation,
                                    "answer": eval_data.get("answer"),
                                    "similarity": similarity,
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
    "gemini-3.1-pro": "Gemini 3.1 Pro",
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

# Dataset sizes per version
DATASET_SIZES = {
    "v1.0": {"real": 81, "synthetic": 530},
    "v1.1": {"real": 92, "synthetic": 543},
}


def compute_recall(
    df: pd.DataFrame,
    eval_type: str = "eval",
    top_k: int = 3,
    dataset_version: str = "v1.1",
) -> pd.DataFrame:
    """
    Compute recall (hit rate) per model.

    Recall is computed as:
    (# discrepancies recalled) / (total # discrepancies in dataset)

    By default, considers only the top-3 most similar predictions per
    (discrepancy, gt_source) pair, matching the paper's methodology.
    Set top_k=0 to consider all predictions.

    Returns a dataframe with columns: model, model_pretty,
    recall_overall, recall_real, recall_synthetic
    """
    sizes = DATASET_SIZES.get(dataset_version, DATASET_SIZES["v1.1"])
    REAL_DATASET_SIZE = sizes["real"]
    SYNTHETIC_DATASET_SIZE = sizes["synthetic"]
    TOTAL_DATASET_SIZE = REAL_DATASET_SIZE + SYNTHETIC_DATASET_SIZE

    # Filter to specific eval_type and non-code-only
    df = df[(df["eval_type"] == eval_type) & ~df["code_only"]].copy()

    # Filter to top-k most similar predictions per
    # (model, discrepancy_id, gt_source)
    if top_k > 0 and "similarity" in df.columns and df["similarity"].notna().any():
        df["sim_rank"] = df.groupby(
            ["model", "discrepancy_id", "gt_source"]
        )["similarity"].rank(method="first", ascending=False)
        before = len(df)
        df = df[df["sim_rank"] <= top_k]
        print(
            f"Filtered to top-{top_k} most similar: "
            f"{before:,} -> {len(df):,} evaluations"
        )
        df = df.drop(columns=["sim_rank"])
    elif top_k > 0:
        print(
            "WARNING: No similarity scores found. "
            "Using all predictions (top-k filtering disabled)."
        )

    # Mark matches
    df["is_match"] = df["answer"].str.lower() == "yes"

    # Add pretty names (combines model variants like gpt-5 and gpt-5-flex)
    df["model_pretty"] = df["model"].map(MODEL_TO_PRETTY).fillna(df["model"])

    # Compute recall: for each discrepancy, did any prediction match?
    # Group by model_pretty (not raw model) to combine variants
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

    # Compute recall by dividing by total dataset size
    recalls_by_origin["recall"] = recalls_by_origin.apply(
        lambda row: row["discrepancies_recalled"] / (
            REAL_DATASET_SIZE if row["origin"] == "real"
            else SYNTHETIC_DATASET_SIZE
        ),
        axis=1,
    )

    # Pivot to get recall_real and recall_synthetic columns
    recalls_pivot = recalls_by_origin.pivot_table(
        index="model_pretty",
        columns="origin",
        values="recall",
        aggfunc="first",
    ).reset_index()

    # Rename columns
    recalls_pivot.columns.name = None
    if "real" in recalls_pivot.columns:
        recalls_pivot.rename(columns={"real": "recall_real"}, inplace=True)
    else:
        recalls_pivot["recall_real"] = np.nan

    if "synthetic" in recalls_pivot.columns:
        recalls_pivot.rename(
            columns={"synthetic": "recall_synthetic"}, inplace=True
        )
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
        how="left",
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
        default="eval",
        help="Evaluation type to use (default: eval)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help=(
            "Keep only the top-k most similar predictions per discrepancy. "
            "Default: 0 (all predictions, matching the paper's main results). "
            "The paper uses top-3 only for the manual annotation validation."
        ),
    )
    parser.add_argument(
        "--dataset-version",
        type=str,
        default="v1.1",
        help="Dataset version for denominator sizes (default: v1.1)",
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
    print("\nLoading evaluations and similarities...")
    df_evals = load_evaluations(base_dirs)
    print(f"Total evaluations loaded: {len(df_evals):,}")

    if len(df_evals) == 0:
        print("\nNo evaluation files found!")
        print("Make sure you have run discrepancy_eval.py on your results.")
        return

    sim_count = df_evals["similarity"].notna().sum()
    print(f"Evaluations with similarity scores: {sim_count:,}")

    # Exclude mid-tier models as per the notebook
    df_evals = df_evals[
        ~df_evals["model"].isin(["gpt-oss-20b-mid", "gpt-oss-120b-mid"])
    ]

    # Compute recall metrics
    top_k_desc = f"top-{args.top_k} similar" if args.top_k > 0 else "all"
    sizes = DATASET_SIZES.get(args.dataset_version, DATASET_SIZES["v1.1"])
    print(
        f"\nComputing recall metrics "
        f"(eval_type={args.eval_type}, {top_k_desc}, "
        f"dataset={args.dataset_version}: "
        f"{sizes['real']} real + {sizes['synthetic']} synthetic)..."
    )
    df_recall = compute_recall(
        df_evals,
        eval_type=args.eval_type,
        top_k=args.top_k,
        dataset_version=args.dataset_version,
    )

    # Format for display (convert to percentages)
    df_display = df_recall.copy()
    df_display["recall_overall"] = (df_display["recall_overall"] * 100).round(1)
    df_display["recall_real"] = (df_display["recall_real"] * 100).round(1)
    df_display["recall_synthetic"] = (
        (df_display["recall_synthetic"] * 100).round(1)
    )

    # Display results
    print("\n" + "=" * 80)
    top_k_label = f"Top-{args.top_k} similar" if args.top_k > 0 else "All"
    print(
        f"RESULTS: Mean Recall by Model "
        f"({top_k_label} predictions, {args.eval_type}, Paper+Code)"
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
    print(
        f"Mean recall (overall): "
        f"{df_recall['recall_overall'].mean() * 100:.1f}%"
    )
    print(
        f"Mean recall (real): "
        f"{df_recall['recall_real'].mean() * 100:.1f}%"
    )
    print(
        f"Mean recall (synthetic): "
        f"{df_recall['recall_synthetic'].mean() * 100:.1f}%"
    )
    print()

    # Save to CSV if requested
    if args.output:
        output_df.to_csv(args.output, index=False)
        print(f"Results saved to: {args.output}")
        print()


if __name__ == "__main__":
    main()
