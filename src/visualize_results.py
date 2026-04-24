"""
Visualization script: generates comparison charts from evaluation results.

Produces:
1. Quality metrics bar chart (BLEU, ROUGE, BERTScore)
2. Diversity metrics bar chart (Self-BLEU, Jaccard, Lexical Diversity)
3. Latency comparison bar chart
4. Length ratio check

Usage: python -m src.visualize_results
"""

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np


RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
RESULTS_FILE = os.path.join(RESULTS_DIR, "evaluation_results.json")


def load_results() -> dict:
    """Load evaluation results from JSON."""
    if not os.path.exists(RESULTS_FILE):
        print(f"Results file not found: {RESULTS_FILE}")
        print("Run `python -m src.run_comparison` first.")
        sys.exit(1)

    with open(RESULTS_FILE, "r") as f:
        return json.load(f)


def plot_quality_metrics(cpg_metrics: dict, llm_metrics: dict = None):
    """Bar chart comparing quality metrics (BLEU, ROUGE, BERTScore)."""
    metrics = ["bleu", "rouge1_f1", "rouge2_f1", "rougeL_f1", "bert_score_f1"]
    labels = ["BLEU", "ROUGE-1", "ROUGE-2", "ROUGE-L", "BERTScore"]

    cpg_vals = [cpg_metrics.get(m, 0) for m in metrics]
    # Normalize BLEU to 0-1 scale for chart consistency
    cpg_vals[0] = cpg_vals[0] / 100.0

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, cpg_vals, width, label="CPG (T5-base)", color="#2196F3")

    if llm_metrics:
        llm_vals = [llm_metrics.get(m, 0) for m in metrics]
        llm_vals[0] = llm_vals[0] / 100.0
        bars2 = ax.bar(x + width / 2, llm_vals, width, label="LLM Baseline", color="#FF9800")

    ax.set_ylabel("Score")
    ax.set_title("Quality Metrics: CPG vs LLM")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bar in bars1:
        h = bar.get_height()
        ax.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "quality_metrics.png")
    plt.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close()


def plot_diversity_metrics(cpg_metrics: dict, llm_metrics: dict = None):
    """Bar chart comparing diversity metrics."""
    metrics = ["self_bleu", "jaccard_similarity", "lexical_diversity"]
    labels = ["Self-BLEU\n(lower=better)", "Jaccard\n(0.3-0.6 ideal)", "Lexical Diversity\n(higher=better)"]

    cpg_vals = [cpg_metrics.get(m, 0) for m in metrics]
    cpg_vals[0] = cpg_vals[0] / 100.0  # Normalize Self-BLEU

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 6))
    bars1 = ax.bar(x - width / 2, cpg_vals, width, label="CPG (T5-base)", color="#2196F3")

    if llm_metrics:
        llm_vals = [llm_metrics.get(m, 0) for m in metrics]
        llm_vals[0] = llm_vals[0] / 100.0
        bars2 = ax.bar(x + width / 2, llm_vals, width, label="LLM Baseline", color="#FF9800")

    ax.set_ylabel("Score")
    ax.set_title("Diversity Metrics: CPG vs LLM")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)

    for bar in bars1:
        h = bar.get_height()
        ax.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "diversity_metrics.png")
    plt.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close()


def plot_latency(cpg_metrics: dict, llm_metrics: dict = None):
    """Bar chart comparing inference latency."""
    systems = ["CPG (T5-base)"]
    latencies = [cpg_metrics.get("latency_ms", 0)]
    colors = ["#2196F3"]

    if llm_metrics:
        systems.append("LLM Baseline")
        latencies.append(llm_metrics.get("latency_ms", 0))
        colors.append("#FF9800")

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(systems, latencies, color=colors, width=0.5)

    ax.set_ylabel("Latency (ms)")
    ax.set_title("Inference Latency Comparison")
    ax.grid(axis="y", alpha=0.3)

    for bar, lat in zip(bars, latencies):
        ax.annotate(f"{lat:.0f}ms", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 5), textcoords="offset points", ha="center", fontweight="bold")

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "latency_comparison.png")
    plt.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close()


def plot_length_ratio(results: dict):
    """Bar chart showing length ratios vs 80% threshold."""
    systems = ["CPG"]
    ratios = [results["cpg"]["length_ratio"]]
    colors = ["#2196F3"]

    if "llm" in results:
        systems.append("LLM")
        ratios.append(results["llm"]["length_ratio"])
        colors.append("#FF9800")

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(systems, ratios, color=colors, width=0.5)

    # Draw the 80% threshold line
    ax.axhline(y=0.80, color="red", linestyle="--", label="80% threshold")

    ax.set_ylabel("Length Ratio (output/input)")
    ax.set_title("Length Ratio Check (≥ 0.80 required)")
    ax.legend()
    ax.set_ylim(0, max(max(ratios) * 1.2, 1.1))
    ax.grid(axis="y", alpha=0.3)

    for bar, ratio in zip(bars, ratios):
        status = "✓" if ratio >= 0.80 else "✗"
        ax.annotate(f"{ratio:.2f} {status}",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 5), textcoords="offset points", ha="center", fontweight="bold")

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "length_ratio.png")
    plt.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close()


def main():
    results = load_results()

    cpg_metrics = results["cpg"]["metrics"]
    llm_metrics = results.get("llm", {}).get("metrics")

    print("Generating comparison charts...")
    plot_quality_metrics(cpg_metrics, llm_metrics)
    plot_diversity_metrics(cpg_metrics, llm_metrics)
    plot_latency(cpg_metrics, llm_metrics)
    plot_length_ratio(results)

    # Also generate a combined summary image
    print("\nAll charts saved to results/ directory.")


if __name__ == "__main__":
    main()
