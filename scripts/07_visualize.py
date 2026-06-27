#!/usr/bin/env python3
"""Step 7: Generate all figures for the paper.

This script produces publication-quality figures from the experiment results:

  1. Concept distribution bar chart (Table 1 / Figure visualization)
  2. Ablation dose-response curve (Figure 2)
  3. Sensitivity analysis comparison table (Table 4 companion)
  4. Feature activation heatmap for example stocks
  5. Block-permutation inflation comparison

Paper reference: Figures 1-3, supporting visualizations.
Output: figures/*.pdf
"""

import json
import logging
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

# Style configuration
plt.rcParams.update({
    "font.size": 10,
    "font.family": "serif",
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def plot_concept_distribution(data: dict, fig_dir: Path) -> None:
    """Plot concept distribution bar chart (Table 1 visualization).

    Shows the percentage of feature assignments for each concept family
    at the Tier-1 threshold (|r| > 0.15).
    """
    tier1 = data["tier_results"].get("0.15", {})
    dist_pct = tier1.get("distribution_pct", {})

    if not dist_pct:
        logger.warning("No concept distribution data found for Tier 1")
        return

    # Sort by percentage
    sorted_items = sorted(dist_pct.items(), key=lambda x: -x[1])
    names = [item[0].replace("_", " ").title() for item in sorted_items]
    values = [item[1] for item in sorted_items]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.barh(range(len(names)), values, color="#4C72B0", alpha=0.85)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("Percentage of Feature Assignments (%)")
    ax.set_title(
        f"Concept Distribution (|r| > 0.15, {data['n_stocks']} stocks, "
        f"{tier1.get('total_assignments', '?')} assignments)"
    )
    ax.invert_yaxis()

    # Add percentage labels on bars
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%", va="center", fontsize=8,
        )

    # Add dashed line at uniform distribution
    n_fam = len(names)
    if n_fam > 0:
        uniform = 100.0 / n_fam
        ax.axvline(x=uniform, color="red", linestyle="--", alpha=0.5,
                    label=f"Uniform ({uniform:.1f}%)")
        ax.legend(loc="lower right")

    plt.tight_layout()
    fig.savefig(fig_dir / "concept_distribution.pdf")
    plt.close(fig)
    logger.info("Saved concept_distribution.pdf")


def plot_dose_response(data: dict, fig_dir: Path) -> None:
    """Plot ablation dose-response curve (Figure 2).

    Shows cosine similarity and top-1 agreement as functions of the number
    of ablated features.
    """
    dose_data = data.get("dose_response", {})
    if not dose_data:
        logger.warning("No dose-response data found")
        return

    # Filter to integer keys (dose levels)
    dose_levels = sorted([int(k) for k in dose_data.keys() if k.isdigit()])
    cosines = [dose_data[str(d)]["cosine_mean"] for d in dose_levels]
    top1s = [dose_data[str(d)]["top1_mean"] for d in dose_levels]

    fig, ax1 = plt.subplots(figsize=(6, 4))

    color1 = "#4C72B0"
    color2 = "#DD8452"

    ax1.plot(dose_levels, cosines, "o-", color=color1, linewidth=2, markersize=6,
             label="Cosine Similarity")
    ax1.set_xlabel("Number of Features Ablated")
    ax1.set_ylabel("Cosine Similarity", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_ylim(0.8, 1.01)

    ax2 = ax1.twinx()
    ax2.plot(dose_levels, top1s, "s--", color=color2, linewidth=2, markersize=6,
             label="Top-1 Agreement")
    ax2.set_ylabel("Top-1 Agreement", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.set_ylim(0, 1.0)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left")

    ax1.set_title("Ablation Dose-Response")
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(fig_dir / "ablation_dose_response.pdf")
    plt.close(fig)
    logger.info("Saved ablation_dose_response.pdf")


def plot_sensitivity_comparison(data: dict, fig_dir: Path) -> None:
    """Plot sensitivity analysis comparison.

    Creates two subplots: k sensitivity and expansion sensitivity.
    """
    k_sens = data.get("k_sensitivity", {})
    exp_sens = data.get("expansion_sensitivity", {})

    if not k_sens and not exp_sens:
        logger.warning("No sensitivity data found")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # k sensitivity
    if k_sens:
        configs = sorted(k_sens.keys())
        k_vals = [int(c.split("_")[0][1:]) for c in configs]
        ve = [k_sens[c]["var_explained_mean"] * 100 for c in configs]
        dead = [k_sens[c]["dead_rate_mean"] * 100 for c in configs]
        largest = [k_sens[c]["largest_pct_mean"] * 100 for c in configs]

        x = np.arange(len(k_vals))
        width = 0.25
        ax1.bar(x - width, ve, width, label="Var Explained (%)", color="#4C72B0", alpha=0.8)
        ax1.bar(x, dead, width, label="Dead Rate (%)", color="#DD8452", alpha=0.8)
        ax1.bar(x + width, largest, width, label="Largest Family (%)", color="#55A868", alpha=0.8)

        ax1.set_xticks(x)
        ax1.set_xticklabels([f"k={k}" for k in k_vals])
        ax1.set_title("k Sensitivity (Expansion 4x)")
        ax1.legend(fontsize=8)
        ax1.set_ylabel("Percentage (%)")

    # Expansion sensitivity
    if exp_sens:
        configs = sorted(exp_sens.keys())
        exp_vals = [int(c.split("_")[1][:-1].replace("exp", "")) for c in configs]
        ve = [exp_sens[c]["var_explained_mean"] * 100 for c in configs]
        dead = [exp_sens[c]["dead_rate_mean"] * 100 for c in configs]
        largest = [exp_sens[c]["largest_pct_mean"] * 100 for c in configs]

        x = np.arange(len(exp_vals))
        width = 0.25
        ax2.bar(x - width, ve, width, label="Var Explained (%)", color="#4C72B0", alpha=0.8)
        ax2.bar(x, dead, width, label="Dead Rate (%)", color="#DD8452", alpha=0.8)
        ax2.bar(x + width, largest, width, label="Largest Family (%)", color="#55A868", alpha=0.8)

        ax2.set_xticks(x)
        ax2.set_xticklabels([f"{e}x" for e in exp_vals])
        ax2.set_title("Expansion Sensitivity (k=64)")
        ax2.legend(fontsize=8)
        ax2.set_ylabel("Percentage (%)")

    plt.tight_layout()
    fig.savefig(fig_dir / "sensitivity_analysis.pdf")
    plt.close(fig)
    logger.info("Saved sensitivity_analysis.pdf")


def plot_financial_metrics(data: dict, fig_dir: Path) -> None:
    """Plot financial validation summary."""
    if "output_preservation" not in data:
        logger.warning("No financial validation data found")
        return

    op = data["output_preservation"]
    gt = data["ground_truth"]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Volatility ratio distribution
    per_stock = data.get("per_stock", [])
    if per_stock:
        vol_ratios = [r["vol_ratio"] for r in per_stock]
        axes[0].hist(vol_ratios, bins=15, color="#4C72B0", alpha=0.7, edgecolor="white")
        axes[0].axvline(x=1.0, color="red", linestyle="--", label="No change (1.0)")
        axes[0].axvline(
            x=op["volatility_ratio"]["mean"], color="green", linestyle="-",
            label=f"Mean ({op['volatility_ratio']['mean']:.2f}x)",
        )
        axes[0].set_xlabel("Volatility Ratio (Ablated / Baseline)")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Volatility Stability")
        axes[0].legend(fontsize=8)

    # Directional agreement
    if per_stock:
        dir_agrees = [r["dir_agreement"] for r in per_stock]
        axes[1].hist(dir_agrees, bins=15, color="#DD8452", alpha=0.7, edgecolor="white")
        axes[1].axvline(x=0.5, color="red", linestyle="--", label="Chance (50%)")
        axes[1].axvline(
            x=op["directional_agreement"]["mean"], color="green", linestyle="-",
            label=f"Mean ({op['directional_agreement']['mean']:.1%})",
        )
        axes[1].set_xlabel("Directional Agreement")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Direction Preservation")
        axes[1].legend(fontsize=8)

    # Baseline vs ablated directional accuracy
    if per_stock:
        base_dirs = [r["base_dir_acc"] for r in per_stock]
        ab_dirs = [r["ab_dir_acc"] for r in per_stock]
        axes[2].scatter(base_dirs, ab_dirs, alpha=0.6, s=30, c="#55A868", edgecolor="white")
        axes[2].plot([0, 1], [0, 1], "k--", alpha=0.3)
        axes[2].set_xlabel("Baseline Dir. Accuracy")
        axes[2].set_ylabel("Ablated Dir. Accuracy")
        axes[2].set_title("Ground-Truth Alignment")

    plt.tight_layout()
    fig.savefig(fig_dir / "financial_validation.pdf")
    plt.close(fig)
    logger.info("Saved financial_validation.pdf")


def main() -> None:
    cfg = load_config(PROJECT_ROOT / "configs" / "default.yaml")
    fig_dir = Path(cfg["paths"]["figures_dir"])
    results_dir = Path(cfg["paths"]["results_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. Concept distribution
    concept_path = results_dir / "concept_labeling.json"
    if concept_path.exists():
        with open(concept_path) as f:
            concept_data = json.load(f)
        plot_concept_distribution(concept_data, fig_dir)
    else:
        logger.warning("concept_labeling.json not found, skipping concept distribution plot")

    # 2. Dose-response ablation
    ablation_path = results_dir / "ablation_results.json"
    if ablation_path.exists():
        with open(ablation_path) as f:
            ablation_data = json.load(f)
        plot_dose_response(ablation_data, fig_dir)
    else:
        logger.warning("ablation_results.json not found, skipping dose-response plot")

    # 3. Sensitivity analysis
    sensitivity_path = results_dir / "sensitivity_results.json"
    if sensitivity_path.exists():
        with open(sensitivity_path) as f:
            sensitivity_data = json.load(f)
        plot_sensitivity_comparison(sensitivity_data, fig_dir)
    else:
        logger.warning("sensitivity_results.json not found, skipping sensitivity plot")

    # 4. Financial validation
    financial_path = results_dir / "financial_validation.json"
    if financial_path.exists():
        with open(financial_path) as f:
            financial_data = json.load(f)
        plot_financial_metrics(financial_data, fig_dir)
    else:
        logger.warning("financial_validation.json not found, skipping financial plot")

    logger.info("All available figures generated in %s", fig_dir)


if __name__ == "__main__":
    main()
