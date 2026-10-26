"""
plot_experiment3_bars_and_curves.py
-----------------------------------
Processes the raw JSON outputs of Experiment 3 (Semantic Perturbation Scale)
to generate grouped bar charts and ROC/PR curves.

This script scans the results directory for completed `run.py` executions across
different T5 mask-filler capacities, extracts the detector and mask-filler configurations
alongside their evaluation metrics, and plots:
1. Grouped bar charts comparing AUROC and PR-AUC across mask-filler sizes for each detector.
2. ROC and Precision-Recall curves for each detector, colored by mask-filler capacity.

The script handles both the refactored argument names (`detector_name`, `mask_filler_name`)
and the legacy argument names (`scoring_model_name`, `mask_filling_model_name`) for backwards compatibility.
"""

import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Root directory containing the raw JSON outputs for Experiment 3
RESULTS_ROOT = "../../results/experiment3"

# Output directory for the generated figures, placed relative to this script
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures", "Experiment3")
os.makedirs(OUT_DIR, exist_ok=True)

# Maps HuggingFace model paths to clean display names
DET_MAP = {
    "EleutherAI/pythia-1.4b": "Pythia-1.4B",
    "EleutherAI/pythia-410m": "Pythia-410M",
    "EleutherAI/pythia-70m": "Pythia-70M",
}
MASK_MAP = {
    "t5-small": "T5-small",
    "t5-large": "T5-large",
    "t5-3b": "T5-3B",
}

# Exact ordering for the axes and legends
DETECTOR_ORDER = ["Pythia-1.4B", "Pythia-410M", "Pythia-70M"]
MASK_ORDER = ["T5-small", "T5-large", "T5-3B"]

# Global plot styling
plt.rcParams.update({"figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False})


def clean(obj):
    """
    Recursively strips stray whitespace from JSON keys and values.
    Handles edge cases where the JSON writer might inject spaces.
    """
    if isinstance(obj, dict):
        return {str(k).strip(): clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean(v) for v in obj]
    if isinstance(obj, str):
        return obj.strip()
    return obj


def load_results(root):
    """
    Reads the primary results JSON file. Falls back to merging chunked 
    JSON files if the primary file is missing.
    """
    path = os.path.join(root, "perturbation_50_d_results.json")
    if os.path.exists(path):
        with open(path) as f:
            return clean(json.load(f))
            
    merged = {}
    for c in sorted(glob.glob(os.path.join(root, "perturbation_50_d_results_chunk_*.json"))):
        with open(c) as f:
            data = clean(json.load(f))
        for key in ("metrics", "pr_metrics"):
            if key in data and key not in merged:
                merged[key] = data[key]
    return merged or None


def extract_runs():
    """
    Scans the results directory, extracts metrics from JSON files, and returns 
    a pandas DataFrame containing the detector, mask-filler, and evaluation metrics.
    
    Handles both the refactored argument names (`detector_name`, `mask_filler_name`) 
    and the legacy argument names (`scoring_model_name`, `mask_filling_model_name`).
    """
    runs = []
    for root, dirs, files in os.walk(RESULTS_ROOT):
        # Only descend into trees whose top-level folder starts with "perturbation_"
        if not any(p.startswith("perturbation_") for p in root.replace("\\", "/").split("/")):
            continue
            
        # Only the timestamped leaf folder contains args.json
        if "args.json" not in files:
            continue
            
        results = load_results(root)
        if results is None:
            continue
            
        with open(os.path.join(root, "args.json")) as f:
            args = clean(json.load(f))
            
        # Fallback to legacy argument names for older runs
        det_raw = args.get("detector_name") or args.get("scoring_model_name", "")
        mask_raw = args.get("mask_filler_name") or args.get("mask_filling_model_name", "")
        
        det = DET_MAP.get(det_raw)
        mask = MASK_MAP.get(mask_raw)
        
        if det is None or mask is None:
            continue
            
        runs.append({
            "Detector": det,
            "Mask model": mask,
            "roc_auc": results.get("metrics", {}).get("roc_auc"),
            "pr_auc": results.get("pr_metrics", {}).get("pr_auc"),
            "fpr": results.get("metrics", {}).get("fpr"),
            "tpr": results.get("metrics", {}).get("tpr"),
            "precision": results.get("pr_metrics", {}).get("precision"),
            "recall": results.get("pr_metrics", {}).get("recall"),
        })
        
    return pd.DataFrame(runs)


def plot_bars(df):
    """
    Generates and saves grouped bar charts comparing AUROC and PR-AUC 
    across mask-filler capacities for each detector scale.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    x = np.arange(len(MASK_ORDER))
    width = 0.25
    colors = {"Pythia-1.4B": "#4C72B0", "Pythia-410M": "#DD8452", "Pythia-70M": "#55A868"}
    
    for ax, metric, title in [(axes[0], "roc_auc", "AUROC"), (axes[1], "pr_auc", "PR-AUC")]:
        for i, det in enumerate(DETECTOR_ORDER):
            vals = [df.loc[(df.Detector == det) & (df["Mask model"] == m), metric].mean()
                    for m in MASK_ORDER]
            bars = ax.bar(x + (i - 1) * width, vals, width, label=det, color=colors[det])
            ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)
            
        ax.axhline(0.5, ls="--", lw=1, color="grey", label="Chance Level (0.5)")
        ax.set_ylim(0, 1.05)
        ax.set_xticks(x)
        ax.set_xticklabels(MASK_ORDER)
        ax.set_xlabel("Mask-filling model")
        ax.set_ylabel(title)
        ax.set_title(f"{title} by mask-filling model")
        ax.legend(fontsize=8)
        
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "perturbation_scale_bars.png")
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


def plot_curves(df):
    """
    Generates and saves ROC and Precision-Recall curves for each detector, 
    with lines colored by mask-filler capacity.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    colors = {"T5-small": "#4C72B0", "T5-large": "#DD8452", "T5-3B": "#55A868"}
    
    for col, det in enumerate(DETECTOR_ORDER):
        sub = df[df.Detector == det]
        
        # ROC Curves (Top Row)
        ax = axes[0, col]
        for _, r in sub.iterrows():
            if r.fpr and r.tpr:
                ax.plot(r.fpr, r.tpr, color=colors[r["Mask model"]],
                        label=f"{r['Mask model']} ({r.roc_auc:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Chance Level")
        ax.set_title(f"ROC: {det}")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.legend(fontsize=8)
        
        # PR Curves (Bottom Row)
        ax = axes[1, col]
        for _, r in sub.iterrows():
            if r.precision and r.recall:
                ax.plot(r.recall, r.precision, color=colors[r["Mask model"]],
                        label=f"{r['Mask model']} ({r.pr_auc:.3f})")
        ax.set_title(f"PR: {det}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.legend(fontsize=8)
        
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "perturbation_scale_curves.png")
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


def main():
    """Orchestrates the extraction of JSON results and the generation of plots."""
    df = extract_runs()
    if df.empty:
        print("No perturbation runs found under", RESULTS_ROOT)
        return
        
    print(df[["Detector", "Mask model", "roc_auc", "pr_auc"]].to_string(index=False))
    plot_bars(df)
    plot_curves(df)
    print("\nAll plotting tasks completed successfully!")


if __name__ == "__main__":
    main()