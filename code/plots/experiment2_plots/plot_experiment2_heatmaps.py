"""
plot_experiment2_heatmaps.py
----------------------------
Processes the raw JSON outputs of Experiment 2 (Training Checkpoint Dynamics)
and generates sorted heatmaps for AUROC and PR-AUC.

This script scans the results directory for completed `run.py` executions across
different Pythia training checkpoints, extracts the generator and detector configurations
alongside their evaluation metrics, and saves the aggregated matrices as CSV files.
It then uses these matrices to plot and save high-resolution heatmaps, sorting the
generators from largest to smallest parameter count and the detectors chronologically
by training step to visually highlight the inverse relationship between training
duration and detection efficacy.
"""

import os
import json
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Root directory containing the raw JSON outputs for Experiment 2
RESULTS_ROOT = "../../results/experiment2"

# Maps HuggingFace detector paths to clean display names
DET_MAP = {
    'EleutherAI/pythia-1.4b': 'Pythia-1.4B',
    'EleutherAI/pythia-410m': 'Pythia-410M',
    'EleutherAI/pythia-70m': 'Pythia-70M'
}

# Exact Y-axis order (Generators), sorted from largest to smallest parameter count
GENERATOR_ORDER = [
    'HuggingFaceTB/SmolLM2-1.7B',  # 1.7B
    'gpt2-xl',                     # 1.5B
    'EleutherAI/gpt-neo-1.3B',     # 1.3B
    'facebook/opt-1.3b',           # 1.3B
    'gpt2-large',                  # 774M     
    'HuggingFaceTB/SmolLM2-360M',  # 360M
    'gpt2-medium',                 # 355M
    'facebook/opt-350m',           # 350M
    'HuggingFaceTB/SmolLM2-135M',  # 135M
    'EleutherAI/gpt-neo-125M',     # 125M
    'facebook/opt-125m',           # 125M
    'gpt2',                        # 124M
    'distilgpt2'                   # 82M
]

# Exact X-axis order (Detector and Step combinations), sorted chronologically
DETECTORS = ['Pythia-1.4B', 'Pythia-410M', 'Pythia-70M']
STEPS = ['1k', '5k', '10k', '50k', '100k', '143k']
X_AXIS_ORDER = [f"{det}_{step}" for det in DETECTORS for step in STEPS]


def extract_and_save_csvs():
    """
    Scans experiment folders, parses metrics from JSON files, and exports 
    ROC-AUC & PR-AUC pivot matrices to CSV.
    
    Handles both the refactored argument names (`generator_name`, `detector_name`, 
    `detector_revision`) and the legacy argument names (`base_model_name`, 
    `scoring_model_name`, `scoring_model_revision`) for backwards compatibility.
    
    Returns:
        bool: True if data was successfully extracted and saved, False otherwise.
    """
    aggregated_data = []
    print("Scanning directories for 'stepsizes_' experiment results...")

    for root, dirs, files in os.walk(RESULTS_ROOT):
        normalized_root = root.replace('\\', '/')
        if "stepsizes_" not in normalized_root:
            continue
        
        if "args.json" in files and "perturbation_50_d_results.json" in files:
            args_path = os.path.join(root, "args.json")
            results_path = os.path.join(root, "perturbation_50_d_results.json")
            
            try:
                with open(args_path, "r") as f:
                    args_data = json.load(f)
                    
                    # Fallback to legacy argument names for older runs
                    gen_model = args_data.get("generator_name") or args_data.get("base_model_name")
                    det_model = args_data.get("detector_name") or args_data.get("scoring_model_name")
                    
                    # Extract training step from revision argument or fallback to folder name parsing
                    step_str = args_data.get("detector_revision") or args_data.get("scoring_model_revision")
                    if step_str and step_str.startswith("step"):
                        step_size_num = int(step_str.replace("step", ""))
                        step_str = f"{step_size_num // 1000}k" if step_size_num >= 1000 else str(step_size_num)
                    else:
                        # Fallback: parse step from the parent folder name (e.g., ...-step10000-...)
                        parent_folder = os.path.basename(os.path.dirname(root))
                        step_match = re.search(r'step(\d+)', parent_folder)
                        if step_match:
                            step_size_num = int(step_match.group(1))
                            step_str = f"{step_size_num // 1000}k" if step_size_num >= 1000 else str(step_size_num)
                        else:
                            continue  # Skip if step cannot be determined
                    
                with open(results_path, "r") as f:
                    results_data = json.load(f)
                    roc_auc = results_data.get("metrics", {}).get("roc_auc")
                    pr_auc = results_data.get("pr_metrics", {}).get("pr_auc")
                    
                if (roc_auc is not None or pr_auc is not None) and gen_model and det_model:
                    clean_det = DET_MAP.get(det_model, det_model.split('/')[-1])
                    detector_step = f"{clean_det}_{step_str}"
                    
                    aggregated_data.append({
                        "Generator Model": gen_model,
                        "Detector_Step": detector_step,
                        "ROC-AUC": float(roc_auc) if roc_auc is not None else None,
                        "PR-AUC": float(pr_auc) if pr_auc is not None else None
                    })
            except Exception as e:
                print(f"Error parsing folder {root}: {e}")

    df = pd.DataFrame(aggregated_data)

    if df.empty:
        print("No valid 'stepsizes_' results found. Check your RESULTS_ROOT path.")
        return False

    print(f"Successfully collected {len(df)} 'stepsizes_' experiment runs!")

    # Save ROC-AUC CSV matrix
    if "ROC-AUC" in df.columns and not df["ROC-AUC"].isna().all():
        roc_matrix = df.pivot_table(
            index="Generator Model",
            columns="Detector_Step",
            values="ROC-AUC",
            aggfunc="mean"
        )
        roc_matrix.to_csv("stepsizes_heatmap_roc_auc_matrix.csv")
        print("Saved matrix layout to 'stepsizes_heatmap_roc_auc_matrix.csv'")

    # Save PR-AUC CSV matrix
    if "PR-AUC" in df.columns and not df["PR-AUC"].isna().all():
        pr_matrix = df.pivot_table(
            index="Generator Model",
            columns="Detector_Step",
            values="PR-AUC",
            aggfunc="mean"
        )
        pr_matrix.to_csv("stepsizes_heatmap_pr_auc_matrix.csv")
        print("Saved matrix layout to 'stepsizes_heatmap_pr_auc_matrix.csv'")

    return True


def generate_heatmap(metric):
    """
    Generates and saves the formatted heatmap image for a given metric.
    
    Args:
        metric (str): The metric to plot, either 'roc_auc' or 'pr_auc'.
    """
    csv_filename = f'stepsizes_heatmap_{metric}_matrix.csv'
    png_filename = f'stepsizes_combined_{metric}_matrix.png'
    
    display_metric = "AUROC" if metric == 'roc_auc' else "PR-AUC"

    if not os.path.exists(csv_filename):
        print(f"\nFile {csv_filename} not found. Skipping {display_metric} heatmap...")
        return

    print(f"\nGenerating heatmap for {display_metric}...")
    
    df = pd.read_csv(csv_filename, index_col=0)

    # Reindex to enforce the specified Y-axis and X-axis order
    df_sorted = df.reindex(index=GENERATOR_ORDER, columns=X_AXIS_ORDER)

    plt.figure(figsize=(20, 6))
    sns.heatmap(df_sorted, annot=True, cmap="mako", vmin=0.0, vmax=1.0, fmt=".3f", linewidths=0.5)

    plt.title(f"{display_metric} Across Detector Training Steps for Various Generator Models", pad=20, fontsize=16)
    plt.ylabel("Generator Model", labelpad=10, fontsize=12)
    plt.xlabel("Detector Model (Training Steps)", labelpad=10, fontsize=12)

    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=11)

    plt.tight_layout()
    plt.savefig(png_filename, dpi=300, bbox_inches='tight')
    print(f"Saved combined heatmap image to '{png_filename}'")
    plt.close()


def main():
    """Orchestrates the extraction of JSON results and the generation of heatmaps."""
    success = extract_and_save_csvs()
    
    if success:
        generate_heatmap('roc_auc')
        generate_heatmap('pr_auc')
        print("\nAll tasks completed successfully!")

if __name__ == '__main__':
    main()