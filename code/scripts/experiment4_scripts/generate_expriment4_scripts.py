"""
generate_experiment4_scripts.py
-------------------------------
Generates a Windows batch (.bat) launcher for Experiment 4
(Linguistic Perturbation Strategies).

Experiment 4 crosses one fixed generator (SmolLM2-1.7B) with 3 Pythia
detectors and 4 perturbation strategies, giving 12 independent runs of
run.py. The perturbation strategy is the manipulated constituent; the four
strategies span a spectrum from full semantic coherence (T5 mask-filling)
to pure token noise (random subword insertion).

The commands use the refactored run.py argument names:
    --generator_name   : model that writes the machine text
    --detector_name    : model that scores the text
    --mask_filler_name : model that writes the perturbation edits (control only)
"""

# Single generator of Experiment 4, fixed so synthetic-text quality is held
# constant at the strongest level tested in prior experiments.
GENERATORS = {
    "HuggingFaceTB/SmolLM2-1.7B": "SmolLM2-1.7B"
}

# 3 Pythia detector scales used in Experiment 4 (HuggingFace model IDs).
DETECTORS = [
    "EleutherAI/pythia-1.4b",
    "EleutherAI/pythia-410m",
    "EleutherAI/pythia-70m"
]

# Four perturbation strategies spanning the semantic-to-noise spectrum.
# Each entry is (label, extra_flags, skip_baselines).
# The control runs baselines; the three treatments skip them to avoid
# redundant computation since baselines are independent of perturbation strategy.
STRATEGIES = [
    ("1_control_t5",              "",                                    False),
    ("2_treatment_a_synonyms",    "--synonym_fills",                     True),
    ("3_treatment_b_random_words","--random_fills",                      True),
    ("4_treatment_c_random_tokens","--random_fills --random_fills_tokens", True),
]

# Shared hyperparameters matching the dissertation configuration:
# T5-large mask-filler (used only by the control), 50 perturbations,
# 250+250 samples, WritingPrompts, 30% masking in two-token spans.
SHARED_ARGS = (
    "--mask_filler_name t5-large "
    "--n_perturbation_list 50 "
    "--n_samples 250 "
    "--pct_words_masked 0.3 "
    "--span_length 2 "
    "--dataset writing"
)


def build_command(gen, det, clean_det, strategy_label, extra_flags, skip_baselines):
    """
    Constructs a single run.py command string for a given detector and strategy.

    Args:
        gen (str): HuggingFace model ID of the generator.
        det (str): HuggingFace model ID of the detector.
        clean_det (str): Filesystem-safe detector name for the output folder.
        strategy_label (str): Subfolder name identifying the perturbation strategy.
        extra_flags (str): Additional CLI flags specific to the strategy.
        skip_baselines (bool): Whether to append --skip_baselines.

    Returns:
        str: The fully formatted command string.
    """
    cmd = (
        f"python run.py "
        f"--output_name \"experiment4/{clean_det}/{strategy_label}\" "
        f"--generator_name {gen} "
        f"--detector_name {det} "
        f"{SHARED_ARGS}"
    )
    if extra_flags:
        cmd += f" {extra_flags}"
    if skip_baselines:
        cmd += " --skip_baselines"
    return cmd


def write_batch_file(gen_full, clean_name):
    """
    Writes one .bat file that runs all 4 perturbation strategies for every
    detector (3 detectors x 4 strategies) against a single generator.

    Args:
        gen_full (str): HuggingFace model ID of the generator.
        clean_name (str): Filesystem-safe alias used for the filename.

    Returns:
        str: The name of the batch file that was created.
    """
    filename = f"run_{clean_name}_perturbation_strategies.bat"
    with open(filename, "w") as f:
        f.write("@echo off\n")
        f.write("setlocal\n\n")
        f.write("echo ==================================================\n")
        f.write("echo STARTING PERTURBATION STRATEGY ABLATION RUN\n")
        f.write(f"echo Generator: {gen_full}\n")
        f.write("echo ==================================================\n\n")

        for det in DETECTORS:
            clean_det = det.split('/')[-1]

            f.write("echo --------------------------------------------------\n")
            f.write(f"echo Running all strategies for Detector: {clean_det}\n")
            f.write("echo --------------------------------------------------\n")

            for strategy_label, extra_flags, skip_baselines in STRATEGIES:
                cmd = build_command(gen_full, det, clean_det, strategy_label, extra_flags, skip_baselines)
                f.write(f"echo [{clean_det}] {strategy_label}...\n")
                f.write(cmd + "\n\n")

        f.write("echo ==================================================\n")
        f.write("echo SUCCESS: All perturbation strategy experiments completed!\n")
        f.write("echo Results saved in: experiment4/\n")
        f.write("echo ==================================================\n")
        f.write("pause\n")
    return filename


def main():
    """Generates one batch launcher per generator and reports progress."""
    for gen_full, clean_name in GENERATORS.items():
        filename = write_batch_file(gen_full, clean_name)
        print(f"Created batch script: {filename}")


if __name__ == '__main__':
    main()