"""
generate_experiment2_scripts.py
-------------------------------
Generates one Windows batch (.bat) launcher per generator for Experiment 2
(Training Checkpoint Dynamics).

Experiment 2 crosses 13 generators with 3 Pythia detectors at 6 training
checkpoints each, giving 234 independent runs of run.py. The checkpoints are
addressed through HuggingFace revision identifiers (e.g. "step1000"), which
run.py loads via --detector_revision. Writing one batch file per generator
lets the workload be spread across several machines/GPUs and run in parallel,
whilst each file remains a self-contained, sequential job.

The commands use the refactored run.py argument names:
    --generator_name    : model that writes the machine text
    --detector_name     : model that scores the text
    --detector_revision : HuggingFace revision (training checkpoint) of the detector
    --mask_filler_name  : model that writes the perturbation edits
"""

# 13 generators used in Experiment 2 (identical to Experiment 1).
# Keys are HuggingFace model IDs; values are filesystem-safe aliases used for
# the batch filename and the results subfolder.
GENERATORS = {
    "gpt2-xl": "gpt2-xl",
    "facebook/opt-1.3b": "opt-1.3b",
    "EleutherAI/gpt-neo-1.3B": "gpt-neo-1.3B",
    "gpt2-large": "gpt2-large",
    "gpt2-medium": "gpt2-medium",
    "facebook/opt-350m": "opt-350m",
    "facebook/opt-125m": "opt-125m",
    "EleutherAI/gpt-neo-125M": "gpt-neo-125M",
    "gpt2": "gpt2",
    "distilgpt2": "distilgpt2",
    "HuggingFaceTB/SmolLM2-1.7B": "SmolLM2-1.7B",
    "HuggingFaceTB/SmolLM2-360M": "SmolLM2-360M",
    "HuggingFaceTB/SmolLM2-135M": "SmolLM2-135M"
}

# 3 Pythia detector scales used in Experiment 2 (HuggingFace model IDs).
DETECTORS = [
    "EleutherAI/pythia-1.4b",
    "EleutherAI/pythia-410m",
    "EleutherAI/pythia-70m"
]

# HuggingFace revision identifiers for the six Pythia training checkpoints,
# spanning early training (step1000) through full convergence (step143000).
CHECKPOINT_REVISIONS = [
    "step1000", "step5000", "step10000",
    "step50000", "step100000", "step143000"
]

# Template for a single run; {clean_name}, {gen}, {det} and {step} are
# substituted at write time. Hyperparameters match the shared dissertation
# configuration: semantic (T5-large) control, 50 perturbations, 250+250
# samples, WritingPrompts. --half loads the detector in float16 so the 1.4B
# model fits alongside the mask-filler on a single GPU.
BASE_COMMAND = (
    "python run.py --output_name experiment2/stepsizes_{clean_name} "
    "--generator_name {gen} "
    "--detector_name {det} "
    "--detector_revision {step} "
    "--mask_filler_name t5-large "
    "--n_perturbation_list 50 "
    "--n_samples 250 "
    "--pct_words_masked 0.3 "
    "--span_length 2 "
    "--dataset writing "
    "--half"
)


def write_batch_file(gen_full, clean_name):
    """
    Writes one .bat file that runs all detector/checkpoint configurations
    (3 detectors x 6 checkpoints) for a single generator.

    Args:
        gen_full (str): HuggingFace model ID of the generator.
        clean_name (str): Filesystem-safe alias used for the filename and output folder.

    Returns:
        str: The name of the batch file that was created.
    """
    filename = f"run_{clean_name}.bat"
    with open(filename, "w") as f:
        f.write("@echo off\n")
        f.write(f"echo Starting experiments for Generator: {gen_full}\n\n")
        for det in DETECTORS:
            for step in CHECKPOINT_REVISIONS:
                cmd = BASE_COMMAND.format(clean_name=clean_name, gen=gen_full, det=det, step=step)
                f.write(cmd + "\n")
        f.write(f"\necho Finished all experiments for {gen_full}!\n")
        f.write("pause\n")
    return filename


def main():
    """Generates one batch launcher per generator and reports progress."""
    for gen_full, clean_name in GENERATORS.items():
        filename = write_batch_file(gen_full, clean_name)
        print(f"Created {filename}")
    print(f"\nAll {len(GENERATORS)} batch files generated successfully!")


if __name__ == '__main__':
    main()