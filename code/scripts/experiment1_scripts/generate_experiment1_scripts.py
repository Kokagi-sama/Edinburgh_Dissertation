"""
generate_experiment1_scripts.py
-------------------------------
Generates one Windows batch (.bat) launcher per generator for Experiment 1
(Cross-Detection and Model Scale).

Experiment 1 crosses 13 generators with 17 detectors, giving 221 independent
runs of run.py. Writing one batch file per generator lets the workload be
spread across several machines/GPUs and run in parallel, whilst each file
remains a self-contained, sequential job.

The commands use the refactored run.py argument names:
    --generator_name   : model that writes the machine text
    --detector_name    : model that scores the text
    --mask_filler_name : model that writes the perturbation edits
"""

# 13 generators used in Experiment 1.
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

# 17 detectors used in Experiment 1 (HuggingFace model IDs).
DETECTORS = [
    "gpt2-xl", "EleutherAI/pythia-1.4b", "facebook/opt-1.3b",
    "EleutherAI/gpt-neo-1.3B", "gpt2-large", "EleutherAI/pythia-410m",
    "gpt2-medium", "facebook/opt-350m", "EleutherAI/pythia-160m",
    "facebook/opt-125m", "EleutherAI/gpt-neo-125M", "gpt2",
    "distilgpt2", "EleutherAI/pythia-70m", "HuggingFaceTB/SmolLM2-1.7B",
    "HuggingFaceTB/SmolLM2-360M", "HuggingFaceTB/SmolLM2-135M"
]

# Template for a single run; {clean_name}, {gen} and {det} are substituted at
# write time. Hyperparameters match the shared dissertation configuration:
# semantic (T5-large) control, 50 perturbations, 250+250 samples, WritingPrompts.
BASE_COMMAND = (
    "python run.py --output_name experiment1/cross_{clean_name} "
    "--generator_name {gen} "
    "--detector_name {det} "
    "--mask_filler_name t5-large "
    "--n_perturbation_list 50 "
    "--n_samples 250 "
    "--pct_words_masked 0.3 "
    "--span_length 2 "
    "--dataset writing"
)


def write_batch_file(gen_full, clean_name):
    """
    Writes one .bat file that runs all 17 detector configurations for a single generator.

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
            cmd = BASE_COMMAND.format(clean_name=clean_name, gen=gen_full, det=det)
            f.write(cmd + "\n")
        f.write(f"\necho Finished all experiments for {gen_full}!\n")
        f.write("pause\n")
    return filename


def main():
    """Generates one batch launcher per generator and reports progress."""
    for gen_full, clean_name in GENERATORS.items():
        filename = write_batch_file(gen_full, clean_name)
        print(f"Created {filename}")
    print("\nAll 13 batch files generated successfully!")


if __name__ == '__main__':
    main()