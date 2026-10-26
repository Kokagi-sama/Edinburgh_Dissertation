"""
generate_expriment3_scripts.py
-------------------------------
Generates a Windows batch (.bat) launcher for Experiment 3
(Semantic Perturbation Scale).

Experiment 3 crosses one fixed generator (SmolLM2-1.7B) with 3 Pythia
detectors and 3 T5 mask-filler capacities, giving 9 independent runs of
run.py. The mask-filler is the manipulated constituent; the three T5 sizes
span roughly two orders of magnitude in parameter count.

The commands use the refactored run.py argument names:
    --generator_name   : model that writes the machine text
    --detector_name    : model that scores the text
    --mask_filler_name : model that writes the perturbation edits
"""

# Single generator of Experiment 3, fixed so synthetic-text quality is held
# constant. The value is a filesystem-safe alias for the filename and folder.
GENERATORS = {
    "HuggingFaceTB/SmolLM2-1.7B": "SmolLM2-1.7B"
}

# 3 Pythia detector scales used in Experiment 3 (HuggingFace model IDs).
DETECTORS = [
    "EleutherAI/pythia-1.4b",
    "EleutherAI/pythia-410m",
    "EleutherAI/pythia-70m"
]

# Ablation targets: three T5 mask-filler capacities.
MASK_FILLERS = ["t5-small", "t5-large", "t5-3b"]

# Template for a single run; placeholders are substituted at write time.
# Hyperparameters match the shared dissertation configuration (50
# perturbations, 250+250 samples, WritingPrompts). --batch_size 4 and --half
# load the models in float16 with small batches so the 3B mask-filler fits
# alongside the detector on a single GPU.
BASE_COMMAND = (
    "python run.py --output_name experiment3/perturbation_{clean_name}_{clean_det}_{mask_model} "
    "--generator_name {gen} "
    "--detector_name {det} "
    "--mask_filler_name {mask_model} "
    "--n_perturbation_list 50 "
    "--n_samples 250 "
    "--pct_words_masked 0.30 "
    "--span_length 2 "
    "--dataset writing "
    "--batch_size 4 "
    "--half"
)


def write_batch_file(gen_full, clean_name):
    """
    Writes one .bat file that runs all mask-filler capacities for every
    detector (3 detectors x 3 mask-fillers) against a single generator.

    Args:
        gen_full (str): HuggingFace model ID of the generator.
        clean_name (str): Filesystem-safe alias used for the filename and output folder.

    Returns:
        str: The name of the batch file that was created.
    """
    filename = f"run_{clean_name}_t5_perturbation.bat"
    with open(filename, "w") as f:
        f.write("@echo off\n")
        f.write("echo ==================================================\n")
        f.write("echo STARTING T5 MASK-FILLER ABLATION RUN\n")
        f.write("echo ==================================================\n\n")

        for det in DETECTORS:
            clean_det = det.split('/')[-1]

            f.write("echo --------------------------------------------------\n")
            f.write(f"echo Running T5 sizes for Detector: {clean_det}\n")
            f.write("echo --------------------------------------------------\n")

            for mask_model in MASK_FILLERS:
                cmd = BASE_COMMAND.format(
                    clean_name=clean_name,
                    clean_det=clean_det,
                    gen=gen_full,
                    det=det,
                    mask_model=mask_model
                )
                f.write(cmd + "\n")

            f.write("\n")

        f.write("echo ==================================================\n")
        f.write("echo SUCCESS: All tasks completed successfully!\n")
        f.write("echo ==================================================\n")
        f.write("pause\n")
    return filename


def main():
    """Generates one batch launcher per generator and reports progress."""
    for gen_full, clean_name in GENERATORS.items():
        filename = write_batch_file(gen_full, clean_name)
        print(f"Created unified batch script: {filename}")


if __name__ == '__main__':
    main()