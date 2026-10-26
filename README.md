# Small Language Models for AI-Generated Text Detection: A Comparative Analysis of Curvature Methods

**MSc Dissertation — School of Informatics, University of Edinburgh, 2026**
**Author:** Royceton Yeoh Tze Jian
**Supervisor:** Dr. Rik Sarkar

## Table of Contents
* [Description](#description)
* [Report](#report)
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
  * [Setting up the Environment](#setting-up-the-environment)
  * [Downloading the WritingPrompts Data](#downloading-the-writingprompts-data)
* [Project Structure](#project-structure)
* [Experiment Configuration](#experiment-configuration)
* [Usage](#usage)
  * [Running Experiments](#running-experiments)
  * [Regenerating the Plots](#regenerating-the-plots)
* [Acknowledgments](#acknowledgments)

## Description

AI text generators write by always choosing the most statistically "safe" and predictable next words. Because of this, AI-written text sits in a narrow, fragile groove: if you slightly scramble a few words of an AI essay, the AI's own confidence score for that text drops off a cliff. Human writing is naturally messier, so scrambling a few words does not cause the same dramatic drop. This difference in confidence drops is called **curvature**, and it is one of the most reliable ways to catch AI text without needing any training data or access to the generator.

However, the standard way of measuring curvature requires huge, expensive AI models that cannot fit on a normal laptop or phone. This dissertation asks a simple question: **do we actually need these giant models to measure the confidence drop?**

The answer is no. Across four controlled experiments, this project shows that smaller, cheaper AI models are actually *better* at spotting these confidence drops than giant ones; that partially trained models beat fully trained ones; that the expensive "smart" word-scrambling model only needs to be medium-sized; and that smart scrambling is not needed at all, since random word swapping works just as well. The final result is a blueprint for a lightweight, cheap AI-text detector that runs locally on standard consumer hardware.

The four experiments:

1. **Experiment 1 (Cross-Detection and Model Scale):** 13 generators × 17 detectors. Detection accuracy falls as the detector grows; 70–82 million parameter detectors exceed 0.90 AUROC against most unseen generators, whilst 1.7 billion parameter detectors fall to a coin flip.
2. **Experiment 2 (Training Checkpoint Dynamics):** partially trained detectors (1,000–10,000 training steps) outperform fully converged ones.
3. **Experiment 3 (Semantic Perturbation Scale):** perturbation quality rises with the scrambling model's size, but with clear diminishing returns beyond T5-large.
4. **Experiment 4 (Linguistic Perturbation Strategies):** semantically coherent edits are unnecessary; uniform random token substitution matches or exceeds the coherent control.

## Report

The written dissertation and its supporting documents are in the `report/` folder at the root of the project, alongside `code/`.

| Folder | Contents |
|---|---|
| `report/Final Report/` | The final submitted dissertation |
| `report/Halfway_Progress_Report/` | The mid-project progress report |
| `report/MSc Template/` | The MSc dissertation template used to build the final report |
| `report/Proposal Template/` | The template used for the initial project proposal |

The final report covers the method, the four experiments and their results. All code needed to reproduce those results and regenerate the figures is in `code/`; the sections below ([Getting Started](#getting-started) onwards) refer to that folder.

## Getting Started

### Prerequisites

* Python 3.9 or newer
* CUDA-compatible GPU (highly recommended for model inference)
* The libraries listed in `code/requirements.txt`

### Installation

1. Extract the provided ZIP file to your desired location.
2. Open a terminal and navigate into the `code` directory:

   ```bash
   cd PROJECT/code
   ```

### Setting up the Environment

It is recommended to use a Python virtual environment:

```bash
# 1. Create a Python virtual environment
python -m venv venv

# 2. Activate the virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows (Command Prompt):
# venv\Scripts\activate.bat
# On Windows (PowerShell):
# venv\Scripts\Activate.ps1

# 3. Install PyTorch with CUDA support (as noted in requirements.txt)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

# 4. Install the remaining dependencies
pip install -r requirements.txt

# 5. Download the NLTK data needed for the WordNet synonym perturbation
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

### Downloading the WritingPrompts Data

The detectors are evaluated on the WritingPrompts corpus, a collection of human-written creative stories. Download it from Kaggle:

🔗 <https://www.kaggle.com/datasets/ratthachat/writing-prompts>

The download provides the source/target file pairs (train, valid, test). The code reads the validation split. Create the following structure inside `code/` and place the two validation files there:

```text
code/
└── data/
    └── writingPrompts/
        ├── valid.wp_source   (the writing prompts)
        └── valid.wp_target   (the human-authored stories)
```

No further preprocessing is required: `dataset_loader.py` automatically strips metadata tags (e.g. `[ WP ]`), normalises punctuation and spacing, filters NSFW content, and shuffles the data deterministically.

## Project Structure

```text
PROJECT/
├── README.md
├── report/
│   ├── Final Report/                # Final submitted dissertation
│   ├── Halfway_Progress_Report/     # Mid-project progress report
│   ├── MSc Template/                # MSc dissertation template the final report is built from
│   └── Proposal Template/           # Template used for the initial project proposal
└── code/
    ├── dataset_loader.py            # Loads and cleans the WritingPrompts data
    ├── requirements.txt             # Python dependencies
    ├── run.py                       # Main evaluation script (curvature scoring,
    │                                #   perturbations, baselines)
    ├── plots/                       # Plotting scripts (figures/CSVs are written
    │   │                            #   next to each script when it is run)
    │   ├── experiment1_plots/
    │   │   ├── plot_experiment1_heatmaps.py
    │   │   └── plot_experiment1_scale_trends.py
    │   ├── experiment2_plots/
    │   │   ├── plot_experiment2_checkpoint_trends.py
    │   │   └── plot_experiment2_heatmaps.py
    │   ├── experiment3_plots/
    │   │   ├── plot_experiment3_bars_and_curves.py
    │   │   └── plot_experiment3_scale_trends.py
    │   └── experiment4_plots/
    │       ├── plot_experiment4_bars_and_curves.py
    │       └── plot_experiment4_strategy_trend.py
    └── scripts/                     # Generators that write one launcher (.bat)
        ├── experiment1_scripts/     #   per generator for each experiment matrix
        │   └── generate_experiment1_scripts.py
        ├── experiment2_scripts/
        │   └── generate_experiment2_scripts.py
        ├── experiment3_scripts/
        │   └── generate_expriment3_scripts.py
        └── experiment4_scripts/
            └── generate_expriment4_scripts.py
```

**Note:** the raw JSON results and the multi-gigabyte model checkpoints are not bundled. They can be regenerated by running the experiments (see [Usage](#usage)); the plotting scripts in `code/plots/` recreate every figure used in the final report (`report/Final Report/`) except the unified pipeline diagram (Figure 3.1), which was created manually using Miro.

## Experiment Configuration

`run.py` is configured entirely through command-line arguments. The arguments separate the three distinct models in the pipeline:

**Model selection**

| Argument | Role |
|---|---|
| `--generator_name` | The model that *writes* the machine text (e.g. `gpt2-medium`) |
| `--detector_name` | The model that *scores* the text (e.g. `EleutherAI/pythia-70m`) |
| `--detector_revision` | Optional training checkpoint (e.g. `step1000`), used in Experiment 2 |
| `--mask_filler_name` | The model that *writes the edits* for semantic perturbations (e.g. `t5-large`) |

**Perturbation strategies** (how the text is scrambled to test the confidence drop)

* *(default)* — semantic rewriting with the T5 mask-filler
* `--synonym_fills` — WordNet synonym substitution
* `--random_fills` — random real words from the corpus unigram distribution
* `--random_fills_tokens` — pure random subword noise

**Hyperparameters**

* `--pct_words_masked` — fraction of tokens masked (default `0.3`)
* `--span_length` — length of each masked span (default `2`)
* `--n_perturbation_list` — number of perturbed neighbours (`50` in this project)
* `--n_samples` — human/machine pairs per generator (`250` in this project)

## Usage

Run all commands from inside `PROJECT/code` with the virtual environment activated.

### Running Experiments

Each experiment is a matrix of `run.py` invocations. The generators in `scripts/` write one launcher per generator so the workload can be spread across machines or run sequentially:

```bash
# Generate the launchers for an experiment, then execute them
python scripts/experiment1_scripts/generate_experiment1_scripts.py
```

A single manual run looks like:

```bash
python run.py \
    --generator_name HuggingFaceTB/SmolLM2-1.7B \
    --detector_name EleutherAI/pythia-70m \
    --mask_filler_name t5-large \
    --dataset writing \
    --n_samples 250 \
    --n_perturbation_list 50 \
    --output_name experiment1/manual_test
```

Each run writes `args.json`, `raw_data.json` and `perturbation_50_d_results.json` (plus baseline results) into a timestamped folder under `results/`, which the plotting scripts then aggregate.

### Regenerating the Plots

Once the JSON results exist under `results/`, run the plotting scripts for the corresponding experiment. Each script reads the results, aggregates them into CSV pivot matrices, and writes the high-resolution figures alongside itself in `plots/`:

```bash
# Experiment 1
python plots/experiment1_plots/plot_experiment1_heatmaps.py
python plots/experiment1_plots/plot_experiment1_scale_trends.py

# Experiment 2
python plots/experiment2_plots/plot_experiment2_heatmaps.py
python plots/experiment2_plots/plot_experiment2_checkpoint_trends.py

# Experiment 3
python plots/experiment3_plots/plot_experiment3_scale_trends.py
python plots/experiment3_plots/plot_experiment3_bars_and_curves.py

# Experiment 4
python plots/experiment4_plots/plot_experiment4_bars_and_curves.py
python plots/experiment4_plots/plot_experiment4_strategy_trend.py
```

The plotting scripts are backwards-compatible: they read both the current argument names (`generator_name`, `detector_name`, `mask_filler_name`) and the legacy names (`base_model_name`, `scoring_model_name`, `mask_filling_model_name`) from older `args.json` files.

## Acknowledgments

* **Supervisor:** Dr. Rik Sarkar, for his invaluable tutelage, insightful feedback, and steadfast support throughout this research.
* **Family:** my parents, whose constant encouragement made this work possible.
* **Datasets & Models:** the creators of the [WritingPrompts corpus](https://www.kaggle.com/datasets/ratthachat/writing-prompts), and the teams behind Pythia, SmolLM2, OPT, GPT-Neo, GPT-2 and T5 for releasing open-weight models that made this comparative analysis possible.
