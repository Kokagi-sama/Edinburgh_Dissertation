"""
dataset_loader.py
-----------------
Data loading and preprocessing script for the curvature-based detection pipeline.

This module is responsible for loading, cleaning, and formatting the human-authored 
text corpora. These human texts serve two critical purposes in the main pipeline:
1. They act as the negative class (human text) for the detector to score.
2. Their initial tokens are used as prompts to seed the generator model, 
   which produces the positive class (machine-generated text).
   
Currently supports the WritingPrompts corpus.
"""

import random

# Special separator token used for specific datasets (e.g., PubMed QA)
SEPARATOR = '<<<SEP>>>'

# Registry of supported human-text datasets
DATASETS = ['writing']

def process_prompt(prompt: str) -> str:
    """
    Cleans WritingPrompts specific metadata tags from the prompt string.
    Removes bracketed tags like '[ WP ]' (Writing Prompt) and '[ OT ]' (Off Topic)
    so the generator model receives a clean natural language prefix.
    """
    return prompt.replace('[ WP ]', '').replace('[ OT ]', '')

def process_spaces(human_text: str) -> str:
    """
    Normalizes punctuation, spacing, and quotation marks in the text.
    Ensures that punctuation marks are attached to the preceding words 
    (e.g., ' ,' becomes ',') and standardizes quote formatting to prevent 
    tokenization artifacts during model evaluation.
    """
    # Chained replacements enclosed in parentheses for clean PEP8 formatting.
    # Handles standard punctuation spacing and normalizes unicode quotes.
    return (human_text
        .replace(' ,', ',')
        .replace(' .', '.')
        .replace(' ?', '?')
        .replace(' !', '!')
        .replace(' ;', ';')
        .replace(' \u2018', '\u2018')   # Space + left single quote
        .replace(' \u2019 ', '\u2019')  # Space + right single quote + space
        .replace(' :', ':')
        .replace('<newline>', '\n')
        .replace(' \u201c', '\u201c')   # Space + left double quote
        .replace(' \u201d', '\u201d')   # Space + right double quote
        .replace('\u2019\u201c', '\u201c') 
        .replace('.. ', '... ')
        .replace(' )', ')')
        .replace('( ', '(')
        .replace(' n\u2019t', "n't")
        .replace(' i ', ' I ')
        .replace(' i\u2019', " I'")
        .replace('\'', '\u2019')        # Standardize apostrophes to right single quote
        .replace('\n ', '\n')
        .strip())

def load_writing(cache_dir=None):
    """
    Loads the WritingPrompts corpus from local storage.
    
    Reads the source prompts and target human-authored stories, cleans them,
    filters out NSFW content, and shuffles the dataset deterministically.
    
    Returns:
        list: A shuffled list of cleaned, human-authored texts.
    """
    writing_path = 'data/writingPrompts'
    
    # Load raw prompts and human stories
    with open(f'{writing_path}/valid.wp_source', 'r', encoding='utf-8') as f:
        prompts = f.readlines()
    with open(f'{writing_path}/valid.wp_target', 'r', encoding='utf-8') as f:
        human_stories = f.readlines()

    # Clean metadata tags from prompts
    prompts = [process_prompt(prompt) for prompt in prompts]
    
    # Combine prompts and stories, then normalize spacing/punctuation
    cleaned_human_texts = [
        process_spaces(prompt + " " + story) 
        for prompt, story in zip(prompts, human_stories)
    ]
    
    # Filter out Not-Safe-For-Work content to maintain a clean evaluation pool
    filtered_human_texts = [
        text for text in cleaned_human_texts 
        if 'nsfw' not in text.lower()
    ]

    # Deterministic shuffle for reproducible dataset splitting
    random.seed(0)
    random.shuffle(filtered_human_texts)
    
    return filtered_human_texts

def load(name, cache_dir, **kwargs):
    """
    Router function to load a specific human-text dataset by name.
    
    Args:
        name (str): The name of the dataset to load (must be in DATASETS).
        cache_dir (str): Directory for caching (passed to underlying loaders).
        **kwargs: Additional arguments passed to the specific load function.
        
    Returns:
        list: The loaded and preprocessed human texts.
        
    Raises:
        ValueError: If the requested dataset is not supported.
    """
    if name in DATASETS:
        # Dynamically call the appropriate load_<name> function
        load_fn = globals()[f'load_{name}']
        return load_fn(cache_dir=cache_dir, **kwargs)
    else:
        raise ValueError(f'Unknown dataset: {name}. Supported datasets: {DATASETS}')