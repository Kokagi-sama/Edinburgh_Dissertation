"""
run.py
------
Main execution script for curvature-based machine-generated text detection.
This script implements the DetectGPT pipeline and various experimental ablations, including:
- Cross-detection across different generator and detector scales.
- Perturbation strategies (Semantic T5, WordNet Synonyms, Random Words, Random Tokens).
- Baseline zero-shot metrics (Log-Likelihood, Rank, Entropy).
- Supervised baseline evaluation (RoBERTa-based detectors).

The script outputs raw log-likelihoods, perturbed texts, and evaluation metrics 
(AUROC, PR-AUC) into a timestamped results directory for downstream analysis.
"""

import matplotlib.pyplot as plt
import numpy as np
import datasets
import transformers
import re
import torch
import torch.nn.functional as F
import tqdm
import random
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import argparse
import datetime
import os
import json
import functools
import dataset_loader
from multiprocessing.pool import ThreadPool
import time
import nltk
from nltk.corpus import wordnet

# Ensure WordNet is downloaded for the synonym perturbation strategy
try:
    wordnet.synsets('test')
except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-1.4')

# Colorblind-friendly palette for plotting ROC/PR curves
COLORS = ["#0072B2", "#009E73", "#D55E00", "#CC79A7", "#F0E442",
          "#56B4E9", "#E69F00", "#000000", "#0072B2", "#009E73",
          "#D55E00", "#CC79A7", "#F0E442", "#56B4E9", "#E69F00"]

# Regex to identify T5 mask tokens (e.g., <extra_id_0>, <extra_id_1>)
pattern = re.compile(r"<extra_id_\d+>")

def load_generator():
    """Moves the generator model to the GPU for faster text sampling."""
    print('MOVING GENERATOR TO GPU...', end='', flush=True)
    start = time.time()
    try:
        mask_filler.cpu()
    except NameError:
        pass
    generator_model.to(DEVICE)
    print(f'DONE ({time.time() - start:.2f}s)')

def load_detector():
    """Moves the detector model to the GPU for faster log-likelihood scoring."""
    print('MOVING DETECTOR TO GPU...', end='', flush=True)
    start = time.time()
    try:
        mask_filler.cpu()
    except NameError:
        pass
    detector_model.to(DEVICE)
    print(f'DONE ({time.time() - start:.2f}s)')

def load_mask_filler():
    """
    Moves the mask-filler model (e.g., T5) to the GPU. 
    Skips loading if the experiment uses non-neural perturbations 
    (random tokens or WordNet synonyms) to save VRAM.
    """
    print('MOVING MASK-FILLER TO GPU...', end='', flush=True)
    start = time.time()
    try:
        generator_model.cpu()
    except NameError:
        pass
    try:
        detector_model.cpu()
    except NameError:
        pass
        
    if not args.random_fills and not args.synonym_fills:
        mask_filler.to(DEVICE)
        print(f'DONE ({time.time() - start:.2f}s)')
    else:
        print('SKIPPED (Using custom perturbations)')

def tokenize_and_mask(text, span_length, pct, ceil_pct=False):
    """
    Masks a percentage of tokens in the text for perturbation.
    Masks are applied in contiguous spans of `span_length` tokens, 
    ensuring they are separated by at least `buffer_size` tokens.
    Replaces masks with `<extra_id_N>` tokens for T5 infilling.
    """
    tokens = text.split(' ')
    mask_string = '<<<mask>>>'
    n_spans = pct * len(tokens) / (span_length + args.buffer_size * 2)
    if ceil_pct:
        n_spans = np.ceil(n_spans)
    n_spans = int(n_spans)
    
    n_masks = 0
    while n_masks < n_spans:
        start = np.random.randint(0, len(tokens) - span_length)
        end = start + span_length
        search_start = max(0, start - args.buffer_size)
        search_end = min(len(tokens), end + args.buffer_size)
        # Ensure masks don't overlap or sit too close to each other
        if mask_string not in tokens[search_start:search_end]:
            tokens[start:end] = [mask_string]
            n_masks += 1
            
    # Replace generic mask strings with T5-specific <extra_id_N> tokens
    num_filled = 0
    for idx, token in enumerate(tokens):
        if token == mask_string:
            tokens[idx] = f'<extra_id_{num_filled}>'
            num_filled += 1
            
    assert num_filled == n_masks, f"num_filled {num_filled} != n_masks {n_masks}"
    text = ' '.join(tokens)
    return text

def count_masks(texts):
    """Counts the number of `<extra_id_N>` mask tokens in each text."""
    return [len([x for x in text.split() if x.startswith("<extra_id_")]) for text in texts]

def replace_masks(texts):
    """
    Uses the mask-filler model (e.g., T5) to generate replacements 
    for the `<extra_id_N>` masked spans via autoregressive sampling.
    """
    n_expected = count_masks(texts)
    stop_id = mask_filler_tokenizer.encode(f"<extra_id_{max(n_expected)}>")[0]
    tokens = mask_filler_tokenizer(texts, return_tensors="pt", padding=True).to(DEVICE)
    outputs = mask_filler.generate(
        **tokens, 
        max_length=150, 
        do_sample=True, 
        top_p=args.mask_top_p, 
        num_return_sequences=1, 
        eos_token_id=stop_id
    )
    return mask_filler_tokenizer.batch_decode(outputs, skip_special_tokens=False)

def extract_fills(texts):
    """
    Parses the raw output of the mask-filler model to extract the 
    generated text corresponding to each `<extra_id_N>` mask.
    Strips padding and EOS tokens.
    """
    texts = [x.replace("<pad>", "").replace("</s>", "").strip() for x in texts]
    extracted_fills = [pattern.split(x)[1:-1] for x in texts]
    extracted_fills = [[y.strip() for y in x] for x in extracted_fills]
    return extracted_fills

def apply_extracted_fills(masked_texts, extracted_fills):
    """
    Reconstructs the perturbed texts by inserting the extracted fills 
    back into the masked texts in place of the `<extra_id_N>` tokens.
    """
    tokens = [x.split(' ') for x in masked_texts]
    n_expected = count_masks(masked_texts)
    for idx, (text, fills, n) in enumerate(zip(tokens, extracted_fills, n_expected)):
        if len(fills) < n:
            tokens[idx] = [] # Discard if T5 failed to generate enough fills
        else:
            for fill_idx in range(n):
                text[text.index(f"<extra_id_{fill_idx}>")] = fills[fill_idx]
    texts = [" ".join(x) for x in tokens]
    return texts

def perturb_texts_(texts, span_length, pct, ceil_pct=False):
    """
    Core perturbation logic. Applies one of four perturbation strategies:
    1. Synonym replacement (WordNet) - Treatment A
    2. Semantic mask-filling (T5) - Control
    3. Random token noise (subwords) - Treatment C
    4. Random real words (unigram sampling) - Treatment B
    """
    # Treatment A: synonym replacement (WordNet)
    if args.synonym_fills:
        perturbed_texts = []
        for text in texts:
            words = text.split(' ')
            n_masks = max(1, int(len(words) * pct))
            mask_indices = random.sample(range(len(words)), min(n_masks, len(words)))
            for idx in mask_indices:
                word = words[idx]
                synonyms = []
                for syn in wordnet.synsets(word):
                    for lm in syn.lemmas():
                        syn_word = lm.name().replace('_', ' ')
                        if syn_word.lower() != word.lower():
                            synonyms.append(syn_word)
                if synonyms:
                    words[idx] = random.choice(synonyms)
            perturbed_texts.append(' '.join(words))
        return perturbed_texts

    # Control: Semantic mask-filling (T5)
    if not args.random_fills:
        masked_texts = [tokenize_and_mask(x, span_length, pct, ceil_pct) for x in texts]
        raw_fills = replace_masks(masked_texts)
        extracted_fills = extract_fills(raw_fills)
        perturbed_texts = apply_extracted_fills(masked_texts, extracted_fills)
        
        # Retry logic for texts where T5 failed to generate the correct number of fills
        attempts = 1
        while '' in perturbed_texts:
            idxs = [idx for idx, x in enumerate(perturbed_texts) if x == '']
            print(f'WARNING: {len(idxs)} texts have no fills. Trying again [attempt {attempts}].')
            masked_texts = [tokenize_and_mask(x, span_length, pct, ceil_pct) for idx, x in enumerate(texts) if idx in idxs]
            raw_fills = replace_masks(masked_texts)
            extracted_fills = extract_fills(raw_fills)
            new_perturbed_texts = apply_extracted_fills(masked_texts, extracted_fills)
            for idx, x in zip(idxs, new_perturbed_texts):
                perturbed_texts[idx] = x
            attempts += 1
            
    # Treatments B & C: Random perturbations
    else:
        if args.random_fills_tokens:
            # Treatment C: pure noise (random tokens/subwords)
            tokens = detector_tokenizer(texts, return_tensors="pt", padding=True).to(DEVICE)
            valid_tokens = tokens.input_ids != detector_tokenizer.pad_token_id
            replace_pct = args.pct_words_masked * (args.span_length / (args.span_length + 2 * args.buffer_size))
            random_mask = torch.rand(tokens.input_ids.shape, device=DEVICE) < replace_pct
            random_mask &= valid_tokens
            random_tokens = torch.randint(0, detector_tokenizer.vocab_size, (random_mask.sum(),), device=DEVICE)
            
            # Ensure we don't accidentally insert special tokens (like <eos> or <pad>)
            while any(detector_tokenizer.decode(x) in detector_tokenizer.all_special_tokens for x in random_tokens):
                random_tokens = torch.randint(0, detector_tokenizer.vocab_size, (random_mask.sum(),), device=DEVICE)
                
            tokens.input_ids[random_mask] = random_tokens
            perturbed_texts = detector_tokenizer.batch_decode(tokens.input_ids, skip_special_tokens=True)
        else:
            # Treatment B: statistical plausibility (random real words from corpus)
            masked_texts = [tokenize_and_mask(x, span_length, pct, ceil_pct) for x in texts]
            perturbed_texts = masked_texts
            for idx, text in enumerate(perturbed_texts):
                filled_text = text
                for fill_idx in range(count_masks([text])[0]):
                    fill = random.sample(FILL_DICTIONARY, span_length)
                    filled_text = filled_text.replace(f"<extra_id_{fill_idx}>", " ".join(fill))
                assert count_masks([filled_text])[0] == 0, "Failed to replace all masks"
                perturbed_texts[idx] = filled_text
                
    return perturbed_texts

def perturb_texts(texts, span_length, pct, ceil_pct=False):
    """
    Wrapper that applies `perturb_texts_` in batches (chunks) 
    to avoid GPU out-of-memory errors during perturbation generation.
    """
    chunk_size = args.chunk_size
    if '11b' in mask_filler_name:
        chunk_size //= 2 # Halve batch size for massive models
        
    outputs = []
    for i in tqdm.tqdm(range(0, len(texts), chunk_size), desc="Applying perturbations"):
        outputs.extend(perturb_texts_(texts[i:i + chunk_size], span_length, pct, ceil_pct=ceil_pct))
    return outputs

def drop_last_word(text):
    """Removes the last word from a text (used for prompt formatting)."""
    return ' '.join(text.split(' ')[:-1])

def sample_from_model(texts, min_words=55, prompt_tokens=30):
    """
    Generates machine-text samples using the generator model.
    Takes the first `prompt_tokens` of the human texts as prompts and 
    generates continuations until they reach at least `min_words`.
    """
    if args.dataset == 'pubmed':
        texts = [t[:t.index(dataset_loader.SEPARATOR)] for t in texts]
        
    all_encoded = generator_tokenizer(texts, return_tensors="pt", padding=True).to(DEVICE)
    all_encoded = {key: value[:, :prompt_tokens] for key, value in all_encoded.items()}
    
    decoded = ['' for _ in range(len(texts))]
    tries = 0
    
    # Regenerate until all samples meet the minimum word count
    while (m := min(len(x.split()) for x in decoded)) < min_words:
        if tries != 0:
            print()
            print(f"min words: {m}, needed {min_words}, regenerating (try {tries})")
            
        sampling_kwargs = {}
        if args.do_top_p:
            sampling_kwargs['top_p'] = args.top_p
        elif args.do_top_k:
            sampling_kwargs['top_k'] = args.top_k
            
        min_length = 50 if args.dataset in ['pubmed'] else 150
        outputs = generator_model.generate(
            **all_encoded, 
            min_length=min_length, 
            max_length=200, 
            do_sample=True, 
            **sampling_kwargs, 
            pad_token_id=generator_tokenizer.eos_token_id, 
            eos_token_id=generator_tokenizer.eos_token_id
        )
        decoded = generator_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        tries += 1
        
    return decoded

def get_likelihood(logits, labels):
    """
    Computes the mean log-likelihood of the labels given the logits.
    (Utility function, largely superseded by HuggingFace's built-in `.loss`).
    """
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1
    logits = logits.view(-1, logits.shape[-1])[:-1]
    labels = labels.view(-1)[1:]
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    log_likelihood = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    return log_likelihood.mean()

def get_ll(text):
    """
    Computes the mean log-likelihood of a single text under the detector model.
    Returns the negative of the HuggingFace cross-entropy loss.
    """
    with torch.no_grad():
        tokenized = detector_tokenizer(text, return_tensors="pt").to(DEVICE)
        labels = tokenized.input_ids
        return -detector_model(**tokenized, labels=labels).loss.item()

def get_lls(texts):
    """Computes the mean log-likelihood for a list of texts."""
    return [get_ll(text) for text in texts]

def get_rank(text, log=False):
    """
    Computes the average rank of each observed token in the detector's 
    likelihood ordering (1-indexed). Optionally returns the log-rank.
    """
    with torch.no_grad():
        tokenized = detector_tokenizer(text, return_tensors="pt").to(DEVICE)
        logits = detector_model(**tokenized).logits[:, :-1]
        labels = tokenized.input_ids[:, 1:]
        
        # Find the rank of the actual label token at each timestep
        matches = (logits.argsort(-1, descending=True) == labels.unsqueeze(-1)).nonzero()
        assert matches.shape[1] == 3, f"Expected 3 dimensions in matches tensor, got {matches.shape}"
        
        ranks, timesteps = matches[:, -1], matches[:, -2]
        assert (timesteps == torch.arange(len(timesteps)).to(timesteps.device)).all(), "Expected one match per timestep"
        
        ranks = ranks.float() + 1 # Convert 0-indexed to 1-indexed rank
        if log:
            ranks = torch.log(ranks)
        return ranks.float().mean().item()

def get_entropy(text):
    """Computes the average token-level entropy of the text under the detector."""
    with torch.no_grad():
        tokenized = detector_tokenizer(text, return_tensors="pt").to(DEVICE)
        logits = detector_model(**tokenized).logits[:, :-1]
        neg_entropy = F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)
        return -neg_entropy.sum(-1).mean().item()

def get_roc_metrics(real_preds, sample_preds):
    """
    Computes the False Positive Rate, True Positive Rate, and AUROC 
    for the given human (real) and machine (sample) predictions.
    """
    fpr, tpr, _ = roc_curve([0] * len(real_preds) + [1] * len(sample_preds), real_preds + sample_preds)
    roc_auc = auc(fpr, tpr)
    return fpr.tolist(), tpr.tolist(), float(roc_auc)

def get_precision_recall_metrics(real_preds, sample_preds):
    """
    Computes the Precision, Recall, and PR-AUC 
    for the given human (real) and machine (sample) predictions.
    """
    precision, recall, _ = precision_recall_curve([0] * len(real_preds) + [1] * len(sample_preds), real_preds + sample_preds)
    pr_auc = auc(recall, precision)
    return precision.tolist(), recall.tolist(), float(pr_auc)

def save_roc_curves(experiments):
    """Plots and saves the ROC curves for all evaluated experiments."""
    plt.clf()
    for experiment, color in zip(experiments, COLORS):
        metrics = experiment["metrics"]
        plt.plot(metrics["fpr"], metrics["tpr"], label=f"{experiment['name']}, roc_auc={metrics['roc_auc']:.3f}", color=color)
        print(f"{experiment['name']} roc_auc: {metrics['roc_auc']:.3f}")
    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves ({scorer_name} - {args.mask_filler_name})')
    plt.legend(loc="lower right", fontsize=6)
    plt.savefig(f"{SAVE_FOLDER}/roc_curves.png")

def save_ll_histograms(experiments):
    """Plots and saves histograms of raw log-likelihoods for original vs perturbed texts."""
    plt.clf()
    for experiment in experiments:
        try:
            results = experiment["raw_results"]
            plt.figure(figsize=(20, 6))
            plt.subplot(1, 2, 1)
            plt.hist([r["sampled_ll"] for r in results], alpha=0.5, bins='auto', label='sampled')
            plt.hist([r["perturbed_sampled_ll"] for r in results], alpha=0.5, bins='auto', label='perturbed sampled')
            plt.xlabel("log likelihood")
            plt.ylabel('count')
            plt.legend(loc='upper right')
            plt.subplot(1, 2, 2)
            plt.hist([r["original_ll"] for r in results], alpha=0.5, bins='auto', label='original')
            plt.hist([r["perturbed_original_ll"] for r in results], alpha=0.5, bins='auto', label='perturbed original')
            plt.xlabel("log likelihood")
            plt.ylabel('count')
            plt.legend(loc='upper right')
            plt.savefig(f"{SAVE_FOLDER}/ll_histograms_{experiment['name']}.png")
        except:
            pass

def save_llr_histograms(experiments):
    """Plots and saves histograms of log-likelihood ratios (curvature proxies)."""
    plt.clf()
    for experiment in experiments:
        try:
            results = experiment["raw_results"]
            plt.figure(figsize=(20, 6))
            plt.subplot(1, 2, 1)
            for r in results:
                r["sampled_llr"] = r["sampled_ll"] - r["perturbed_sampled_ll"]
                r["original_llr"] = r["original_ll"] - r["perturbed_original_ll"]
            plt.hist([r["sampled_llr"] for r in results], alpha=0.5, bins='auto', label='sampled')
            plt.hist([r["original_llr"] for r in results], alpha=0.5, bins='auto', label='original')
            plt.xlabel("log likelihood ratio")
            plt.ylabel('count')
            plt.legend(loc='upper right')
            plt.savefig(f"{SAVE_FOLDER}/llr_histograms_{experiment['name']}.png")
        except:
            pass

def get_perturbation_results(span_length=10, n_perturbations=1, n_samples=500):
    """
    Orchestrates the perturbation phase and the scoring phase.
    1. Generates `n_perturbations` for each human and machine text.
    2. Computes the log-likelihood of the original and perturbed texts.
    3. Returns a dictionary containing the raw texts and their log-likelihoods.
    """
    load_mask_filler()
    torch.manual_seed(0)
    np.random.seed(0)
    results = []
    original_text = data["original"]
    sampled_text = data["sampled"]
    
    perturb_fn = functools.partial(perturb_texts, span_length=span_length, pct=args.pct_words_masked)
    
    p_sampled_text = perturb_fn([x for x in sampled_text for _ in range(n_perturbations)])
    p_original_text = perturb_fn([x for x in original_text for _ in range(n_perturbations)])
    
    assert len(p_sampled_text) == len(sampled_text) * n_perturbations
    assert len(p_original_text) == len(original_text) * n_perturbations
    
    for idx in range(len(original_text)):
        results.append({
            "original": original_text[idx],
            "sampled": sampled_text[idx],
            "perturbed_sampled": p_sampled_text[idx * n_perturbations: (idx + 1) * n_perturbations],
            "perturbed_original": p_original_text[idx * n_perturbations: (idx + 1) * n_perturbations]
        })
        
    load_detector()
    for res in tqdm.tqdm(results, desc="Computing log likelihoods"):
        p_sampled_ll = get_lls(res["perturbed_sampled"])
        p_original_ll = get_lls(res["perturbed_original"])
        res["original_ll"] = get_ll(res["original"])
        res["sampled_ll"] = get_ll(res["sampled"])
        res["all_perturbed_sampled_ll"] = p_sampled_ll
        res["all_perturbed_original_ll"] = p_original_ll
        res["perturbed_sampled_ll"] = np.mean(p_sampled_ll)
        res["perturbed_original_ll"] = np.mean(p_original_ll)
        res["perturbed_sampled_ll_std"] = np.std(p_sampled_ll) if len(p_sampled_ll) > 1 else 1
        res["perturbed_original_ll_std"] = np.std(p_original_ll) if len(p_original_ll) > 1 else 1
        
    return results

def run_perturbation_experiment(results, criterion, span_length=10, n_perturbations=1, n_samples=500):
    """
    Evaluates the detection performance using the perturbation results.
    Computes the curvature score:
    - criterion 'd': raw log-likelihood difference
    - criterion 'z': standardized difference (divided by std dev)
    Calculates and returns AUROC and PR-AUC metrics.
    """
    predictions = {'real': [], 'samples': []}
    for res in results:
        if criterion == 'd':
            predictions['real'].append(res['original_ll'] - res['perturbed_original_ll'])
            predictions['samples'].append(res['sampled_ll'] - res['perturbed_sampled_ll'])
        elif criterion == 'z':
            if res['perturbed_original_ll_std'] == 0:
                res['perturbed_original_ll_std'] = 1
            if res['perturbed_sampled_ll_std'] == 0:
                res['perturbed_sampled_ll_std'] = 1
            predictions['real'].append((res['original_ll'] - res['perturbed_original_ll']) / res['perturbed_original_ll_std'])
            predictions['samples'].append((res['sampled_ll'] - res['perturbed_sampled_ll']) / res['perturbed_sampled_ll_std'])
            
    fpr, tpr, roc_auc = get_roc_metrics(predictions['real'], predictions['samples'])
    p, r, pr_auc = get_precision_recall_metrics(predictions['real'], predictions['samples'])
    name = f'perturbation_{n_perturbations}_{criterion}'
    print(f"{name} ROC AUC: {roc_auc}, PR AUC: {pr_auc}")
    
    return {
        'name': name,
        'predictions': predictions,
        'info': {'pct_words_masked': args.pct_words_masked, 'span_length': span_length,
                 'n_perturbations': n_perturbations, 'n_samples': n_samples},
        'raw_results': results,
        'metrics': {'roc_auc': roc_auc, 'fpr': fpr, 'tpr': tpr},
        'pr_metrics': {'pr_auc': pr_auc, 'precision': p, 'recall': r},
        'loss': 1 - pr_auc,
    }

def run_baseline_threshold_experiment(criterion_fn, name, n_samples=500):
    """
    Evaluates baseline zero-shot detection methods that do not use perturbations.
    Applies `criterion_fn` (e.g., log-likelihood, rank, entropy) directly to the texts.
    """
    torch.manual_seed(0)
    np.random.seed(0)
    results = []
    for batch in tqdm.tqdm(range(n_samples // batch_size), desc=f"Computing {name} criterion"):
        original_text = data["original"][batch * batch_size:(batch + 1) * batch_size]
        sampled_text = data["sampled"][batch * batch_size:(batch + 1) * batch_size]
        for idx in range(len(original_text)):
            results.append({
                "original": original_text[idx],
                "original_crit": criterion_fn(original_text[idx]),
                "sampled": sampled_text[idx],
                "sampled_crit": criterion_fn(sampled_text[idx]),
            })
            
    predictions = {'real': [x["original_crit"] for x in results], 'samples': [x["sampled_crit"] for x in results]}
    fpr, tpr, roc_auc = get_roc_metrics(predictions['real'], predictions['samples'])
    p, r, pr_auc = get_precision_recall_metrics(predictions['real'], predictions['samples'])
    print(f"{name}_threshold ROC AUC: {roc_auc}, PR AUC: {pr_auc}")
    
    return {'name': f'{name}_threshold', 'predictions': predictions, 'info': {'n_samples': n_samples},
            'raw_results': results, 'metrics': {'roc_auc': roc_auc, 'fpr': fpr, 'tpr': tpr},
            'pr_metrics': {'pr_auc': pr_auc, 'precision': p, 'recall': r}, 'loss': 1 - pr_auc}

def strip_newlines(text):
    """Removes newlines from text, replacing them with spaces."""
    return ' '.join(text.split())

def trim_to_shorter_length(texta, textb):
    """Truncates two texts to the length of the shorter one (word-wise)."""
    shorter_length = min(len(texta.split(' ')), len(textb.split(' ')))
    texta = ' '.join(texta.split(' ')[:shorter_length])
    textb = ' '.join(textb.split(' ')[:shorter_length])
    return texta, textb

def truncate_to_substring(text, substring, idx_occurrence):
    """Truncates text at the Nth occurrence of a specific substring."""
    assert idx_occurrence > 0
    idx = -1
    for _ in range(idx_occurrence):
        idx = text.find(substring, idx + 1)
        if idx == -1:
            return text
    return text[:idx]

def generate_samples(raw_data, batch_size):
    """
    Generates the paired dataset of human and machine texts.
    Uses human texts as prompts to generate machine continuations, 
    then trims both to the exact same length for fair comparison.
    """
    torch.manual_seed(42)
    np.random.seed(42)
    data = {"original": [], "sampled": []}
    
    for batch in range(len(raw_data) // batch_size):
        print('Generating samples for batch', batch, 'of', len(raw_data) // batch_size)
        original_text = raw_data[batch * batch_size:(batch + 1) * batch_size]
        sampled_text = sample_from_model(original_text, min_words=30 if args.dataset in ['pubmed'] else 55)
        
        for o, s in zip(original_text, sampled_text):
            if args.dataset == 'pubmed':
                s = truncate_to_substring(s, 'Question:', 2)
                o = o.replace(dataset_loader.SEPARATOR, ' ')
            o, s = trim_to_shorter_length(o, s)
            data["original"].append(o)
            data["sampled"].append(s)
            
    if args.pre_perturb_pct > 0:
        print(f'APPLYING {args.pre_perturb_pct}, {args.pre_perturb_span_length} PRE-PERTURBATIONS')
        load_mask_filler()
        data["sampled"] = perturb_texts(data["sampled"], args.pre_perturb_span_length, args.pre_perturb_pct, ceil_pct=True)
        load_generator()
        
    return data

def generate_data(dataset, key):
    """
    Loads, cleans, and filters the source dataset, then calls 
    `generate_samples` to create the final human/machine pairs.
    """
    if dataset in dataset_loader.DATASETS:
        data = dataset_loader.load(dataset, cache_dir)
    else:
        data = datasets.load_dataset(dataset, split='train', cache_dir=cache_dir)[key]
        
    data = list(dict.fromkeys(data)) # Remove duplicates
    data = [x.strip() for x in data]
    data = [strip_newlines(x) for x in data]
    
    if dataset in ['writing', 'squad', 'xsum']:
        long_data = [x for x in data if len(x.split()) > 250]
        if len(long_data) > 0:
            data = long_data
            
    random.seed(0)
    random.shuffle(data)
    data = data[:5_000]
    
    # Filter by max token length to prevent OOM during mask-filling
    tokenized_data = preproc_tokenizer(data)
    data = [x for x, y in zip(data, tokenized_data["input_ids"]) if len(y) <= 512]
    
    print(f"Total number of samples: {len(data)}")
    print(f"Average number of words: {np.mean([len(x.split()) for x in data])}")
    return generate_samples(data[:n_samples], batch_size=batch_size)

def load_causal_model_and_tokenizer(name, revision=None):
    """
    Loads a causal language model (generator/detector) and its tokenizer 
    from HuggingFace, handling specific quirks for models like OPT and GPT-J.
    """
    print(f'Loading model {name} ' + (f'(revision: {revision})...' if revision else '...'))
    model_kwargs = {}
    if 'gpt-j' in name or 'neox' in name:
        model_kwargs.update(dict(torch_dtype=torch.float16))
    if 'gpt-j' in name:
        model_kwargs.update(dict(revision='float16'))
    elif revision is not None:
        model_kwargs.update(dict(revision=revision))
        
    model = transformers.AutoModelForCausalLM.from_pretrained(name, **model_kwargs, cache_dir=cache_dir)
    
    optional_tok_kwargs = {}
    if "facebook/opt-" in name:
        print("Using non-fast tokenizer for OPT")
        optional_tok_kwargs['fast'] = False
    if args.dataset in ['pubmed']:
        optional_tok_kwargs['padding_side'] = 'left'
    if revision is not None:
        optional_tok_kwargs['revision'] = revision
        
    tokenizer = transformers.AutoTokenizer.from_pretrained(name, **optional_tok_kwargs, cache_dir=cache_dir)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer

def eval_supervised(data, model):
    """
    Evaluates a supervised baseline detector (e.g., RoBERTa-based OpenAI detector)
    on the generated human/machine dataset.
    """
    print(f'Beginning supervised evaluation with {model}...')
    supervised_detector = transformers.AutoModelForSequenceClassification.from_pretrained(model, cache_dir=cache_dir).to(DEVICE)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model, cache_dir=cache_dir)
    real, fake = data['original'], data['sampled']
    
    with torch.no_grad():
        real_preds = []
        for batch in tqdm.tqdm(range(len(real) // batch_size), desc="Evaluating real"):
            batch_real = real[batch * batch_size:(batch + 1) * batch_size]
            batch_real = tokenizer(batch_real, padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEVICE)
            real_preds.extend(supervised_detector(**batch_real).logits.softmax(-1)[:, 0].tolist())
            
        fake_preds = []
        for batch in tqdm.tqdm(range(len(fake) // batch_size), desc="Evaluating fake"):
            batch_fake = fake[batch * batch_size:(batch + 1) * batch_size]
            batch_fake = tokenizer(batch_fake, padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEVICE)
            fake_preds.extend(supervised_detector(**batch_fake).logits.softmax(-1)[:, 0].tolist())
            
    predictions = {'real': real_preds, 'samples': fake_preds}
    fpr, tpr, roc_auc = get_roc_metrics(real_preds, fake_preds)
    p, r, pr_auc = get_precision_recall_metrics(real_preds, fake_preds)
    print(f"{model} ROC AUC: {roc_auc}, PR AUC: {pr_auc}")
    
    del supervised_detector
    torch.cuda.empty_cache()
    
    return {'name': model, 'predictions': predictions, 'info': {'n_samples': n_samples},
            'metrics': {'roc_auc': roc_auc, 'fpr': fpr, 'tpr': tpr},
            'pr_metrics': {'pr_auc': pr_auc, 'precision': p, 'recall': r}, 'loss': 1 - pr_auc}

if __name__ == '__main__':
    DEVICE = "cuda"
    parser = argparse.ArgumentParser()
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, default="writing")
    parser.add_argument('--dataset_key', type=str, default="document")
    
    # Perturbation arguments
    parser.add_argument('--pct_words_masked', type=float, default=0.3) 
    parser.add_argument('--span_length', type=int, default=2)
    parser.add_argument('--n_samples', type=int, default=200)
    parser.add_argument('--n_perturbation_list', type=str, default="1,10")
    parser.add_argument('--n_perturbation_rounds', type=int, default=1)
    parser.add_argument('--buffer_size', type=int, default=1)
    parser.add_argument('--mask_top_p', type=float, default=1.0)
    parser.add_argument('--pre_perturb_pct', type=float, default=0.0)
    parser.add_argument('--pre_perturb_span_length', type=int, default=5)
    parser.add_argument('--random_fills', action='store_true', help='Use random words from corpus')
    parser.add_argument('--random_fills_tokens', action='store_true', help='Use random subword tokens')
    parser.add_argument('--synonym_fills', action='store_true', help='Use WordNet synonyms for perturbation (Treatment A)')
    
    # Model arguments
    parser.add_argument('--generator_name', type=str, default="gpt2-medium", help='Model that writes the machine text')
    parser.add_argument('--detector_name', type=str, default="", help='Model that scores the text (defaults to the generator)')
    parser.add_argument('--mask_filler_name', type=str, default="t5-large", help='Model that writes the perturbation edits')
    parser.add_argument('--generator_revision', type=str, default=None)
    parser.add_argument('--detector_revision', type=str, default=None)
    
    # Generation arguments
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--chunk_size', type=int, default=20)
    parser.add_argument('--n_similarity_samples', type=int, default=20)
    parser.add_argument('--int8', action='store_true')
    parser.add_argument('--half', action='store_true')
    parser.add_argument('--base_half', action='store_true')
    parser.add_argument('--do_top_k', action='store_true')
    parser.add_argument('--top_k', type=int, default=40)
    parser.add_argument('--do_top_p', action='store_true')
    parser.add_argument('--top_p', type=float, default=0.96)
    
    # Execution arguments
    parser.add_argument('--output_name', type=str, default="")
    parser.add_argument('--baselines_only', action='store_true')
    parser.add_argument('--skip_baselines', action='store_true')
    parser.add_argument('--cache_dir', type=str, default="~/.cache")
    
    args = parser.parse_args()

    START_DATE = datetime.datetime.now().strftime('%Y-%m-%d')
    START_TIME = datetime.datetime.now().strftime('%H-%M-%S-%f')
    
    precision_string = "int8" if args.int8 else ("fp16" if args.half else "fp32")
    sampling_string = "top_k" if args.do_top_k else ("top_p" if args.do_top_p else "temp")
    output_subfolder = f"{args.output_name}/" if args.output_name else ""
    
    generator_name = args.generator_name.replace('/', '_')
    detector_string = (f"-{args.detector_name}" if args.detector_name else "").replace('/', '_')
    if args.detector_revision:
        detector_string += f"-{args.detector_revision}"
    if args.generator_revision:
        generator_name += f"-{args.generator_revision}"
        
    scorer_name = (args.detector_name or args.generator_name).replace('/', '_')
    
    SAVE_FOLDER = f"tmp_results/{output_subfolder}{generator_name}{detector_string}-{args.mask_filler_name}-{sampling_string}/{START_DATE}-{START_TIME}-{precision_string}-{args.pct_words_masked}-{args.n_perturbation_rounds}-{args.dataset}-{args.n_samples}"
    if not os.path.exists(SAVE_FOLDER):
        os.makedirs(SAVE_FOLDER)
    print(f"Saving results to absolute path: {os.path.abspath(SAVE_FOLDER)}")
    
    # Save arguments for reproducibility
    with open(os.path.join(SAVE_FOLDER, "args.json"), "w") as f:
        json.dump(args.__dict__, f, indent=4)
        
    mask_filler_name = args.mask_filler_name
    n_samples = args.n_samples
    batch_size = args.batch_size
    n_perturbation_list = [int(x) for x in args.n_perturbation_list.split(",")]
    n_perturbation_rounds = args.n_perturbation_rounds
    n_similarity_samples = args.n_similarity_samples
    
    cache_dir = os.path.expanduser(args.cache_dir)
    os.environ["XDG_CACHE_HOME"] = cache_dir
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    print(f"Using cache dir {cache_dir}")
    
    GPT2_TOKENIZER = transformers.GPT2Tokenizer.from_pretrained('gpt2', cache_dir=cache_dir)
    
    # Load Generator
    generator_model, generator_tokenizer = load_causal_model_and_tokenizer(args.generator_name, args.generator_revision)
    
    # Load Mask-Filler (if needed)
    if not args.baselines_only and not args.random_fills and not args.synonym_fills:
        int8_kwargs = {}
        half_kwargs = {}
        if args.int8:
            int8_kwargs = dict(load_in_8bit=True, device_map='auto', torch_dtype=torch.bfloat16)
        elif args.half:
            half_kwargs = dict(torch_dtype=torch.bfloat16)
        print(f'Loading mask-filler {mask_filler_name}...')
        mask_filler = transformers.AutoModelForSeq2SeqLM.from_pretrained(mask_filler_name, **int8_kwargs, **half_kwargs, cache_dir=cache_dir)
        try:
            n_positions = mask_filler.config.n_positions
        except AttributeError:
            n_positions = 512
    else:
        n_positions = 512
        
    preproc_tokenizer = transformers.AutoTokenizer.from_pretrained('t5-small', model_max_length=512, cache_dir=cache_dir)
    mask_filler_tokenizer = transformers.AutoTokenizer.from_pretrained(mask_filler_name, model_max_length=n_positions, cache_dir=cache_dir)
    
    if args.dataset in ['english', 'german']:
        preproc_tokenizer = mask_filler_tokenizer
        
    load_generator()
    print(f'Loading dataset {args.dataset}...')
    data = generate_data(args.dataset, args.dataset_key)
    
    # Build dictionary for Treatment B (Random real words)
    if args.random_fills:
        FILL_DICTIONARY = set()
        for texts in data.values():
            for text in texts:
                FILL_DICTIONARY.update(text.split())
        FILL_DICTIONARY = sorted(list(FILL_DICTIONARY))
        
    # Load Detector (Swap out generator to save VRAM)
    if args.detector_name:
        print(f'Loading DETECTOR {args.detector_name}...')
        del generator_model
        del generator_tokenizer
        torch.cuda.empty_cache()
        detector_model, detector_tokenizer = load_causal_model_and_tokenizer(args.detector_name, args.detector_revision)
        load_detector()
    else:
        detector_model, detector_tokenizer = generator_model, generator_tokenizer
        
    # Save generated data
    with open(os.path.join(SAVE_FOLDER, "raw_data.json"), "w") as f:
        print(f"Writing raw data to {os.path.join(SAVE_FOLDER, 'raw_data.json')}")
        json.dump(data, f)
        
    # Run Baselines
    if not args.skip_baselines:
        baseline_outputs = [run_baseline_threshold_experiment(get_ll, "likelihood", n_samples=n_samples)]
        baseline_outputs.append(run_baseline_threshold_experiment(lambda text: -get_rank(text, log=False), "rank", n_samples=n_samples))
        baseline_outputs.append(run_baseline_threshold_experiment(lambda text: -get_rank(text, log=True), "log_rank", n_samples=n_samples))
        baseline_outputs.append(run_baseline_threshold_experiment(get_entropy, "entropy", n_samples=n_samples))
        baseline_outputs.append(eval_supervised(data, model='roberta-base-openai-detector'))
        baseline_outputs.append(eval_supervised(data, model='roberta-large-openai-detector'))
        
    # Run Curvature Perturbation Experiments
    outputs = []
    if not args.baselines_only:
        for n_perturbations in n_perturbation_list:
            perturbation_results = get_perturbation_results(args.span_length, n_perturbations, n_samples)
            for perturbation_mode in ['d', 'z']:
                output = run_perturbation_experiment(perturbation_results, perturbation_mode, span_length=args.span_length, n_perturbations=n_perturbations, n_samples=n_samples)
                outputs.append(output)
                with open(os.path.join(SAVE_FOLDER, f"perturbation_{n_perturbations}_{perturbation_mode}_results.json"), "w") as f:
                    json.dump(output, f)
                    
    # Save Baseline Results
    if not args.skip_baselines:
        with open(os.path.join(SAVE_FOLDER, "likelihood_threshold_results.json"), "w") as f:
            json.dump(baseline_outputs[0], f)
        with open(os.path.join(SAVE_FOLDER, "rank_threshold_results.json"), "w") as f:
            json.dump(baseline_outputs[1], f)
        with open(os.path.join(SAVE_FOLDER, "logrank_threshold_results.json"), "w") as f:
            json.dump(baseline_outputs[2], f)
        with open(os.path.join(SAVE_FOLDER, "entropy_threshold_results.json"), "w") as f:
            json.dump(baseline_outputs[3], f)
        with open(os.path.join(SAVE_FOLDER, "roberta-base-openai-detector_results.json"), "w") as f:
            json.dump(baseline_outputs[-2], f)
        with open(os.path.join(SAVE_FOLDER, "roberta-large-openai-detector_results.json"), "w") as f:
            json.dump(baseline_outputs[-1], f)
        outputs += baseline_outputs
        
    # Generate Plots
    save_roc_curves(outputs)
    save_ll_histograms(outputs)
    save_llr_histograms(outputs)
    
    # Move from tmp_results to final results directory
    new_folder = SAVE_FOLDER.replace("tmp_results", "results")
    if not os.path.exists(os.path.dirname(new_folder)):
        os.makedirs(os.path.dirname(new_folder))
    os.rename(SAVE_FOLDER, new_folder)
    
    print("Execution complete.")