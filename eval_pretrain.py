#!/usr/bin/env python
import argparse
import sys
import time
import math
from pathlib import Path
import re
from typing import Optional, List, Dict, Tuple, Set

# Add parent dir to path so we can import rhapsody modules directly
sys.path.append(str(Path(__file__).parent.resolve()))

import torch
import torch.nn.functional as F

from rhapsody.inference import load_model
from rhapsody.data import get_tokenizer

# =============================================================================
# Evaluation Metrics & Helper Functions
# =============================================================================

def lcs(x: List[str], y: List[str]) -> int:
    """Computes the Longest Common Subsequence between two word lists."""
    m = len(x)
    n = len(y)
    L = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                L[i][j] = 0
            elif x[i-1] == y[j-1]:
                L[i][j] = L[i-1][j-1] + 1
            else:
                L[i][j] = max(L[i-1][j], L[i][j-1])
    return L[m][n]


def compute_rouge_l(reference: str, prediction: str) -> float:
    """Computes ROUGE-L F1 score using LCS on words."""
    ref_words = reference.lower().split()
    pred_words = prediction.lower().split()
    if not ref_words or not pred_words:
        return 0.0
    lcs_len = lcs(ref_words, pred_words)
    r_lcs = lcs_len / len(ref_words)
    p_lcs = lcs_len / len(pred_words)
    if r_lcs + p_lcs == 0:
        return 0.0
    beta = 1.0  # standard ROUGE-L uses beta = 1.0 (F1-score)
    f_lcs = ((1 + beta**2) * r_lcs * p_lcs) / (r_lcs + (beta**2) * p_lcs)
    return f_lcs


def compute_ngram_repetition(text: str, n: int = 3) -> float:
    """Computes n-gram repetition score on generated outputs."""
    words = text.strip().split()
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
    unique_ngrams = set(ngrams)
    return (len(ngrams) - len(unique_ngrams)) / len(ngrams)


@torch.no_grad()
def generate_text(model, tokenizer, prompt, max_new_tokens=128, temperature=0.0, top_p=0.9, repetition_penalty=1.15, device="cpu"):
    """
    Standard autoregressive generation helper.
    Uses KV caching and supports greedy (temp=0.0) or sampling mode.
    Applies repetition_penalty to logits of already generated tokens.
    """
    model.eval()
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    
    generated_tokens = []
    past_key_values = None
    
    # Pre-fill
    outputs = model(input_ids, use_cache=True)
    logits = outputs["logits"]
    past_key_values = outputs["past_key_values"]
    next_token_logits = logits[:, -1, :]
    
    # Sample first token (no repetition penalty needed yet)
    if temperature == 0.0:
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
    else:
        next_token_logits = next_token_logits / temperature
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            next_token_logits = next_token_logits.masked_fill(indices_to_remove, float("-inf"))
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
    generated_tokens.append(next_token.item())
    
    eos_tokens = {tokenizer.eos_token_id}
    abc_end_id = tokenizer.convert_tokens_to_ids("<|abc_end|>")
    if abc_end_id != tokenizer.unk_token_id:
        eos_tokens.add(abc_end_id)
        
    for _ in range(1, max_new_tokens):
        if next_token.item() in eos_tokens:
            break
            
        outputs = model(next_token, past_key_values=past_key_values, use_cache=True)
        logits = outputs["logits"]
        past_key_values = outputs["past_key_values"]
        next_token_logits = logits[:, -1, :]
        
        # Apply repetition penalty to already generated tokens
        if repetition_penalty != 1.0 and len(generated_tokens) > 0:
            next_token_logits = next_token_logits.clone()
            for token_id in set(generated_tokens):
                logit = next_token_logits[0, token_id]
                if logit > 0:
                    next_token_logits[0, token_id] = logit / repetition_penalty
                else:
                    next_token_logits[0, token_id] = logit * repetition_penalty
        
        if temperature == 0.0:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        else:
            next_token_logits = next_token_logits / temperature
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits = next_token_logits.masked_fill(indices_to_remove, float("-inf"))
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
        generated_tokens.append(next_token.item())
        
    return tokenizer.decode(generated_tokens, skip_special_tokens=True)


# =============================================================================
# Evaluation Tiers
# =============================================================================

def evaluate_perplexity(model, tokenizer, device, seq_len=1024, stride=512):
    """Tier 2: Compute sliding-window perplexity over WikiText-103 test split using Salesforce namespace."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("[Error] Hugging Face 'datasets' package is required. Install it using 'pip install datasets'.")
        return float('nan')
        
    print("\n" + "="*50)
    print("Tier 2: Computing WikiText-103 Test Perplexity")
    print("="*50)
    
    try:
        # Load from canonical Salesforce/wikitext repository
        dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")
    except Exception as e:
        print(f"[Warning] Failed to load WikiText-103 test split: {e}")
        print("Falling back to Salesforce/wikitext-2...")
        try:
            dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
        except Exception as e2:
            print(f"[Error] Failed to load fallback dataset WikiText-2: {e2}")
            return float('nan')
            
    print("[Eval] Concatenating test documents...")
    full_text = "\n\n".join(dataset["text"])
    
    print("[Eval] Tokenizing entire test corpus...")
    encodings = tokenizer(full_text, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)
    
    total_len = input_ids.size(1)
    print(f"[Eval] Total tokens in test split: {total_len}")
    
    model.eval()
    nlls = []
    prev_end_loc = 0
    total_active_tokens = 0
    total_nll = 0.0
    
    try:
        from tqdm import tqdm
        progress = tqdm(range(0, total_len, stride), desc="WikiText Perplexity")
    except ImportError:
        progress = range(0, total_len, stride)
        print("[Eval] Progress updates printed periodically...")
        
    step_count = 0
    total_steps = len(range(0, total_len, stride))
    
    for begin_loc in progress:
        end_loc = min(begin_loc + seq_len, total_len)
        cur_seq_len = end_loc - begin_loc
        if cur_seq_len <= 1:
            break
            
        seq = input_ids[:, begin_loc:end_loc]
        inputs = seq[:, :-1]
        labels = seq[:, 1:].clone()
        
        # Mask out target tokens that were predicted in previous steps
        mask_idx = prev_end_loc - begin_loc - 1
        if mask_idx > 0:
            labels[:, :mask_idx] = -100
            
        with torch.no_grad():
            outputs = model(inputs, labels=labels)
            loss = outputs["loss"]
            
        num_active = (cur_seq_len - 1) - max(0, mask_idx)
        total_nll += loss.item() * num_active
        total_active_tokens += num_active
        prev_end_loc = end_loc
        
        step_count += 1
        if isinstance(progress, range) and step_count % 100 == 0:
            percent = (step_count / total_steps) * 100
            current_ppl = math.exp(total_nll / total_active_tokens) if total_active_tokens > 0 else float('nan')
            print(f"  [{percent:.1f}%] Evaluated {prev_end_loc}/{total_len} tokens. Running PPL: {current_ppl:.2f}")
            
    ppl = math.exp(total_nll / total_active_tokens) if total_active_tokens > 0 else float('nan')
    print(f"\n[Eval] WikiText Perplexity: {ppl:.4f}")
    return ppl


def evaluate_sst2(model, tokenizer, device, max_samples=100):
    """Tier 4.1: 5-shot sentiment classification on SST-2 validation split."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("[Error] Hugging Face 'datasets' package is required.")
        return float('nan')
        
    print("\n" + "="*50)
    print("Tier 4.1: Few-shot Sentiment Classification (SST-2)")
    print("="*50)
    
    try:
        # Try nyu-mll/glue first, fall back to glue
        try:
            sst2 = load_dataset("nyu-mll/glue", "sst2", split="validation")
        except Exception:
            sst2 = load_dataset("glue", "sst2", split="validation")
    except Exception as e:
        print(f"[Warning] Failed to load SST-2 from HF GLUE: {e}")
        print("Using synthetic fallback samples...")
        sst2 = [
            {"sentence": "this movie is an absolute joy to watch .", "label": 1},
            {"sentence": "it is a waste of time and money .", "label": 0},
            {"sentence": "the acting was decent but the plot was lacking .", "label": 0},
            {"sentence": "hilarious and heartwarming from start to finish .", "label": 1},
            {"sentence": "i fell asleep halfway through the film .", "label": 0},
            {"sentence": "a visual masterpiece with a compelling story .", "label": 1},
            {"sentence": "the special effects could not save a terrible script .", "label": 0},
            {"sentence": "highly recommended for fans of the genre .", "label": 1},
            {"sentence": "it was just boring and predictable .", "label": 0},
            {"sentence": "wonderfully written and beautifully acted .", "label": 1}
        ]
        
    few_shot_examples = [
        {"sentence": "a stirring , funny and elevated boundary-stretcher .", "label": 1},
        {"sentence": "it 's slowly paced and not very exciting .", "label": 0},
        {"sentence": "a warm and inviting cinematic experience .", "label": 1},
        {"sentence": "the story is predictable and the acting is flat .", "label": 0},
        {"sentence": "an absolute masterpiece of modern filmmaking .", "label": 1}
    ]
    
    pos_words = ["Positive", " positive", "positive"]
    neg_words = ["Negative", " negative", "negative"]
    pos_ids = []
    neg_ids = []
    
    for word in pos_words:
        ids = tokenizer.encode(word, add_special_tokens=False)
        if ids:
            pos_ids.append(ids[0])
    for word in neg_words:
        ids = tokenizer.encode(word, add_special_tokens=False)
        if ids:
            neg_ids.append(ids[0])
            
    pos_ids = list(set(pos_ids))
    neg_ids = list(set(neg_ids))
    
    if not pos_ids or not neg_ids:
        print("[Warning] Could not dynamically find token IDs. Guessing based on common values.")
        pos_ids = [tokenizer.encode("Positive")[0]]
        neg_ids = [tokenizer.encode("Negative")[0]]
        
    correct = 0
    total = 0
    
    samples = sst2[:max_samples] if isinstance(sst2, list) else [sst2[i] for i in range(min(max_samples, len(sst2)))]
    
    model.eval()
    for sample in samples:
        test_sentence = sample["sentence"]
        test_label = sample["label"]
        
        prompt_parts = []
        for ex in few_shot_examples:
            label_str = "Positive" if ex["label"] == 1 else "Negative"
            prompt_parts.append(f"Sentence: {ex['sentence']}\nSentiment: {label_str}")
        prompt_parts.append(f"Sentence: {test_sentence}\nSentiment:")
        prompt = "\n\n".join(prompt_parts)
        
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
        
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs["logits"][0, -1, :]  # shape: [vocab_size]
            
            pos_score = sum(logits[pid].item() for pid in pos_ids)
            neg_score = sum(logits[nid].item() for nid in neg_ids)
            
            pred_label = 1 if pos_score > neg_score else 0
            
        if pred_label == test_label:
            correct += 1
        total += 1
        
    accuracy = correct / total if total > 0 else 0.0
    print(f"[Eval] SST-2 Few-shot Accuracy: {accuracy:.4f} ({correct}/{total})")
    return accuracy


def evaluate_ner(model, tokenizer, device, max_samples=20, repetition_penalty=1.15):
    """Tier 4.2: 3-shot Named Entity Recognition (NER) on CoNLL-2003 validation split."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("[Error] Hugging Face 'datasets' package is required.")
        return float('nan')
        
    print("\n" + "="*50)
    print("Tier 4.2: Few-shot Named Entity Recognition (NER)")
    print("="*50)
    
    try:
        # Try script-less lhoestq/conll2003 first, fall back to conll2003
        try:
            ner_dataset = load_dataset("lhoestq/conll2003", split="validation")
        except Exception:
            ner_dataset = load_dataset("conll2003", split="validation", trust_remote_code=True)
    except Exception as e:
        print(f"[Warning] Failed to load CoNLL-2003: {e}")
        print("Using synthetic fallback samples...")
        ner_dataset = [
            {
                "tokens": ["John", "Smith", "went", "to", "New", "York", "to", "work", "for", "Microsoft", "."],
                "ner_tags": [1, 2, 0, 0, 5, 6, 0, 0, 0, 3, 0]
            },
            {
                "tokens": ["Alice", "visited", "Paris", "in", "July", "."],
                "ner_tags": [1, 0, 5, 0, 0, 0]
            },
            {
                "tokens": ["The", "United", "Nations", "has", "headquarters", "in", "Geneva", "."],
                "ner_tags": [0, 3, 4, 0, 0, 0, 5, 0]
            },
            {
                "tokens": ["Steve", "Jobs", "was", "the", "CEO", "of", "Apple", "in", "California", "."],
                "ner_tags": [1, 2, 0, 0, 0, 0, 3, 0, 5, 0]
            },
            {
                "tokens": ["London", "is", "the", "capital", "of", "the", "United", "Kingdom", "."],
                "ner_tags": [5, 0, 0, 0, 0, 0, 5, 6, 0]
            }
        ]
        
    ner_few_shot_examples = [
        {
            "sentence": "U.N. official Nick Thorne visited Baghdad today.",
            "entities": "U.N. (ORG), Nick Thorne (PER), Baghdad (LOC)"
        },
        {
            "sentence": "The German government announced new trade agreements with China.",
            "entities": "German (MISC), China (LOC)"
        },
        {
            "sentence": "Larry Page and Sergey Brin founded Google in California.",
            "entities": "Larry Page (PER), Sergey Brin (PER), Google (ORG), California (LOC)"
        }
    ]
    
    def extract_entities_from_tags(tokens, tags):
        tag_names = ["O", "PER", "PER", "ORG", "ORG", "LOC", "LOC", "MISC", "MISC"]
        entities = []
        current_entity = []
        current_type = None
        
        for token, tag in zip(tokens, tags):
            if tag == 0:
                if current_entity:
                    entities.append(f"{' '.join(current_entity)} ({current_type})")
                    current_entity = []
                    current_type = None
            else:
                tag_type = tag_names[tag]
                is_b = (tag % 2 == 1)
                if is_b:
                    if current_entity:
                        entities.append(f"{' '.join(current_entity)} ({current_type})")
                    current_entity = [token]
                    current_type = tag_type
                else:
                    if current_entity and tag_type == current_type:
                        current_entity.append(token)
                    else:
                        if current_entity:
                            entities.append(f"{' '.join(current_entity)} ({current_type})")
                        current_entity = [token]
                        current_type = tag_type
        if current_entity:
            entities.append(f"{' '.join(current_entity)} ({current_type})")
            
        if not entities:
            return "None"
        return ", ".join(entities)
        
    def parse_entity_string(ent_str):
        if not ent_str or ent_str.strip().lower() == "none":
            return set()
        pattern = r"([^,()]+)\s*\(([^,()]+)\)"
        matches = re.findall(pattern, ent_str)
        parsed = set()
        for entity, etype in matches:
            t = etype.strip().upper()
            if t == "PERSON":
                t = "PER"
            elif t == "ORGANIZATION":
                t = "ORG"
            elif t == "LOCATION":
                t = "LOC"
            parsed.add((entity.strip().lower(), t))
        return parsed

    precisions = []
    recalls = []
    f1s = []
    
    samples = ner_dataset[:max_samples] if isinstance(ner_dataset, list) else [ner_dataset[i] for i in range(min(max_samples, len(ner_dataset)))]
    
    for i, sample in enumerate(samples):
        tokens = sample["tokens"]
        tags = sample["ner_tags"]
        test_sentence = " ".join(tokens)
        ground_truth_str = extract_entities_from_tags(tokens, tags)
        
        prompt_parts = []
        for ex in ner_few_shot_examples:
            prompt_parts.append(f"Sentence: {ex['sentence']}\nEntities: {ex['entities']}")
        prompt_parts.append(f"Sentence: {test_sentence}\nEntities:")
        prompt = "\n\n".join(prompt_parts)
        
        generated_output = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=50,
            temperature=0.0,
            repetition_penalty=repetition_penalty,
            device=device
        )
        
        true_ents = parse_entity_string(ground_truth_str)
        pred_ents = parse_entity_string(generated_output)
        
        intersection = true_ents.intersection(pred_ents)
        
        p = len(intersection) / len(pred_ents) if pred_ents else 0.0
        r = len(intersection) / len(true_ents) if true_ents else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)
        
    avg_p = sum(precisions) / len(precisions) if precisions else 0.0
    avg_r = sum(recalls) / len(recalls) if recalls else 0.0
    avg_f1 = sum(f1s) / len(f1s) if f1s else 0.0
    
    print(f"[Eval] NER Results - Precision: {avg_p:.4f}, Recall: {avg_r:.4f}, F1 Score: {avg_f1:.4f}")
    return avg_f1


def evaluate_summarization(model, tokenizer, device, max_samples=10, repetition_penalty=1.15):
    """Tier 4.3: 2-shot dialogue summarization evaluated with pure Python ROUGE-L."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("[Error] Hugging Face 'datasets' package is required.")
        return float('nan')
        
    print("\n" + "="*50)
    print("Tier 4.3: Few-shot Summarization (SAMSum / ROUGE-L)")
    print("="*50)
    
    try:
        # Try knkarthick/samsum first, fall back to samsum
        try:
            samsum = load_dataset("knkarthick/samsum", split="test")
        except Exception:
            samsum = load_dataset("samsum", split="test")
    except Exception as e:
        print(f"[Warning] Failed to load SAMSum dataset: {e}")
        print("Using synthetic fallback samples...")
        samsum = [
            {
                "dialogue": "Tom: Where are you?\nBob: Stuck in traffic.\nTom: When will you arrive?\nBob: Probably in 30 mins.",
                "summary": "Bob is stuck in traffic and will arrive at Tom's location in about 30 minutes."
            },
            {
                "dialogue": "Lisa: Did you feed the dog?\nMark: Yes, I fed him in the morning.\nLisa: Don't forget to walk him too.\nMark: OK, I will do it after lunch.",
                "summary": "Mark fed the dog in the morning and will walk him after lunch."
            },
            {
                "dialogue": "Ann: Let's order pizza.\nKen: Pepperoni?\nAnn: Sure, and some garlic bread.\nKen: Great, ordering now.",
                "summary": "Ann and Ken are ordering pepperoni pizza and garlic bread."
            }
        ]
        
    few_shot_examples = [
        {
            "dialogue": "Amanda: I'm thinking of buying a new phone.\nJess: Which one?\nAmanda: The latest iPhone. But it's so expensive.\nJess: Maybe wait for a discount or buy a refurbished one.",
            "summary": "Amanda wants to buy a new iPhone but finds it expensive. Jess suggests waiting for a discount or getting a refurbished model."
        },
        {
            "dialogue": "John: Are we meeting today?\nSarah: Yes, at 5 PM in the library.\nJohn: Got it. See you there.",
            "summary": "John and Sarah are meeting at the library at 5 PM today."
        }
    ]
    
    rouge_scores = []
    
    samples = samsum[:max_samples] if isinstance(samsum, list) else [samsum[i] for i in range(min(max_samples, len(samsum)))]
    
    for sample in samples:
        test_dialogue = sample["dialogue"]
        ground_truth = sample["summary"]
        
        prompt_parts = []
        for ex in few_shot_examples:
            prompt_parts.append(f"Dialogue:\n{ex['dialogue']}\n\nSummary: {ex['summary']}")
        prompt_parts.append(f"Dialogue:\n{test_dialogue}\n\nSummary:")
        prompt = "\n\n".join(prompt_parts)
        
        generated_summary = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=100,
            temperature=0.0,
            repetition_penalty=repetition_penalty,
            device=device
        )
        
        score = compute_rouge_l(ground_truth, generated_summary)
        rouge_scores.append(score)
        
    avg_rouge_l = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0.0
    print(f"[Eval] Summarization ROUGE-L Score: {avg_rouge_l:.4f}")
    return avg_rouge_l


def run_qualitative_prompts(model, tokenizer, device, repetition_penalty=1.15):
    """Tier 1 (Qualitative checks) & Tier 3 (N-gram repetition computation)"""
    print("\n" + "="*50)
    print("Tier 1 & Tier 3: Qualitative Prompts & Repetition Scores")
    print("="*50)
    
    prompts = [
        "In the year 2045, humanity made its greatest discovery:",
        "The capital of France is",
        "If a jacket costs $100 after a 20% discount, its original price was",
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    else:",
        "List of 3 primary colors:\n1.",
        "Classify the sentiment of the following sentences as Positive or Negative.\nSentence: I loved this movie!\nSentiment: Positive\nSentence: The service was terrible.\nSentiment: Negative\nSentence: The weather is okay.\nSentiment: Neutral\nSentence: I had a great time today.\nSentiment:",
        "Once upon a time in a galaxy far away, there was a small robot who",
        "Question: Who wrote the play Romeo and Juliet?\nAnswer:"
    ]
    
    qualitative_results = []
    repetition_scores_3g = []
    repetition_scores_4g = []
    
    for i, prompt in enumerate(prompts):
        print(f"\n--- Prompt {i+1}: {repr(prompt)} ---")
        completion = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=128,
            temperature=0.3,
            repetition_penalty=repetition_penalty,
            device=device
        )
        rep_3g = compute_ngram_repetition(completion, n=3)
        rep_4g = compute_ngram_repetition(completion, n=4)
        
        repetition_scores_3g.append(rep_3g)
        repetition_scores_4g.append(rep_4g)
        
        print(f"Generated text:\n{completion.strip()}")
        print(f"Repetition Score (3-gram): {rep_3g:.4f}")
        print(f"Repetition Score (4-gram): {rep_4g:.4f}")
        
        qualitative_results.append({
            "prompt": prompt,
            "completion": completion.strip(),
            "rep_3g": rep_3g,
            "rep_4g": rep_4g
        })
        
    avg_rep_3g = sum(repetition_scores_3g) / len(repetition_scores_3g)
    avg_rep_4g = sum(repetition_scores_4g) / len(repetition_scores_4g)
    
    print(f"\n[Eval] Average 3-gram Repetition Score: {avg_rep_3g:.4f}")
    print(f"[Eval] Average 4-gram Repetition Score: {avg_rep_4g:.4f}")
    
    return qualitative_results, avg_rep_3g, avg_rep_4g


# =============================================================================
# Main Program
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate Rhapsody Pretrained TextLM.")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the model checkpoint.pt file")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run evaluation on (cuda or cpu). T4 GPU is highly recommended.")
    parser.add_argument("--seq-len", type=int, default=1024,
                        help="Context window length for sliding perplexity (default: 1024)")
    parser.add_argument("--stride", type=int, default=512,
                        help="Stride length for sliding perplexity (default: 512)")
    parser.add_argument("--max-sst2", type=int, default=100,
                        help="Max validation samples for SST-2 evaluation (default: 100)")
    parser.add_argument("--max-ner", type=int, default=20,
                        help="Max validation samples for NER evaluation (default: 20)")
    parser.add_argument("--max-sum", type=int, default=10,
                        help="Max validation samples for Summarization evaluation (default: 10)")
    parser.add_argument("--skip-perplexity", action="store_true",
                        help="Skip WikiText-103 perplexity evaluation (saves time)")
    parser.add_argument("--repetition-penalty", type=float, default=1.15,
                        help="Repetition penalty for autoregressive text generation (default: 1.15)")
    args = parser.parse_args()
    
    print("="*60)
    print(" RHAPSODY PRETRAINED MODEL EVALUATION SUITE ")
    print("="*60)
    print(f"Checkpoint:          {args.checkpoint}")
    print(f"Device:              {args.device}")
    print(f"Repetition Penalty:  {args.repetition_penalty}")
    
    if args.device == "cpu":
        print("\n[WARNING] You are running on CPU.")
        print("  Evaluating WikiText perplexity requires thousands of forward passes.")
        print("  This can take a VERY long time on CPU (possibly hours).")
        print("  We strongly recommend running this on a T4 GPU (or other GPU) runtime in Google Colab.")
        print("  If you must run on CPU, consider using --skip-perplexity to only run the downstream tasks.\n")
        
    # Load model
    print("[Rhapsody] Loading model from checkpoint...")
    model = load_model(args.checkpoint, device=args.device)
    
    # Load tokenizer
    print("[Rhapsody] Loading tokenizer...")
    tokenizer = get_tokenizer(symbolic=False)
    
    results = {}
    
    # Tier 1 & 3: Qualitative & Repetitions
    qualitative_results, avg_rep_3g, avg_rep_4g = run_qualitative_prompts(model, tokenizer, args.device, repetition_penalty=args.repetition_penalty)
    results["avg_rep_3g"] = avg_rep_3g
    results["avg_rep_4g"] = avg_rep_4g
    
    # Tier 2: Perplexity
    if not args.skip_perplexity:
        ppl = evaluate_perplexity(model, tokenizer, args.device, seq_len=args.seq_len, stride=args.stride)
        results["wikitext_ppl"] = ppl
    else:
        results["wikitext_ppl"] = float('nan')
        print("\n[Eval] WikiText-103 Perplexity evaluation skipped.")
        
    # Tier 4.1: SST-2 Sentiment
    sst2_acc = evaluate_sst2(model, tokenizer, args.device, max_samples=args.max_sst2)
    results["sst2_accuracy"] = sst2_acc
    
    # Tier 4.2: NER F1
    ner_f1 = evaluate_ner(model, tokenizer, args.device, max_samples=args.max_ner, repetition_penalty=args.repetition_penalty)
    results["ner_f1"] = ner_f1
    
    # Tier 4.3: Summarization ROUGE-L
    sum_rouge_l = evaluate_summarization(model, tokenizer, args.device, max_samples=args.max_sum, repetition_penalty=args.repetition_penalty)
    results["summarization_rouge_l"] = sum_rouge_l
    
    # Final Summary Report
    print("\n" + "="*60)
    print(" FINAL EVALUATION SUMMARY REPORT ")
    print("="*60)
    print(f"Model Checkpoint:              {args.checkpoint}")
    print(f"Evaluation Device:             {args.device}")
    print(f"WikiText-103 Perplexity:       {results['wikitext_ppl']:.4f}" if not math.isnan(results['wikitext_ppl']) else "WikiText-103 Perplexity:       SKIPPED")
    print(f"Average 3-gram Repetition:     {results['avg_rep_3g']:.4f} (Target: < 0.15)")
    print(f"Average 4-gram Repetition:     {results['avg_rep_4g']:.4f} (Target: < 0.10)")
    print(f"SST-2 Few-shot Accuracy:       {results['sst2_accuracy']:.4f}")
    print(f"CoNLL-2003 NER Few-shot F1:    {results['ner_f1']:.4f}")
    print(f"SAMSum Summarization ROUGE-L:  {results['summarization_rouge_l']:.4f}")
    print("="*60)
    print("Evaluation Complete. ✅")

if __name__ == "__main__":
    main()
