#!/usr/bin/env python
import argparse
import sys
import time
from pathlib import Path
from typing import Optional, List, Tuple

import torch
import torch.nn.functional as F

from rhapsody.data import get_tokenizer
from rhapsody.model import LmConfig, TextLM


def load_model(checkpoint_path: str, device: str = "cpu") -> torch.nn.Module:
    """Load model from a checkpoint (safetensors or .pt file)."""
    config_dict = {}
    
    if str(checkpoint_path).endswith('.safetensors'):
        from safetensors.torch import load_file
        state_dict = load_file(checkpoint_path, device=device)
    else:
        # Use weights_only=False to ensure compatibility with custom checkpoint metadata (like config)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        config_dict = checkpoint.get("config", {})
        state_dict = checkpoint.get("model", checkpoint)  # Fallback if saved directly
    
    if not config_dict:
        print("[Rhapsody] No config found in checkpoint. Creating model with default settings.")
        from rhapsody.model import create_text_only_65m
        model = create_text_only_65m()
    else:
        config = LmConfig.from_config_dict(config_dict)
        print(f"[Rhapsody] Config loaded: {config}")
        model = TextLM(config)
            
    # Load weights
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"[Rhapsody] Loaded Text-only model on {device}.")
    return model


@torch.no_grad()
def generate_text(model, tokenizer, prompt, max_new_tokens=128, temperature=0.0, top_p=0.9, repetition_penalty=1.15, no_repeat_ngram_size=3, device="cpu"):
    """
    Standard autoregressive generation helper.
    Uses KV caching and supports greedy (temp=0.0) or sampling mode.
    Applies repetition_penalty and no_repeat_ngram_size to logits.
    """
    model.eval()
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    
    # If the tokenizer automatically appended an EOS token, strip it off so we can continue generation
    if input_ids.shape[1] > 1 and input_ids[0, -1] == tokenizer.eos_token_id:
        input_ids = input_ids[:, :-1]
        
    generated_tokens = []
    past_key_values = None
    
    # Pre-fill
    outputs = model(input_ids, use_cache=True)
    logits = outputs["logits"]
    past_key_values = outputs["past_key_values"]
    next_token_logits = logits[:, -1, :]
    
    # Apply no_repeat_ngram_size to first token (using prompt context)
    all_tokens = input_ids[0].tolist()
    if no_repeat_ngram_size > 0 and len(all_tokens) >= no_repeat_ngram_size:
        last_tokens = all_tokens[-(no_repeat_ngram_size - 1):]
        banned_tokens = []
        for i in range(len(all_tokens) - no_repeat_ngram_size + 1):
            window = all_tokens[i : i + no_repeat_ngram_size - 1]
            if window == last_tokens:
                banned_token = all_tokens[i + no_repeat_ngram_size - 1]
                banned_tokens.append(banned_token)
        if banned_tokens:
            temp_logits = next_token_logits.clone()
            temp_logits[0, banned_tokens] = float("-inf")
            if not torch.all(torch.isinf(temp_logits)):
                next_token_logits = temp_logits
                
    # Sample first token
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
                    
        # Apply no_repeat_ngram_size
        all_tokens = input_ids[0].tolist() + generated_tokens
        if no_repeat_ngram_size > 0 and len(all_tokens) >= no_repeat_ngram_size:
            last_tokens = all_tokens[-(no_repeat_ngram_size - 1):]
            banned_tokens = []
            for i in range(len(all_tokens) - no_repeat_ngram_size + 1):
                window = all_tokens[i : i + no_repeat_ngram_size - 1]
                if window == last_tokens:
                    banned_token = all_tokens[i + no_repeat_ngram_size - 1]
                    banned_tokens.append(banned_token)
            if banned_tokens:
                temp_logits = next_token_logits.clone()
                temp_logits[0, banned_tokens] = float("-inf")
                if not torch.all(torch.isinf(temp_logits)):
                    next_token_logits = temp_logits
        
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


def main():
    parser = argparse.ArgumentParser(description="Autoregressive generation with KV Cache for TextLM.")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint.")
    parser.add_argument("--prompt", type=str, default="",
                        help="Text prompt prefix.")
    parser.add_argument("--max-new-tokens", type=int, default=128,
                        help="Maximum number of tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature. Use 0.0 for greedy decoding.")
    parser.add_argument("--top-p", type=float, default=0.9,
                        help="Nucleus sampling threshold.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run inference on.")
    args = parser.parse_args()
    
    tokenizer = get_tokenizer(symbolic=False)
    model = load_model(args.checkpoint, device=args.device)
    
    t_start = time.time()
    result = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        device=args.device,
    )
    t_end = time.time()
    
    print("\n[Generated Text]:")
    print(result)
    print("-" * 50)
    print(f"Speed: {args.max_new_tokens / (t_end - t_start):.1f} tokens/second")
    print("-" * 50)


if __name__ == "__main__":
    main()
