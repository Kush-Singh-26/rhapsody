#!/usr/bin/env python
import argparse
import sys
import time
from pathlib import Path
from typing import Optional, List, Tuple

import torch
import torch.nn.functional as F
import torchaudio

from rhapsody.data import get_tokenizer
from rhapsody.model import RhapsodyConfig, RhapsodyModel, TextLM


def load_audio(path: str, target_sr: int = 48000) -> torch.Tensor:
    """Load an audio file and resample to target sampling rate (48kHz for CLAP)."""
    waveform, sr = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)
    return waveform.squeeze(0)


def load_model(checkpoint_path: str, device: str = "cpu") -> torch.nn.Module:
    """Load model from a checkpoint, auto-detecting TextLM vs RhapsodyModel."""
    # Use weights_only=False to ensure compatibility with custom checkpoint metadata (like python_random_state)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    config_dict = checkpoint.get("config", {})
    state_dict = checkpoint.get("model", checkpoint)  # Fallback if saved directly
    
    # Check if checkpoint is multimodal by looking for audio-related parameters
    is_multimodal = any("audio_encoder" in k or "projector" in k for k in state_dict.keys())
    
    if not config_dict:
        print("[Rhapsody] No config found in checkpoint. Creating model with default settings.")
        if is_multimodal:
            from rhapsody.model import create_rhapsody_65m
            model = create_rhapsody_65m()
        else:
            from rhapsody.model import create_text_only_65m
            model = create_text_only_65m()
    else:
        config = RhapsodyConfig.from_config_dict(config_dict)
        print(f"[Rhapsody] Config loaded: {config}")
        if is_multimodal:
            model = RhapsodyModel(config)
        else:
            model = TextLM(config)
            
    # Load weights
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"[Rhapsody] Loaded {'Multimodal' if is_multimodal else 'Text-only'} model on {device}.")
    return model


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    tokenizer,
    audio_path: Optional[str] = None,
    prompt: str = "",
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
    device: str = "cpu",
    task: str = "audio-captioning",
):
    """
    Perform autoregressive text generation using KV cache.
    Supports both RhapsodyModel (with audio prefix) and TextLM (text-only).

    WARNING: The multimodal KV-cache decoding path is experimental and has not
    been fully validated under heavy workloads or complex batching profiles.
    """
    is_multimodal = isinstance(model, RhapsodyModel)
    is_symbolic = task == "symbolic-music"
    
    # ── 1. Process Audio Prefix ──────────────────────────────────────────────
    audio_features = None
    if audio_path is not None:
        if not is_multimodal:
            print("[Warning] Audio path provided, but the model is text-only. Ignoring audio.")
        else:
            print(f"[Rhapsody] Loading audio: {audio_path} ...")
            waveform = load_audio(audio_path, target_sr=48000)
            
            from transformers import ClapProcessor
            processor = ClapProcessor.from_pretrained(model.config.audio_encoder_type)
            
            print("[Rhapsody] Running CLAP encoder...")
            feats = processor(
                audios=waveform.numpy(),
                sampling_rate=48000,
                return_tensors="pt",
            )
            audio_features = feats.input_features.to(device)
            
    # ── 2. Tokenize Text Prompt ──────────────────────────────────────────────
    if is_symbolic:
        if prompt:
            input_text = f"<|music|> {prompt}\n<|abc_start|>"
        else:
            input_text = "<|music|> <|abc_start|>"
    else:
        if not prompt and is_multimodal:
            prompt = "<|text|>"
        input_text = prompt
    
    input_ids = tokenizer(input_text, return_tensors="pt")["input_ids"].to(device)
    
    print(f"[Rhapsody] Prompt: {repr(input_text)}")
    print("[Rhapsody] Starting autoregressive generation with KV Cache...")
    
    # ── 3. Generation Loop ───────────────────────────────────────────────────
    generated_tokens = []
    past_key_values = None
    
    t_start = time.time()
    
    if is_multimodal and audio_features is not None:
        outputs = model(
            input_ids=input_ids,
            audio_features=audio_features,
            use_cache=True,
        )
    else:
        outputs = model(
            input_ids=input_ids,
            use_cache=True,
        )
        
    logits = outputs["logits"]
    past_key_values = outputs["past_key_values"]
    next_token_logits = logits[:, -1, :]
    
    next_token = sample_next_token(next_token_logits, temperature, top_p)
    generated_tokens.append(next_token.item())
    
    print("\n[Generated]: ", end="", flush=True)
    last_text = ""
    
    first_char = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    print(first_char, end="", flush=True)
    last_text = first_char
    
    eos_tokens = {tokenizer.eos_token_id}
    abc_end_id = tokenizer.convert_tokens_to_ids("<|abc_end|>")
    # Only add abc_end if it was actually registered (non-symbolic tokenizer returns unk)
    if abc_end_id != tokenizer.unk_token_id:
        eos_tokens.add(abc_end_id)
    
    for step in range(1, max_new_tokens):
        if next_token.item() in eos_tokens:
            break
            
        outputs = model(
            input_ids=next_token,
            past_key_values=past_key_values,
            use_cache=True,
        )
        logits = outputs["logits"]
        past_key_values = outputs["past_key_values"]
        
        next_token_logits = logits[:, -1, :]
        next_token = sample_next_token(next_token_logits, temperature, top_p)
        generated_tokens.append(next_token.item())
        
        full_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        new_text = full_text[len(last_text):]
        print(new_text, end="", flush=True)
        last_text = full_text
        
    t_end = time.time()
    print("\n")
    
    total_tokens = len(generated_tokens)
    duration = t_end - t_start
    tok_per_sec = total_tokens / duration if duration > 0 else 0.0
    
    print("-" * 50)
    print(f"Generation statistics:")
    print(f"  Generated tokens:    {total_tokens}")
    print(f"  Time taken:          {duration:.2f} seconds")
    print(f"  Speed:               {tok_per_sec:.1f} tokens/second")
    print("-" * 50)
    
    return last_text


def sample_next_token(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    """Apply temperature scaling and top-p (nucleus) filtering to logits, and sample."""
    if temperature == 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)
        
    # Scale logits
    logits = logits / temperature
    
    # Apply top-p filtering
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above top_p (nucleus)
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift indices to keep the first token above threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        # Scatter indices to remove back to original logits tensor
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, float("-inf"))
        
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def main():
    parser = argparse.ArgumentParser(description="Autoregressive generation with KV Cache for Rhapsody.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint. If not provided, initialize a random model.")
    parser.add_argument("--text-only", action="store_true",
                        help="Force TextLM even when initializing a random model.")
    parser.add_argument("--task", type=str, default="audio-captioning",
                        choices=["audio-captioning", "symbolic-music"],
                        help="Task type for generation.")
    parser.add_argument("--audio", type=str, default=None,
                        help="Path to input audio file.")
    parser.add_argument("--prompt", type=str, default="",
                        help="Optional text prompt prefix.")
    parser.add_argument("--max-new-tokens", type=int, default=128,
                        help="Maximum number of tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature. Use 0.0 for greedy decoding.")
    parser.add_argument("--top-p", type=float, default=0.9,
                        help="Nucleus sampling threshold.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run inference on (cpu or cuda).")
    args = parser.parse_args()
    
    is_symbolic = args.task == "symbolic-music"
    tokenizer = get_tokenizer(symbolic=is_symbolic)
    
    if args.checkpoint is not None:
        model = load_model(args.checkpoint, device=args.device)
    else:
        print("[Rhapsody] No checkpoint provided. Creating a randomly initialized model for testing...")
        if args.text_only or (args.audio is None and not args.prompt):
            from rhapsody.model import create_text_only_65m
            model = create_text_only_65m(vocab_size=len(tokenizer))  # len() includes added special tokens
            print("[Rhapsody] Initialized random TextLM (~65M at 32K vocab).")
        else:
            from rhapsody.model import create_rhapsody_65m
            model = create_rhapsody_65m(vocab_size=len(tokenizer))  # len() includes added special tokens
            print("[Rhapsody] Initialized random RhapsodyModel.")
        model.to(args.device)
        model.eval()
        
    result = generate(
        model=model,
        tokenizer=tokenizer,
        audio_path=args.audio,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        device=args.device,
        task=args.task,
    )

    if is_symbolic:
        from rhapsody.abc_utils import extract_abc_from_generated, validate_abc
        abc = extract_abc_from_generated(result)
        if abc:
            print("\n" + "=" * 50)
            print("Extracted ABC notation:")
            print("=" * 50)
            print(abc)
            if validate_abc(abc):
                print("\n[OK] ABC syntax appears valid.")
            else:
                print("\n[WARN] ABC syntax may be incomplete or invalid.")


if __name__ == "__main__":
    main()
