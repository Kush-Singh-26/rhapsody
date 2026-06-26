import argparse
import os
import math
import random
import re
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

try:
    from datasets import load_dataset
except ImportError:
    print("[Error] Please install 'datasets' package (pip install datasets)")
    import sys
    sys.exit(1)

from rhapsody.inference import load_model
from rhapsody.data import get_tokenizer

STOP_WORDS = set([
    "the", "and", "a", "an", "of", "to", "in", "is", "it", "that", "on", "with",
    "for", "as", "are", "was", "this", "but", "by", "from", "at", "or", "which",
    "they", "you", "we", "he", "she", "his", "her", "their", "my", "your", "its"
])

def extract_topic(haiku: str) -> str:
    """Extract a topic from a haiku by picking the longest non-stop word."""
    words = re.findall(r'\b[a-zA-Z]+\b', haiku.lower())
    words = [w for w in words if w not in STOP_WORDS]
    if not words:
        return "nature"
    return max(words, key=len)

class HaikuDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        haiku = item["haiku"].replace(" / ", "\n")
        topic = extract_topic(haiku)
        
        prompt = f"Write a haiku about {topic}.\n"
        # Append eos_token to the end of full_text so the model learns when to stop
        full_text = prompt + haiku + self.tokenizer.eos_token
        
        encoded = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        input_ids = encoded["input_ids"].squeeze(0)
        
        # Create labels: -100 for the prompt so we only train on generating the haiku
        prompt_encoded = self.tokenizer(
            prompt,
            add_special_tokens=False,
            return_tensors="pt"
        )
        prompt_len = prompt_encoded["input_ids"].shape[1]
        
        # Get length of full text without padding to know where actual text ends
        non_padded_encoded = self.tokenizer(
            full_text,
            add_special_tokens=False,
            return_tensors="pt"
        )
        non_padded_len = non_padded_encoded["input_ids"].shape[1]
        
        # Ensure we don't index out of bounds if prompt/full text is longer than max_length
        prompt_len = min(prompt_len, self.max_length)
        non_padded_len = min(non_padded_len, self.max_length)
        
        labels = torch.full_like(input_ids, -100)
        # Shifted labels: input_ids[t] predicts input_ids[t+1]
        # We start predicting from prompt_len - 1 (predicting the first token of the haiku)
        # We stop predicting at non_padded_len - 1 (predicting the EOS token)
        if prompt_len - 1 < non_padded_len - 1:
            labels[prompt_len - 1 : non_padded_len - 1] = input_ids[prompt_len : non_padded_len]
            
        return {
            "input_ids": input_ids,
            "labels": labels
        }

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Rhapsody to be a Constraint Poet (Haiku).")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to pre-trained model.pt")
    parser.add_argument("--output-dir", type=str, default="outputs_poet", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--max-samples", type=int, default=50000, help="Max samples to train on")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    print("============================================================")
    print(" RHAPSODY CONSTRAINT POET FINE-TUNING")
    print("============================================================")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"[1] Loading dataset taucris/haiku_333K...")
    try:
        ds = load_dataset("taucris/haiku_333K", split="train")
        if args.max_samples > 0 and args.max_samples < len(ds):
            ds = ds.select(range(args.max_samples))
        print(f"Loaded {len(ds)} samples.")
    except Exception as e:
        print(f"[Error] Could not load dataset: {e}")
        return
        
    print(f"[2] Loading tokenizer...")
    tokenizer = get_tokenizer(symbolic=False)
    
    train_dataset = HaikuDataset(ds, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    print(f"[3] Loading pre-trained model from {args.checkpoint}...")
    model = load_model(args.checkpoint, device=args.device)
    model.train()
    
    print(f"[4] Setting up optimizer (AdamW)...")
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    def save_checkpoint(model, args, suffix=""):
        out_path = os.path.join(args.output_dir, f"poet_model{suffix}.safetensors")
        print(f"[\u2714] Saving model to {out_path}...")
        try:
            from safetensors.torch import save_file
            tensors = {k: v.cpu().contiguous() for k, v in model.state_dict().items()}
            save_file(tensors, out_path)
        except ImportError:
            print("[Warning] safetensors not installed, saving as .pt instead.")
            out_path = os.path.join(args.output_dir, f"poet_model{suffix}.pt")
            save_dict = {
                "model": model.state_dict(),
                "config": model.config.__dict__ if hasattr(model, 'config') else {}
            }
            torch.save(save_dict, out_path)
    
    print(f"[5] Starting training on {args.device}...")
    global_step = 0
    total_steps = len(train_loader) * args.epochs
    
    try:
        for epoch in range(args.epochs):
            epoch_loss = 0.0
            for batch_idx, batch in enumerate(train_loader):
                input_ids = batch["input_ids"].to(args.device)
                labels = batch["labels"].to(args.device)
                
                optimizer.zero_grad()
                outputs = model(input_ids, labels=labels)
                loss = outputs["loss"]
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                global_step += 1
                
                if global_step % 50 == 0:
                    print(f"Epoch [{epoch+1}/{args.epochs}] Step [{global_step}/{total_steps}] Loss: {loss.item():.4f}")
                    
            avg_loss = epoch_loss / len(train_loader)
            print(f"==> Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
            # Save checkpoint at end of epoch
            save_checkpoint(model, args, suffix=f"_epoch_{epoch+1}")
            
    except KeyboardInterrupt:
        print("\n[!] Training interrupted by user!")
        save_checkpoint(model, args, suffix="_interrupted")
        print("Safely exited and saved interrupted checkpoint.")
        return
        
    print(f"[6] Saving final fine-tuned model...")
    save_checkpoint(model, args)
    print("Done!")

if __name__ == "__main__":
    main()
