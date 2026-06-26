import os
import sys
import subprocess
import time
import torch
import torch.nn.functional as F
from pathlib import Path

def run_cmd(cmd):
    try:
        res = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        return res.stdout.strip()
    except Exception as e:
        return f"Error running {cmd}: {e}"

print("============================================================")
print(" COLAB DIAGNOSTIC REPORT ")
print("============================================================")

# 1. Git Checks
print("\n[1] Repository Status:")
print("Current Commit Hash:", run_cmd("git rev-parse HEAD"))
print("Git Status:")
print(run_cmd("git status -s"))
print("Git Diff for finetune_poet.py:")
print(run_cmd("git diff finetune_poet.py"))

# 2. File Mod times
print("\n[2] File Timestamps:")
files_to_check = [
    "finetune_poet.py",
    "eval_poet.py",
    "rhapsody/inference.py",
    "outputs_poet/poet_model.safetensors",
]
for f in files_to_check:
    path = Path(f)
    if path.exists():
        mtime = time.ctime(path.stat().st_mtime)
        size = path.stat().st_size
        print(f"  {f:38s} | Size: {size:10,d} bytes | Modified: {mtime}")
    else:
        print(f"  {f:38s} | NOT FOUND")

# 3. Model Loading & Prediction Trace
model_path = "outputs_poet/poet_model.safetensors"
if not Path(model_path).exists():
    print(f"\n[Error] Model file {model_path} not found. Cannot proceed with prediction check.")
    sys.exit(0)

print(f"\n[3] Testing Model Inference Trace from {model_path}...")
try:
    from rhapsody.inference import load_model
    from rhapsody.data import get_tokenizer
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on {device}...")
    model = load_model(model_path, device=device)
    model.eval()
    
    tokenizer = get_tokenizer(symbolic=False)
    prompt = "Write a haiku about thoughts.\n"
    print(f"Prompt: {repr(prompt)}")
    
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    print(f"Prompt input_ids: {input_ids.tolist()}")
    
    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)
    logits = outputs["logits"]
    next_token_logits = logits[0, -1, :]
    
    if torch.isnan(next_token_logits).any():
        print("WARNING: Logits contain NaNs!")
        
    probs = F.softmax(next_token_logits, dim=-1)
    top_probs, top_indices = torch.topk(probs, 10)
    
    print("\nTop 10 predicted tokens for the very first step:")
    for i in range(10):
        idx = top_indices[i].item()
        prob = top_probs[i].item()
        token = tokenizer.decode([idx])
        print(f"  ID: {idx:5d} | Token: {repr(token):15s} | Probability: {prob:.4%}")
        
    # Manual greedy autoregressive simulation (10 steps)
    print("\nSimulating greedy generation (10 steps):")
    past_key_values = outputs["past_key_values"]
    curr_token = torch.argmax(next_token_logits, dim=-1, keepdim=True).unsqueeze(0) # shape [1, 1]
    generated = [curr_token.item()]
    print(f"  Step 1: ID {curr_token.item():5d} -> {repr(tokenizer.decode([curr_token.item()]))}")
    
    for step in range(2, 11):
        with torch.no_grad():
            outputs = model(curr_token, past_key_values=past_key_values, use_cache=True)
        past_key_values = outputs["past_key_values"]
        next_token_logits = outputs["logits"][0, -1, :]
        curr_token = torch.argmax(next_token_logits, dim=-1, keepdim=True).unsqueeze(0)
        token_id = curr_token.item()
        generated.append(token_id)
        print(f"  Step {step}: ID {token_id:5d} -> {repr(tokenizer.decode([token_id]))}")
        
    print(f"\nDecoded sequence: {repr(tokenizer.decode(generated))}")

except Exception as e:
    import traceback
    print(f"\n[Error] Exception during model run check: {e}")
    traceback.print_exc()

print("\n============================================================")
