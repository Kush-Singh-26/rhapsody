import torch
import torch.nn as nn
from rhapsody.model import create_text_only_65m, create_rhapsody_65m
from rhapsody.train import Muon, get_wsd_lr, compute_loss

def run_synthetic_training():
    print("=" * 60)
    print("Running 2-Step CPU Synthetic Training Verification...")
    print("=" * 60)

    device = torch.device("cpu")
    vocab_size = 1000
    seq_len = 32
    batch_size = 2
    grad_accum = 2

    # 1. Initialize text model
    print("[Test] Initializing model...")
    model = create_text_only_65m(vocab_size=vocab_size)
    model.to(device)
    model.train()

    # 2. Setup Muon optimizer (same as train.py)
    muon_params = [
        p for n, p in model.named_parameters()
        if p.ndim >= 2 and "embed" not in n and "lm_head" not in n and p.requires_grad
    ]
    adamw_params = [
        p for n, p in model.named_parameters()
        if (p.ndim < 2 or "embed" in n or "lm_head" in n) and p.requires_grad
    ]

    optimizer = Muon(
        muon_params,
        lr=0.015, momentum=0.95, nesterov=True, ns_steps=5,
        adamw_params=adamw_params, adamw_lr=0.0008,
        adamw_betas=(0.9, 0.95), adamw_wd=0.1,
    )

    # 3. Setup Scheduler
    max_steps = 4
    total_opt_steps = max_steps // grad_accum
    wsd = lambda step: get_wsd_lr(step, total_opt_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=[wsd] * len(optimizer.param_groups),
    )

    # 4. Generate synthetic batch data
    print("[Test] Generating synthetic batches...")
    batches = []
    for _ in range(max_steps):
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        batches.append({"input_ids": input_ids, "labels": labels})

    # 5. Run steps
    step = 0
    running_loss = 0.0
    for batch in batches:
        # Forward pass & loss computation
        loss = compute_loss(model, batch, device) / grad_accum
        loss.backward()
        running_loss += loss.item() * grad_accum

        # Optimizer step
        if (step + 1) % grad_accum == 0:
            print(f"[Test] Step {step + 1}: performing optimizer update...")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            
            avg_loss = running_loss / grad_accum
            muon_lr = scheduler.get_last_lr()[0]
            print(f"  -> Loss: {avg_loss:.4f} | Muon LR: {muon_lr:.2e}")
            running_loss = 0.0

        step += 1

    print("\n[Test] Synthetic training run completed successfully! ✅")

if __name__ == "__main__":
    run_synthetic_training()
