import torch
from rhapsody.model import create_text_only_65m

def test_kv_cache():
    print("Testing KV cache correctness...")
    model = create_text_only_65m(vocab_size=32000)
    model.eval()

    batch_size = 2
    prompt_len = 10
    
    # 1. Generate a prompt
    prompt_ids = torch.randint(0, 32000, (batch_size, prompt_len))
    
    # 2. Run standard forward pass on prompt
    with torch.no_grad():
        out_no_cache = model(prompt_ids)
        logits_no_cache = out_no_cache["logits"]  # [B, prompt_len, vocab]

    # 3. Run forward pass with use_cache=True
    with torch.no_grad():
        out_cache = model(prompt_ids, use_cache=True)
        logits_cache = out_cache["logits"]  # [B, prompt_len, vocab]
        past_key_values = out_cache["past_key_values"]

    # Assert logits match at pre-fill stage
    assert torch.allclose(logits_no_cache, logits_cache, atol=1e-5), "Pre-fill logits do not match!"
    print("Pre-fill logits matched successfully.")

    # 4. Generate a new token
    next_token_ids = torch.randint(0, 32000, (batch_size, 1))
    full_sequence = torch.cat([prompt_ids, next_token_ids], dim=-1)

    # 5. Full forward pass without cache on the extended sequence
    with torch.no_grad():
        out_full = model(full_sequence)
        logits_full_last = out_full["logits"][:, -1, :]  # [B, vocab] (logits for the next token)

    # 6. Incremental forward pass with cache
    with torch.no_grad():
        out_incremental = model(next_token_ids, past_key_values=past_key_values, use_cache=True)
        logits_incremental = out_incremental["logits"][:, -1, :]  # [B, vocab]
        new_past_key_values = out_incremental["past_key_values"]

    # Assert logits match for the incremental token
    diff = torch.max(torch.abs(logits_full_last - logits_incremental)).item()
    print(f"Max logit diff for next token: {diff}")
    assert torch.allclose(logits_full_last, logits_incremental, atol=1e-4), f"Logits do not match on incremental step! Diff: {diff}"
    print("Incremental step logits matched successfully.")
    
    # Check cache shape
    # Number of layers should match the small TextLM config.
    assert len(new_past_key_values) == 20, "Expected 20 layers in KV cache"
    # Each layer should have a tuple of (K, V)
    # Shape of K and V should be [batch, kv_heads, seq_len, head_dim]
    # kv_heads = 4, seq_len = prompt_len + 1 = 11, head_dim = 512 / 8 = 64
    k, v = new_past_key_values[0]
    expected_shape = (batch_size, 4, prompt_len + 1, 64)
    assert k.shape == expected_shape, f"K shape mismatch: {k.shape} != {expected_shape}"
    assert v.shape == expected_shape, f"V shape mismatch: {v.shape} != {expected_shape}"
    print("Cache shapes are correct.")
    print("All KV cache correctness tests passed! ✅")

if __name__ == "__main__":
    test_kv_cache()
