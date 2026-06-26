import argparse
import torch
import torch.nn.functional as F
from rhapsody.inference import load_model
from rhapsody.data import get_tokenizer

@torch.no_grad()
def generate_text(model, tokenizer, prompt, max_new_tokens=128, temperature=0.7, repetition_penalty=1.15, device="cpu"):
    model.eval()
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    
    generated_tokens = []
    past_key_values = None
    
    # Pre-fill
    outputs = model(input_ids, use_cache=True)
    logits = outputs["logits"]
    past_key_values = outputs["past_key_values"]
    next_token_logits = logits[:, -1, :]
    
    if temperature == 0.0:
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
    else:
        next_token_logits = next_token_logits / temperature
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
        
        # Apply repetition penalty
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
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
        generated_tokens.append(next_token.item())
        
    return tokenizer.decode(generated_tokens, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="Test the Constraint Poet (Haiku).")
    parser.add_argument("--checkpoint", type=str, default="outputs_poet/poet_model.safetensors", help="Path to fine-tuned model")
    parser.add_argument("--topic", type=str, required=True, help="Topic for the haiku")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--temp", type=float, default=0.7, help="Temperature for generation")
    
    args = parser.parse_args()
    
    print(f"Loading Poet model from {args.checkpoint} on {args.device}...")
    model = load_model(args.checkpoint, device=args.device)
    tokenizer = get_tokenizer(symbolic=False)
    
    prompt = f"Write a haiku about {args.topic}.\n"
    print(f"\n--- Generating Haiku (Topic: {args.topic}) ---\n")
    
    completion = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=40,
        temperature=args.temp,
        repetition_penalty=1.15,
        device=args.device
    )
    
    print(completion.strip())
    print("\n" + "-"*40)

if __name__ == "__main__":
    main()
