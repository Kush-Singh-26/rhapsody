import argparse
import torch
import torch.nn.functional as F
from rhapsody.inference import load_model, generate_text
from rhapsody.data import get_tokenizer

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
