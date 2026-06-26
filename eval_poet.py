import argparse
import re
import torch
from datasets import load_dataset
from rhapsody.inference import load_model, generate_text
from rhapsody.data import get_tokenizer

try:
    import syllables
except ImportError:
    print("[Error] Please install the syllables package for evaluation: pip install syllables")
    import sys
    sys.exit(1)

from finetune_poet import extract_topic

def count_syllables_in_line(line: str) -> int:
    """Counts syllables in a text line by summing syllables of individual words."""
    words = re.findall(r'\b[a-zA-Z]+\b', line.lower())
    return sum(syllables.estimate(w) for w in words)

def main():
    parser = argparse.ArgumentParser(description="Evaluate the Constraint Poet.")
    parser.add_argument("--checkpoint", type=str, default="outputs_poet/poet_model.safetensors")
    parser.add_argument("--samples", type=int, default=50, help="Number of test samples to evaluate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    print("============================================================")
    print(" POETRY EVALUATION SUITE: HAIKU METRICS ")
    print("============================================================")
    
    print(f"Loading Poet model from {args.checkpoint} on {args.device}...")
    model = load_model(args.checkpoint, device=args.device)
    tokenizer = get_tokenizer(symbolic=False)
    
    print(f"Loading test dataset...")
    # Load from the test/val split if available, otherwise grab the end of the train split
    ds = load_dataset("taucris/haiku_333K", split="train[-1000:]") 
    
    samples_to_test = min(args.samples, len(ds))
    
    exact_structure_matches = 0
    three_line_matches = 0
    total_syllable_error = 0
    
    for i in range(samples_to_test):
        item = ds[i]
        haiku_raw = item["haiku"].replace(" / ", "\n")
        topic = extract_topic(haiku_raw)
        
        prompt = f"Write a haiku about {topic}.\n"
        
        completion = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=40,
            temperature=0.7,
            repetition_penalty=1.15,
            no_repeat_ngram_size=0,
            device=args.device
        )
        
        # generate_text only returns the newly generated tokens, so no slicing is needed
        generated = completion.strip()
        lines = [line.strip() for line in generated.split('\n') if line.strip()]
        
        if len(lines) == 3:
            three_line_matches += 1
            
            syl_counts = [count_syllables_in_line(line) for line in lines]
            
            # Target is 5 - 7 - 5
            error = abs(syl_counts[0] - 5) + abs(syl_counts[1] - 7) + abs(syl_counts[2] - 5)
            total_syllable_error += error
            
            if syl_counts == [5, 7, 5]:
                exact_structure_matches += 1
                
        if i < 3: # Print first 3 for qualitative check
            print(f"\n[Sample {i+1}] Topic: {topic}")
            print(generated)
            if len(lines) == 3:
                print(f"Syllables: {syl_counts} (Error: {error})")
            else:
                print(f"Syllables: Failed (Generated {len(lines)} lines)")
                
    
    print("\n============================================================")
    print(" QUANTITATIVE RESULTS ")
    print("============================================================")
    print(f"Total Samples Evaluated:    {samples_to_test}")
    print(f"3-Line Structure Accuracy:  {(three_line_matches/samples_to_test)*100:.1f}%")
    
    if three_line_matches > 0:
        avg_error = total_syllable_error / three_line_matches
        print(f"Exact 5-7-5 Accuracy:       {(exact_structure_matches/samples_to_test)*100:.1f}%")
        print(f"Avg Syllable Error:         {avg_error:.2f} syllables off per poem")
    
    print("============================================================")

if __name__ == "__main__":
    main()
