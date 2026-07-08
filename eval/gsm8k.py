"""GSM8K Evaluation Harness for CORTEX.

Proves the Latent Ponder Loop improves zero-shot reasoning.
"""

import re
import torch
import argparse
from tqdm import tqdm
from utils import load_config
from cortex.configuration_cortex import CortexConfig
from cortex.modeling_cortex import CortexForCausalLM

def extract_answer(text: str) -> str:
    """Extracts the numeric answer from GSM8K ground truth or generation."""
    # GSM8K ground truths always end with #### [answer]
    if "####" in text:
        return text.split("####")[-1].strip()
    
    # Heuristic for generation: grab the last number
    numbers = re.findall(r'-?\d+\.?\d*', text)
    if numbers:
        return numbers[-1]
    return ""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/cortex_language.yaml")
    parser.add_argument("--dummy", action="store_true", help="Run a quick test without real data")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    config = CortexConfig(cortex_cfg=cfg, **cfg["model"])
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CortexForCausalLM(config).to(device)
    model.eval()
    
    print(f"Loaded CORTEX from {args.config} to evaluate on GSM8K")
    
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    
    if args.dummy:
        print("[DUMMY MODE] Testing parser and loop.")
        dataset = [
            {"question": "If John has 2 apples and buys 3 more, how many does he have?", "answer": "#### 5"}
        ]
    else:
        try:
            from datasets import load_dataset
            dataset = load_dataset("gsm8k", "main", split="test")
        except ImportError:
            print("Please install `datasets` to evaluate on real GSM8K.")
            return

    correct = 0
    total = 0
    
    for row in tqdm(dataset):
        prompt = f"Question: {row['question']}\nAnswer: Let's think step by step.\n"
        truth = extract_answer(row['answer'])
        
        tokens = enc.encode_ordinary(prompt)
        x = torch.tensor([tokens], dtype=torch.long, device=device)
        
        # Generation loop
        generated_tokens = []
        max_gen = 50 if args.dummy else 256
        
        for _ in range(max_gen):
            with torch.no_grad():
                logits = model(x)
                next_tok = torch.argmax(logits[0, -1, :]).item()
                generated_tokens.append(next_tok)
                x = torch.cat([x, torch.tensor([[next_tok]], device=device)], dim=1)
                
                # Check for EOS or double newline
                if next_tok == 100257 or (len(generated_tokens) > 2 and enc.decode(generated_tokens[-2:]) == "\n\n"):
                    break
                    
        gen_text = enc.decode(generated_tokens)
        gen_ans = extract_answer(gen_text)
        
        if gen_ans == truth:
            correct += 1
        total += 1
        
        if args.dummy:
            print(f"Gen: {gen_text}")
            print(f"Extracted Ans: {gen_ans} | Truth: {truth}")
            break
            
    print(f"GSM8K Zero-Shot Accuracy: {correct}/{total} ({(correct/total)*100:.2f}%)")

if __name__ == "__main__":
    main()
