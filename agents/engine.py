"""CORTEX Agent Engine.

Wraps the CORTEX language model to provide an easy interface for agents,
including system prompts, generation with stop words, and memory management.
"""

import torch
import tiktoken
import yaml
from pathlib import Path
from cortex.modeling_cortex import CortexForCausalLM
from cortex.configuration_cortex import CortexConfig

class CortexAgentEngine:
    def __init__(self, config_path="configs/cortex_language.yaml", checkpoint_path="results/cortex_language/checkpoint.pt", device="cpu"):
        self.device = device
        self.enc = tiktoken.get_encoding("cl100k_base")
        
        # Load Config
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
            mcfg = cfg['model']
            
        cortex_cfg = {
            "controller": {
                "d_controller": mcfg.get("d_controller", mcfg["d_model"]),
            },
            "patch": {
                "patch_hidden": mcfg.get("patch_hidden", 32),
            },
            "objective": {}
        }
        
        hf_config = CortexConfig(
            vocab_size=mcfg["vocab_size"],
            d_model=mcfg["d_model"],
            n_layers=mcfg["n_layers"],
            n_heads=mcfg["n_heads"],
            d_ff=mcfg["d_ff"],
            max_seq_len=mcfg["max_seq_len"],
            cortex_cfg=cortex_cfg
        )
        
        print(f"Loading CORTEX agent engine on {device}...")
        self.model = CortexForCausalLM(hf_config).to(device)
        self.model.eval()
        
        if Path(checkpoint_path).exists():
            print(f"Loaded weights from {checkpoint_path}")
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        else:
            print("WARNING: Checkpoint not found! Agent will use random weights.")

    def generate(self, prompt: str, max_tokens: int = 100, stop_sequences: list = None, temperature: float = 0.7) -> str:
        """Generates text until a stop sequence is hit or max_tokens is reached."""
        input_ids = self.enc.encode_ordinary(prompt)
        x = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        
        generated_tokens = []
        
        with torch.no_grad():
            for _ in range(max_tokens):
                # We do not pass difficulty to use base behavior
                logits = self.model(x)
                next_token_logits = logits[0, -1, :]
                
                if temperature == 0:
                    next_token = torch.argmax(next_token_logits).item()
                else:
                    probs = torch.nn.functional.softmax(next_token_logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1).item()
                    
                generated_tokens.append(next_token)
                x = torch.cat((x, torch.tensor([[next_token]], device=self.device)), dim=1)
                
                # Check stop sequences
                if stop_sequences:
                    current_text = self.enc.decode(generated_tokens)
                    for stop_seq in stop_sequences:
                        if current_text.endswith(stop_seq):
                            # Remove the stop sequence from the output
                            return current_text[:-len(stop_seq)]
                            
        return self.enc.decode(generated_tokens)
