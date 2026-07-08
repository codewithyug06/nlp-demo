import torch
from cortex.modeling_cortex import CortexForCausalLM
from utils import load_config, get_device
import tiktoken

def colorize(text: str, intensity: float) -> str:
    """Colorize text from blue (0.0) to red (1.0)."""
    
    r = int(intensity * 255)
    b = int((1 - intensity) * 255)
    return f"\033[38;2;{r};0;{b}m{text}\033[0m"

def main():
    device = get_device()
    print(f"Using device: {device}")
    
    
    cfg = load_config("configs/cortex_language.yaml")
    
    
    
    
    from cortex.configuration_cortex import CortexConfig
    config = CortexConfig(cortex_cfg=cfg, **cfg["model"])
    model = CortexForCausalLM(config).to(device)
    model.eval()
    
    enc = tiktoken.get_encoding("cl100k_base")
    
    
    text = "Once upon a time, in a far away land, there lived a small brave knight. He loved to explore the dark caves."
    
    tokens = enc.encode(text)
    x = torch.tensor([tokens], dtype=torch.long, device=device)
    
    print("\n--- CORTEX Controller Analysis ---\n")
    print(f"Input text: '{text}'\n")
    
    with torch.no_grad():
        with torch.autocast(device_type="cuda" if "cuda" in str(device) else "cpu", dtype=torch.bfloat16):
            logits = model(x)
            
    
    if model.controller is None or model.last_signal is None:
        print("Controller is disabled in this configuration.")
        return
        
    g_t = model.last_signal.g[0].cpu().float().tolist()  
    
    print("Token-level Difficulty (g_t):")
    print("Blue = Easy (g_t ~ 0), Red = Hard (g_t ~ 1)\n")
    
    colored_text = ""
    for token, g in zip(tokens, g_t):
        token_str = enc.decode([token])
        colored_text += colorize(token_str, g)
        
    print(colored_text)
    print("\n------------------------------------\n")

if __name__ == "__main__":
    main()
