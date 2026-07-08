import torch
import yaml
from pathlib import Path
from cortex.modeling_cortex import CortexForCausalLM
from cortex.configuration_cortex import CortexConfig

def create_mock_checkpoint():
    with open("configs/cortex_micro.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    
    hf_config = CortexConfig(
        vocab_size=cfg["model"]["vocab_size"],
        d_model=cfg["model"]["d_model"],
        n_layers=cfg["model"]["n_layers"],
        n_heads=cfg["model"]["n_heads"],
        d_ff=cfg["model"]["d_ff"],
        max_seq_len=cfg["model"]["max_seq_len"],
        cortex_cfg=cfg
    )
    
    model = CortexForCausalLM(hf_config)
    
    cp = {
        "model": model.state_dict(),
        "config": cfg
    }
    
    out_dir = Path("results/cortex_micro")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    ckpt_path = out_dir / "model_final.pt"
    torch.save(cp, ckpt_path)
    print(f"Saved mock checkpoint to {ckpt_path}")

if __name__ == "__main__":
    create_mock_checkpoint()
