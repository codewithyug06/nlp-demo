import os
import json
import argparse
import shutil
import torch
from safetensors.torch import save_file
from huggingface_hub import HfApi

from cortex.configuration_cortex import CortexConfig
from cortex.modeling_cortex import CortexForCausalLM

def main():
    parser = argparse.ArgumentParser(description="Export CORTEX to Hugging Face format.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the trained PyTorch checkpoint (.pt).")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for HF export.")
    parser.add_argument("--push_to_hub", type=str, default=None, help="Repository ID to push to (e.g. username/cortex-micro).")
    
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    
    print(f"Loading checkpoint {args.checkpoint}...")
    cp = torch.load(args.checkpoint, map_location="cpu")
    
    
    
    
    
    
    if "config" in cp:
        raw_cfg = cp["config"]
    else:
        raise ValueError("Checkpoint does not contain 'config'. Cannot export automatically without the config dictionary.")
        
    model_cfg = raw_cfg.get("model", {})
    hf_config = CortexConfig(
        vocab_size=model_cfg.get("vocab_size", 100257),
        d_model=model_cfg.get("d_model", 384),
        n_layers=model_cfg.get("n_layers", 6),
        n_heads=model_cfg.get("n_heads", 6),
        d_ff=model_cfg.get("d_ff", 1536),
        max_seq_len=model_cfg.get("max_seq_len", 512),
        dropout=model_cfg.get("dropout", 0.0),
        norm=model_cfg.get("norm", "pre"),
        tie_embeddings=model_cfg.get("tie_embeddings", True),
        mtp_depth=raw_cfg.get("objective", {}).get("mtp_depth", 4),
        cortex_cfg=raw_cfg,
    )
    
    
    
    hf_config.auto_map = {
        "AutoConfig": "configuration_cortex.CortexConfig",
        "AutoModelForCausalLM": "modeling_cortex.CortexForCausalLM"
    }

    
    hf_config.save_pretrained(args.out_dir)
    print(f"Saved config to {args.out_dir}/config.json")
    
    
    
    model = CortexForCausalLM(hf_config)
    model.load_state_dict(cp["model"])
    
    
    sd = model.state_dict()
    if hf_config.tie_embeddings and "lm_head.weight" in sd:
        sd.pop("lm_head.weight")
        
    save_file(sd, os.path.join(args.out_dir, "model.safetensors"), metadata={"format": "pt"})
    print(f"Saved model weights to {args.out_dir}/model.safetensors")

    
    src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cortex")
    
    for f in os.listdir(src_dir):
        if f.endswith(".py") and f not in ["model.py", "decode.py"]:
            src_path = os.path.join(src_dir, f)
            shutil.copy(src_path, os.path.join(args.out_dir, f))
            
    print("Copied architecture files.")
    
    
    if args.push_to_hub:
        print(f"Pushing to Hugging Face Hub: {args.push_to_hub}")
        api = HfApi()
        try:
            api.create_repo(repo_id=args.push_to_hub, exist_ok=True)
            api.upload_folder(
                folder_path=args.out_dir,
                repo_id=args.push_to_hub,
                repo_type="model"
            )
            print("Successfully pushed to Hub!")
        except Exception as e:
            print(f"Failed to push to Hub: {e}")
            print("Ensure you have logged in via `huggingface-cli login`")

if __name__ == "__main__":
    main()
