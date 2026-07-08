import sys
from transformers import AutoModelForCausalLM

print("Loading exported model with trust_remote_code=True...")
model = AutoModelForCausalLM.from_pretrained("./hf_export", trust_remote_code=True)
print("SUCCESS: Loaded model!")
print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

if id(model.lm_head.weight) == id(model.tok_emb.weight):
    print("SUCCESS: lm_head and tok_emb weights are correctly tied.")
else:
    print("WARNING: lm_head and tok_emb weights are NOT tied.")
