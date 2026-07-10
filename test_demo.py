import torch
from cortex.configuration_cortex import CortexConfig
from cortex.modeling_cortex import CortexForCausalLM
from utils import load_config
import tiktoken

cfg = load_config("configs/cortex_language.yaml")
config = CortexConfig(cortex_cfg=cfg, **cfg["model"])
config.quantize_kv = True
config.sliding_window = 128

model = CortexForCausalLM(config).cuda()
# model.load_state_dict(torch.load("results/run/checkpoint.pt"))
model.eval()

enc = tiktoken.get_encoding("cl100k_base")
prompt = "Once upon a time, there was a little boy who"
x = torch.tensor([enc.encode_ordinary(prompt)], dtype=torch.long).cuda()

print(prompt, end="")
past_key_values = None
input_ids = x

for _ in range(20):
    with torch.no_grad():
        logits, past_key_values = model(input_ids, use_cache=True, past_key_values=past_key_values)
        next_token = torch.argmax(logits[0, -1, :], dim=-1, keepdim=True)
        input_ids = next_token.unsqueeze(0)
        print(enc.decode([next_token.item()]), end="")
print()
