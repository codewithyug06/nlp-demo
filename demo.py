import torch
import tiktoken
import time
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
import os

from cortex.configuration_cortex import CortexConfig
from cortex.modeling_cortex import CortexForCausalLM
from utils import load_config

app = FastAPI()

# Load Model
cfg = load_config("configs/cortex_language.yaml")
config = CortexConfig(cortex_cfg=cfg, **cfg["model"])
config.quantize_kv = True 
config.sliding_window = 128

device = "cpu"
model = CortexForCausalLM(config).to(device)
model.eval()
enc = tiktoken.get_encoding("cl100k_base")

class ChatRequest(BaseModel):
    prompt: str

@app.get("/")
def get_ui():
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.post("/api/chat")
def chat(request: ChatRequest):
    def generate():
        tokens = enc.encode_ordinary(request.prompt)
        if len(tokens) == 0:
            yield ""
            return

        x = torch.tensor([tokens], dtype=torch.long, device=device)
        past_key_values = None
        input_ids = x

        generated = 0
        max_tokens = 50
        temperature = 0.8

        while generated < max_tokens:
            with torch.no_grad():
                logits, past_key_values = model(input_ids, use_cache=True, past_key_values=past_key_values)
                next_token_logits = logits[0, -1, :]
                
                if temperature > 0:
                    probs = torch.softmax(next_token_logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                    
                input_ids = next_token.unsqueeze(0)
                
                t_val = next_token.item()
                if t_val >= 100256:
                    word = "[OOB]"
                else:
                    try:
                        word = enc.decode([t_val])
                    except Exception:
                        word = "[ERR]"
                        
                yield word
                generated += 1
                time.sleep(0.05)

    return StreamingResponse(generate(), media_type="text/plain")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
