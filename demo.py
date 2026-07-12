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

SYSTEM_PROMPT = """Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

Tradeoff: These guidelines bias toward caution over speed. For trivial tasks, use judgment.

1. Think Before Coding
Don't assume. Don't hide confusion. Surface tradeoffs.
Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

2. Simplicity First
Minimum code that solves the problem. Nothing speculative.
- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.
Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

3. Surgical Changes
Touch only what you must. Clean up only your own mess.
When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.
When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.
The test: Every changed line should trace directly to the user's request.

4. Goal-Driven Execution
Define success criteria. Loop until verified.
Transform tasks into verifiable goals:
- "Add validation" -> "Write tests for invalid inputs, then make them pass"
- "Fix the bug" -> "Write a test that reproduces it, then make it pass"
- "Refactor X" -> "Ensure tests pass before and after"
For multi-step tasks, state a brief plan:
1. [Step] -> verify: [check]
Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.
"""


@app.get("/")
def get_ui():
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.post("/api/chat")
def chat(request: ChatRequest):
    def generate():
        formatted_prompt = f"System: {SYSTEM_PROMPT}\n\nUser: {request.prompt}\n\nAssistant:"
        tokens = enc.encode_ordinary(formatted_prompt)
        if len(tokens) == 0:
            yield ""
            return

        x = torch.tensor([tokens], dtype=torch.long, device=device)
        past_key_values = None
        input_ids = x

        generated = 0
        max_tokens = 512
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
