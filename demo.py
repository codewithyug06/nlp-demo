import torch
import gradio as gr
import tiktoken
import time
from cortex.configuration_cortex import CortexConfig
from cortex.modeling_cortex import CortexForCausalLM
from utils import load_config

# Load standard config
cfg = load_config("configs/cortex_language.yaml")
config = CortexConfig(cortex_cfg=cfg, **cfg["model"])

# Init model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CortexForCausalLM(config).to(device)
model.eval()

enc = tiktoken.get_encoding("cl100k_base")

def get_color(g: float) -> str:
    """Map controller signal g (0.0 to 1.0) to color from Blue (Easy) to Red (Hard)."""
    # High g = Hard = Red, Low g = Easy = Blue
    g = max(0.0, min(1.0, g))
    r = int(255 * g)
    b = int(255 * (1 - g))
    g_col = 50
    return f"rgb({r},{g_col},{b})"

def generate_thoughts(prompt, max_tokens, temperature):
    tokens = enc.encode_ordinary(prompt)
    if len(tokens) == 0:
        yield "<p>Please enter a prompt.</p>"
        return

    x = torch.tensor([tokens], dtype=torch.long, device=device)
    html_output = "<div>"

    # Add prompt in standard color
    for t in tokens:
        word = enc.decode([t])
        html_output += f"<span>{word}</span>"

    yield html_output + "</div>"
    
    for _ in range(max_tokens):
        with torch.no_grad():
            B, T = x.shape
            
            # Forward pass
            logits = model(x)
            
            # Extract controller signal for the LAST token
            if model.last_signal is not None and model.last_signal.g is not None:
                g_val = model.last_signal.g[0, -1].item()
            else:
                g_val = 0.5
                
            next_token_logits = logits[0, -1, :]
            
            if temperature > 0:
                probs = torch.softmax(next_token_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
            x = torch.cat([x, next_token.unsqueeze(0)], dim=1)
            
            # Decode the next token
            word = enc.decode([next_token.item()])
            word = word.replace("<", "&lt;").replace(">", "&gt;") # Escape HTML
            
            color = get_color(g_val)
            
            # Highlight with g_val tooltip
            span = f'<span style="background-color: {color}; color: white; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Controller g_t: {g_val:.3f}">{word}</span>'
            html_output += span
            
            yield html_output + "</div>"
            time.sleep(0.05) # small delay for visual effect

with gr.Blocks(title="CORTEX: Adaptive Compute Transformer") as demo:
    gr.Markdown("# 🧠 CORTEX Live Demo")
    gr.Markdown("Watch the **Universal Compute Controller** in real-time. Words highlighted in **Red** indicate high compute allocation (Hard tokens), while **Blue** indicates early exit/low compute (Easy tokens). Hover over a word to see its exact $g_t$ signal.")
    
    with gr.Row():
        with gr.Column(scale=1):
            prompt = gr.Textbox(lines=5, label="Input Prompt", value="The Adaptive Compute architecture works by")
            max_tokens = gr.Slider(minimum=10, maximum=200, value=50, step=1, label="Max Tokens")
            temp = gr.Slider(minimum=0.0, maximum=1.5, value=0.8, step=0.1, label="Temperature")
            submit = gr.Button("Generate")
            
        with gr.Column(scale=2):
            output = gr.HTML(label="CORTEX Thought Visualizer")
            
    submit.click(generate_thoughts, inputs=[prompt, max_tokens, temp], outputs=output)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
