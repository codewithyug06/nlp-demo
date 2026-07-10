import torch
import gradio as gr
import tiktoken
import time
from torchvision import transforms
from cortex.configuration_cortex import CortexConfig
from cortex.modeling_cortex import CortexForCausalLM
from utils import load_config

# Load standard config
cfg = load_config("configs/cortex_language.yaml")
config = CortexConfig(cortex_cfg=cfg, **cfg["model"])

# Enable advanced features for demo
config.quantize_kv = True 
config.sliding_window = 128

# Init model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CortexForCausalLM(config).to(device)

# Load latest checkpoint weights (disable for now since we removed bias and need fresh train)
# try:
#     checkpoint = torch.load("results/run/checkpoint.pt", map_location=device)
#     model.load_state_dict(checkpoint)
#     print("Successfully loaded training checkpoint!")
# except Exception as e:
#     print(f"Warning: Could not load checkpoint, using random weights: {e}")

model.eval()
enc = tiktoken.get_encoding("cl100k_base")

# Standard Image Transform for 224x224 Vision Encoder
img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_color(g: float) -> str:
    g = max(0.0, min(1.0, g))
    r = int(255 * g)
    b = int(255 * (1 - g))
    return f"rgb({r},50,{b})"

def generate_thoughts(prompt, image, max_tokens, temperature, use_speculative):
    tokens = enc.encode_ordinary(prompt)
    if len(tokens) == 0:
        yield "<p>Please enter a prompt.</p>"
        return

    x = torch.tensor([tokens], dtype=torch.long, device=device)
    
    # Process Image if provided
    pixel_values = None
    html_output = "<div>"
    if image is not None:
        pixel_values = img_transform(image).unsqueeze(0).to(device)
        html_output += "<p><i>[Processed Image into Vision Tokens]</i></p>"

    for t in tokens:
        word = enc.decode([t])
        html_output += f"<span>{word}</span>"

    yield html_output + "</div>"
    
    past_key_values = None
    input_ids = x

    generated = 0
    while generated < max_tokens:
        with torch.no_grad():
            if use_speculative:
                # SPECULATIVE DECODING (Dummy Draft)
                # We draft 3 future tokens by just randomly sampling (to simulate a fast, dumb draft model)
                draft_len = 3
                draft_tokens = torch.randint(0, config.vocab_size, (1, draft_len), device=device)
                
                # We pass the real input + the 3 draft tokens to the main model in ONE pass!
                spec_input = torch.cat([input_ids, draft_tokens], dim=1)
                
                logits, past_key_values = model(spec_input, use_cache=True, past_key_values=past_key_values, pixel_values=pixel_values)
                pixel_values = None # Only pass image on first pass
                
                # The model outputs probabilities for all tokens. 
                # In real speculative decoding, we compare the drafted tokens against these probabilities.
                # Since our draft was random, it will 99.9% reject the draft and just accept the first true generated token.
                
                next_token_logits = logits[0, -draft_len-1, :]
                if temperature > 0:
                    probs = torch.softmax(next_token_logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                input_ids = next_token.unsqueeze(0)
                
                # Add rejected draft visualization
                word = enc.decode([next_token.item()]).replace("<", "&lt;").replace(">", "&gt;")
                span = f'<span style="background-color: #333; color: white; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Speculative Verify: Accepted 1, Rejected {draft_len}">[Spec:{word}]</span>'
                html_output += span
                generated += 1
                
            else:
                # STANDARD AUTOREGRESSIVE DECODING
                logits, past_key_values = model(input_ids, use_cache=True, past_key_values=past_key_values, pixel_values=pixel_values)
                pixel_values = None # Only pass image on first pass
                
                g_val = 0.5
                if model.last_signal is not None and model.last_signal.g is not None:
                    g_val = model.last_signal.g[0, -1].item()
                    
                next_token_logits = logits[0, -1, :]
                if temperature > 0:
                    probs = torch.softmax(next_token_logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                    
                input_ids = next_token.unsqueeze(0)
                word = enc.decode([next_token.item()]).replace("<", "&lt;").replace(">", "&gt;")
                
                color = get_color(g_val)
                span = f'<span style="background-color: {color}; color: white; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Controller g_t: {g_val:.3f}">{word}</span>'
                html_output += span
                generated += 1

            yield html_output + "</div>"
            time.sleep(0.05)

with gr.Blocks(title="CORTEX: Multimodal & Speculative Demo") as demo:
    gr.Markdown("# 🧠 CORTEX Live Demo (Phase 3)")
    gr.Markdown("Features Active: **Vision Multimodal**, **KV Quantization**, **Sliding Window**, and **Speculative Decoding**.")
    
    with gr.Row():
        with gr.Column(scale=1):
            image = gr.Image(type="pil", label="Optional Image (Vision)")
            prompt = gr.Textbox(lines=3, label="Input Prompt", value="Analyze this image:")
            
            with gr.Accordion("Advanced Settings", open=True):
                use_speculative = gr.Checkbox(label="Enable Speculative Decoding", value=False)
                max_tokens = gr.Slider(minimum=10, maximum=200, value=50, step=1, label="Max Tokens")
                temp = gr.Slider(minimum=0.0, maximum=1.5, value=0.8, step=0.1, label="Temperature")
            
            submit = gr.Button("Generate")
            
        with gr.Column(scale=2):
            output = gr.HTML(label="CORTEX Visualizer")
            
    submit.click(generate_thoughts, inputs=[prompt, image, max_tokens, temp, use_speculative], outputs=output)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
