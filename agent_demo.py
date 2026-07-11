import gradio as gr
import time
from agents.engine import CortexAgentEngine
from agents.tools import execute_tool
from agents.single_agent import SYSTEM_PROMPT

# Load the engine globally
print("Initializing CORTEX Agent Engine for Web UI...")
engine = CortexAgentEngine(device="cpu")

def chat_stream(user_message, history):
    """Generator function to stream the agent's responses and tool executions."""
    
    # 1. Format the full chat history into the prompt
    prompt = SYSTEM_PROMPT
    for msg_idx in range(0, len(history), 2):
        if msg_idx + 1 < len(history):
            user_text = history[msg_idx]["content"]
            assistant_text = history[msg_idx + 1]["content"]
            prompt += f"{user_text}\nCORTEX: {assistant_text}\nUser: "
        
    prompt += f"{user_message}\nCORTEX: "
    
    # We will build the response HTML block by block
    current_response = ""
    
    # We allow the agent a maximum of 5 reasoning steps (tool calls)
    for step in range(5):
        # Notify the UI that the agent is thinking
        yield current_response + "\n\n*🤔 Thinking...*"
        
        # Generate text until it tries to call a tool or finishes
        output = engine.generate(prompt, max_tokens=100, stop_sequences=["</tool_call>"])
        
        prompt += output
        current_response += output
        yield current_response
        
        if "<tool_call>" in output:
            # The agent decided to use a tool!
            tool_json = output.split("<tool_call>")[-1].strip()
            
            # Show a cute loading badge for the tool execution
            current_response += "</tool_call>\n\n"
            current_response += f"> **🛠️ Executing Tool:**\n> ```json\n> {tool_json}\n> ```\n\n"
            yield current_response
            
            time.sleep(1) # Fake delay for visual effect
            
            # Execute the tool safely
            result = execute_tool(tool_json)
            
            # Show the result to the user
            current_response += f"> **✅ Tool Result:**\n> ```\n> {result}\n> ```\n\n"
            yield current_response
            
            # Feed the result back into the prompt for the model
            prompt += f"</tool_call>\n<tool_response>\n{result}\n</tool_response>\nCORTEX: "
            time.sleep(1) # Fake delay
            
        else:
            # The agent didn't output a tool call, so it's done talking!
            break
            
    yield current_response

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

body, .gradio-container {
    font-family: 'Inter', sans-serif !important;
    background-color: #0b0f19 !important;
    color: #f8fafc !important;
}

/* Minimalist panels */
.gr-box, .gr-panel, .gr-accordion, .gr-form, .panel, .chatbot {
    background-color: #0f172a !important;
    border: 1px solid #1e293b !important;
    border-radius: 12px !important;
    box-shadow: none !important;
}

h1, h2, h3, h4, h5, h6, .gr-markdown, label, span, p {
    color: #f8fafc !important;
}

/* Input boxes */
input, textarea {
    background-color: #0a0f1c !important;
    border: 1px solid #1e293b !important;
    color: white !important;
    border-radius: 8px !important;
}
input:focus, textarea:focus {
    border-color: #3b82f6 !important;
    box-shadow: none !important;
}

/* Primary Button */
.primary {
    background-color: #3b82f6 !important;
    border: none !important;
    color: white !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    transition: background-color 0.2s ease !important;
}
.primary:hover {
    background-color: #2563eb !important;
}

/* Chat Bubbles */
.message.user {
    background-color: #1e293b !important;
    border: 1px solid #334155 !important;
    border-radius: 12px 12px 0 12px !important;
}
.message.bot {
    background-color: #0f172a !important;
    border: 1px solid #1e293b !important;
    border-radius: 12px 12px 12px 0 !important;
}
"""

dark_theme = gr.themes.Default(
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
).set(
    body_background_fill="#0b0f19",
    body_text_color="#f8fafc",
    background_fill_primary="#0f172a",
    background_fill_secondary="#0a0f1c",
    border_color_accent="#1e293b",
    border_color_primary="#1e293b",
    color_accent_soft="#1e293b",
    block_background_fill="#0f172a",
    block_border_width="1px",
    block_border_color="#1e293b",
    input_background_fill="#0a0f1c",
    button_primary_background_fill="#3b82f6",
    button_primary_background_fill_hover="#2563eb",
    button_secondary_background_fill="#1e293b",
    button_secondary_background_fill_hover="#334155",
)
"""

with gr.Blocks(title="CORTEX Agentic AI") as demo:
    gr.Markdown("# 🤖 CORTEX Agentic AI")
    gr.Markdown("Chat with the CORTEX Agent! It has been hooked up to a **Python Calculator** tool. Ask it a math question and watch it generate a JSON `<tool_call>`, execute the code in the backend, and return the answer.")
    chatbot = gr.Chatbot(height=500, label="CORTEX Swarm", type="messages")
    with gr.Row():
        msg = gr.Textbox(label="Type your message here...", placeholder="What is 144 / 12?", scale=4)
        submit = gr.Button("Send", scale=1, variant="primary")
        
    def respond(user_message, history):
        # We use a wrapper to handle the generator output properly for Gradio 6
        bot_message_generator = chat_stream(user_message, history)
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": ""})
        
        for partial_bot_message in bot_message_generator:
            history[-1]["content"] = partial_bot_message
            yield "", history

    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    submit.click(respond, [msg, chatbot], [msg, chatbot])

if __name__ == "__main__":
    print("Launching Agent Web UI on port 7861...")
    # We use 7861 so it doesn't conflict if they left demo.py running on 7860
    demo.launch(server_name="0.0.0.0", server_port=7861, css=custom_css, theme=dark_theme)
