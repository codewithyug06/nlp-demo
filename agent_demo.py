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
            current_response += f"**🛠️ Executing Tool:** `{tool_json}`"
            yield current_response
            
            time.sleep(1) # Fake delay for visual effect
            
            # Execute the tool safely
            result = execute_tool(tool_json)
            
            # Show the result to the user
            current_response += f"\n\n**✅ Tool Result:** `{result}`\n\n"
            yield current_response
            
            # Feed the result back into the prompt for the model
            prompt += f"</tool_call>\n<tool_response>\n{result}\n</tool_response>\nCORTEX: "
            time.sleep(1) # Fake delay
            
        else:
            # The agent didn't output a tool call, so it's done talking!
            break
            
    yield current_response

with gr.Blocks(title="CORTEX Agentic AI") as demo:
    gr.Markdown("# 🤖 CORTEX Agentic AI")
    gr.Markdown("Chat with the CORTEX Agent! It has been hooked up to a **Python Calculator** tool. Ask it a math question and watch it generate a JSON `<tool_call>`, execute the code in the backend, and return the answer.")
    chatbot = gr.Chatbot(height=500, label="CORTEX Swarm", type="messages")
    with gr.Row():
        msg = gr.Textbox(label="Type your message here...", placeholder="What is 144 / 12?", scale=4)
        submit = gr.Button("Send", scale=1)
        
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
    demo.launch(server_name="0.0.0.0", server_port=7861)
