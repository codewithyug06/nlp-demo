"""A Single AI Agent that can reason and use tools (ReAct Framework)."""

from agents.engine import CortexAgentEngine
from agents.tools import execute_tool

SYSTEM_PROMPT = """You are CORTEX, an advanced AI agent with access to tools.
You can calculate math expressions using the calculator tool.

To use a tool, you must output a JSON block like this:
<tool_call>
{"name": "calculator", "args": {"expression": "25 * 4"}}
</tool_call>

You will then receive a <tool_response> with the result.
Use this information to answer the user's question.

User: """

def run_agent(question: str):
    engine = CortexAgentEngine(device="cpu")
    
    prompt = SYSTEM_PROMPT + question + "\nCORTEX: "
    print(f"\n[USER]: {question}\n")
    
    # Max 5 reasoning steps
    for step in range(5):
        print(f"[THINKING...] (Step {step+1})")
        # Generate until it tries to call a tool or finishes its thought
        output = engine.generate(prompt, max_tokens=150, stop_sequences=["</tool_call>"])
        
        prompt += output
        print(output, end="")
        
        if "<tool_call>" in output:
            # It wanted to call a tool. We parse it out.
            tool_json = output.split("<tool_call>")[-1].strip()
            print(f"</tool_call>\n[EXECUTING TOOL]...")
            
            # Execute the tool
            result = execute_tool(tool_json)
            print(f"[TOOL RESULT]: {result}\n")
            
            # Feed the result back into the prompt
            prompt += f"</tool_call>\n<tool_response>\n{result}\n</tool_response>\nCORTEX: "
        else:
            # It didn't call a tool, so it must be done answering!
            print("\n\n[AGENT FINISHED]")
            break

if __name__ == "__main__":
    run_agent("What is 144 divided by 12?")
