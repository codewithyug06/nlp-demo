"""Multi-Agent Swarm Orchestrator for CORTEX.

Demonstrates two AI agents (Manager and Coder) working together to solve a task.
Since we use the same CORTEX model in memory, we just switch their system prompts!
"""

from agents.engine import CortexAgentEngine

MANAGER_PROMPT = """You are the Swarm Manager.
Your job is to read the user's request and write a clear, 1-sentence instruction 
for the Coder Agent on what Python script to write.

User Request: {request}

Instruction for Coder: """

CODER_PROMPT = """You are the Swarm Coder.
Write a python script that fulfills the Manager's instruction.
Only output the Python code, nothing else.

Manager Instruction: {instruction}

Python Code:
```python
"""

def run_swarm(user_request: str):
    print("====================================")
    print(f"USER REQUEST: {user_request}")
    print("====================================\n")
    
    # We only need to load the engine once!
    engine = CortexAgentEngine(device="cpu")
    
    # ---------------------------------------------------------
    # 1. MANAGER AGENT RUNS
    # ---------------------------------------------------------
    print("\n[MANAGER AGENT IS THINKING...]")
    manager_prompt = MANAGER_PROMPT.format(request=user_request)
    
    # The manager generates the instruction
    manager_instruction = engine.generate(manager_prompt, max_tokens=50)
    print(f"MANAGER OUTPUT:\n{manager_instruction}")
    
    # ---------------------------------------------------------
    # 2. CODER AGENT RUNS
    # ---------------------------------------------------------
    print("\n[CODER AGENT IS THINKING...]")
    coder_prompt = CODER_PROMPT.format(instruction=manager_instruction.strip())
    
    # The coder generates the code
    coder_code = engine.generate(coder_prompt, max_tokens=150, stop_sequences=["```"])
    print(f"CODER OUTPUT:\n```python\n{coder_code}```\n")
    
    print("\n[SWARM FINISHED SUCCESSFULLY]")

if __name__ == "__main__":
    run_swarm("I need a script that calculates the fibonacci sequence up to N.")
