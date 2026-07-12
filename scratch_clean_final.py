import os
import ast
import re

folder = r"C:\Notes\Nlp\Project\modern_nlp_architectire"

for root, dirs, files in os.walk(folder):
    for file in files:
        if file.endswith(".py"):
            filepath = os.path.join(root, file)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            
            # Replace _cortex with empty string in contents
            content = content.replace("_cortex", "")
            
            try:
                tree = ast.parse(content)
                
                # Remove docstrings
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef, ast.Module)):
                        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
                            node.body.pop(0)
                            
                cleaned = ast.unparse(tree)
            except Exception as e:
                print(f"AST Error in {file}: {e}. Falling back to regex.")
                cleaned = re.sub(r'(?m)^\s*#.*$', '', content)
                cleaned = re.sub(r'(?m)\s+#.*$', '', cleaned)
                cleaned = "\n".join([line for line in cleaned.splitlines() if line.strip() != ""])
            
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(cleaned)
            
            if "_cortex" in file:
                new_file = file.replace("_cortex", "")
                os.rename(filepath, os.path.join(root, new_file))
                print(f"Renamed {file} -> {new_file}")

# Update imports in root scripts
for script in ["demo.py", "train.py", "agent_demo.py", "ablate.py", "train_distributed.py", "train_manager.py"]:
    script_path = os.path.join(r"C:\Notes\Nlp\Project", script)
    if os.path.exists(script_path):
        with open(script_path, "r", encoding="utf-8") as f:
            script_content = f.read()
        
        # Change `cortex.` and `from cortex` to `modern_nlp_architectire`
        script_content = script_content.replace("from cortex ", "from modern_nlp_architectire ")
        script_content = script_content.replace("from cortex.", "from modern_nlp_architectire.")
        script_content = script_content.replace("import cortex\n", "import modern_nlp_architectire\n")
        script_content = script_content.replace("_cortex", "")
        # The user renamed the configs, so replace configs/cortex_ with configs/
        script_content = script_content.replace("configs/cortex_", "configs/")
        
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script_content)

print("Final cleanup complete!")
