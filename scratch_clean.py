import os
import ast

folder = r"C:\Notes\Nlp\Project\modern_nlp_architectire\cortex"

for root, dirs, files in os.walk(folder):
    for file in files:
        if file.endswith(".py"):
            filepath = os.path.join(root, file)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            
            # Replace _cortex with empty string
            content = content.replace("_cortex", "")
            
            try:
                tree = ast.parse(content)
                
                # Remove docstrings
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef, ast.Module)):
                        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
                            node.body.pop(0)
                            
                # Unparsing the AST natively strips all `#` comments!
                cleaned = ast.unparse(tree)
            except Exception as e:
                print(f"AST Error in {file}: {e}. Falling back to basic regex.")
                import re
                cleaned = re.sub(r'(?m)^\s*#.*$', '', content)
                cleaned = re.sub(r'(?m)\s+#.*$', '', cleaned)
                # remove empty lines
                cleaned = "\n".join([line for line in cleaned.splitlines() if line.strip() != ""])
            
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(cleaned)
            
            if "_cortex" in file:
                new_file = file.replace("_cortex", "")
                os.rename(filepath, os.path.join(root, new_file))
                print(f"Renamed {file} -> {new_file}")

print("Cleanup complete!")
