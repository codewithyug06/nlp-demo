"""Basic tools for CORTEX agents."""

import ast
import operator
import json

def calculator(expression: str) -> str:
    """A safe math evaluator."""
    operators = {ast.Add: operator.add, ast.Sub: operator.sub, 
                 ast.Mult: operator.mul, ast.Div: operator.truediv, 
                 ast.Pow: operator.pow, ast.USub: operator.neg}

    def eval_node(node):
        if isinstance(node, ast.Num): 
            return node.n
        elif isinstance(node, ast.BinOp):
            return operators[type(node.op)](eval_node(node.left), eval_node(node.right))
        elif isinstance(node, ast.UnaryOp):
            return operators[type(node.op)](eval_node(node.operand))
        else:
            raise TypeError(node)

    try:
        return str(eval_node(ast.parse(expression, mode='eval').body))
    except Exception as e:
        return f"Error evaluating math: {str(e)}"


def execute_tool(tool_call_json: str) -> str:
    """Parses a JSON tool call and executes the mapped function."""
    try:
        call = json.loads(tool_call_json)
        tool_name = call.get("name")
        args = call.get("args", {})
        
        if tool_name == "calculator":
            return calculator(args.get("expression", ""))
        else:
            return f"Error: Tool '{tool_name}' not found."
    except json.JSONDecodeError:
        return "Error: Invalid JSON format for tool call."
