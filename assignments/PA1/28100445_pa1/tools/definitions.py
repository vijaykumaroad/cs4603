"""
tools/definitions.py
Helper functions to convert Python tools to OpenAI schemas, and basic tool scaffolds.
"""
import inspect
import os 
import re
from typing import get_origin, get_args, Optional

def _python_type_to_json_schema(annotation):
    """Converts Python type annotations to JSON Schema types."""
    if annotation is inspect._empty or annotation is str:
        return {"type": "string"}
    if annotation is int:
        return {"type": "integer"}
    if annotation is float:
        return {"type": "number"}
    if annotation is bool:
        return {"type": "boolean"}

    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is list and args:
        return {"type": "array", "items": _python_type_to_json_schema(args[0])}
    if origin is dict:
        return {"type": "object"}

    return {"type": "string"}

def build_tool_from_function(func) -> dict:
    """
    Takes a Python function and builds an OpenAI tool definition.
    It reads the function's docstring for the description and parses type hints.
    """
    sig = inspect.signature(func)
    properties = {}
    required = []

    for name, param in sig.parameters.items():
        schema = _python_type_to_json_schema(param.annotation)
        properties[name] = schema
        if param.default is inspect._empty:
            required.append(name)

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": (func.__doc__ or "").strip() or f"Call {func.__name__}",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }

# ---------------------------------------------------------
# Students: Implement your actual tools below
# ---------------------------------------------------------

def calculator(expression: str) -> str:
    """
    Evaluates a mathematical expression safely.
    Args:
        expression: The mathematical expression to evaluate (e.g. '17 + 25').
    """
    # TODO: Implement a safe eval or math parsing logic
    expr = expression.replace(" ", "")

    if not re.match(r"^[\d\+\-\*\/\(\)\.]+$", expr):
        return "Error: Unsupported character tokens or dangerous operators detected."
        
    try:
        result = eval(expr, {"__builtins__": None}, {})
        return str(result)
    except ZeroDivisionError:
        return "Error: Division by zero is undefined."
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"

# Simple Knowledge Base for Document Lookup
KNOWLEDGE_BASE = {
    "Project Orion": "Project Orion is our Q4 initiative focused on migrating local databases to AWS.",
    "Data Center Code": "The emergency override code for the main data center is OMEGA-88-TANGO.",
    "Quarterly Earnings": "In Q2, the company saw a 14% increase in net profits, totaling $3.2 million."
}

def document_lookup(query: str, directory: Optional[str] = None) -> str:
    """
    Looks up information in the local knowledge base.
    Args:
        query: The search term or question to look up.
    """
    # TODO: Implement local dictionary or file lookup
    # Should first search folder provided
    # If no folder provided, search the KNOWLEDGE_BASE

    # Matching strategy: Keyword overlap
    if not query.strip():
        return "not found"

    query_words = set(re.findall(r"\w+", query.lower()))
    if not query_words:
        return "not found"

    best_match_content = None
    max_overlap = 0

    if directory and os.path.isdir(directory):
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(".txt"):
                    filepath = os.path.join(root, file)
                    try:
                        with open(filepath, "r", encoding="utf-8") as f:
                            content = f.read()
                            content_words = set(re.findall(r"\w+", content.lower()))
                            overlap = len(query_words.intersection(content_words))
                            
                            if overlap > max_overlap:
                                max_overlap = overlap
                                best_match_content = content
                    except Exception:
                        continue

    if max_overlap == 0:
        for key, value in KNOWLEDGE_BASE.items():
            combined_text = f"{key} {value}".lower()
            kb_words = set(re.findall(r"\w+", combined_text))
            overlap = len(query_words.intersection(kb_words))
            
            if overlap > max_overlap:
                max_overlap = overlap
                best_match_content = value

    return best_match_content if max_overlap > 0 else "not found"
    

# Expose definitions automatically
calculator_tool = build_tool_from_function(calculator)
lookup_tool = build_tool_from_function(document_lookup)

TOOLS_LIST = [calculator_tool, lookup_tool]