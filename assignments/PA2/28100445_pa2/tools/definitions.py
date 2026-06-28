"""
tools/definitions.py
Helpers to convert Python functions into OpenAI tool schemas, plus the
calculator tool.  Students add ``rag_search`` manually in Task 3.2.
"""

from __future__ import annotations

import inspect
import re
from typing import Any, Callable, Dict, get_args, get_origin


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
    """Build an OpenAI tool definition from a Python function.

    Reads the function's docstring for the description and parses type
    hints for the parameter schema.
    """
    sig = inspect.signature(func)
    properties = {}
    required = []

    for name, param in sig.parameters.items():
        properties[name] = _python_type_to_json_schema(param.annotation)
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
# Calculator tool (provided)
# ---------------------------------------------------------


def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression.

    Args:
        expression: Numbers and ``+ - * / ( )`` only. No variables, no
            function calls, no names.

    Returns:
        The result as a string, or an error message.
    """
    expr = expression.strip()
    if not expr:
        return "Error: empty expression."
    if not re.fullmatch(r"[\d\s+\-*/().]+", expr):
        return "Error: only digits and + - * / ( ) are allowed."
    try:
        return str(eval(expr))
    except Exception as e:
        return f"Error evaluating expression: {e}"


def rag_search(query: str) -> str:
    """Search the corpus; return relevant passages.

    This is the RAG tool that the LLM can call during the multi-turn tool loop.
    The LLM decides when retrieval is needed.

    Args:
        query: The user's search query.

    Returns:
        Retrieved passages joined by newlines, or "(no results)".
    """
    # Lazy-import get_default_store from rag.chain to break circular dependency
    # Lazy-import filter_docs_by_similarity from rag.chain, get_embeddings from rag.store
    # Create a retriever with k=4 from the default store
    # Invoke retriever with query
    # Filter docs by cosine similarity (threshold=0.3) using filter_docs_by_similarity
    # Join all page_content from results with double newline
    # Return "(no results)" if empty
    ...


# ---------------------------------------------------------
# Export schemas + function registry
# ---------------------------------------------------------

calculator_tool = build_tool_from_function(calculator)
rag_tool = ...

TOOLS_LIST = [calculator_tool, rag_tool]

TOOL_FUNCTIONS: Dict[str, Callable[..., str]] = {
    "calculator": calculator,
    ...
}
