"""Chat client with direct and RAG modes."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from tools.definitions import TOOLS_LIST, TOOL_FUNCTIONS
from utils.client import DATABRICKS_ENDPOINT, get_chat_llm

DEFAULT_RAG_K: int = 4


_chat_client_singleton: Optional["ChatClient"] = None


def get_chat_client() -> "ChatClient":
    """Return the module-level singleton ChatClient."""
    global _chat_client_singleton
    if _chat_client_singleton is None:
        _chat_client_singleton = ChatClient()
    return _chat_client_singleton


class ChatClient:
    """Chat client with direct and RAG modes."""

    def __init__(
        self,
        model: str = DATABRICKS_ENDPOINT,
        system_prompt: str = "You are a helpful assistant.",
    ):
        self.llm = get_chat_llm(model=model)
        self.system_prompt = system_prompt
        self.history: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt}
        ]
        self.tools: List[Dict[str, Any]] = []

    def chat(self, user_message: str) -> str:
        """Send a user message and return the response."""
        self.history.append({"role": "user", "content": user_message})

        messages = self._build_messages()
        response = self.llm.invoke(messages)

        if isinstance(response, AIMessage):
            self.history.append({"role": "assistant", "content": response.content or ""})
            return response.content or ""
        return ""

    def chat_loop(self, user_message: str, max_turns: int = 8) -> str:
        """Multi-turn tool loop: call the model, run tool calls, repeat."""
        self.bind_tools(TOOLS_LIST)
        self.history.append({"role": "user", "content": user_message})

        for _turn in range(max_turns):
            messages = self._build_messages()
            response = self.llm.invoke(messages, tools=self.tools)

            if not isinstance(response, AIMessage):
                return "Task incomplete: non-AIMessage response."

            self.history.append({"role": "assistant", "content": response.content or ""})

            tool_calls = getattr(response, "tool_calls", None) or []
            if not tool_calls:
                return response.content or ""

            for tool_call in tool_calls:
                tool_name = tool_call.get("name", "")
                tool_args = tool_call.get("args", {}) or {}
                tool_call_id = tool_call.get("id", "")

                try:
                    fn = TOOL_FUNCTIONS.get(tool_name)
                    if fn is None:
                        tool_result = f"Error: unknown tool '{tool_name}'."
                    elif isinstance(tool_args, dict):
                        tool_result = str(fn(**tool_args))
                    else:
                        tool_result = str(fn(tool_args))
                except Exception as e:
                    tool_result = f"Tool error in {tool_name}: {e}"

                self.history.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": tool_result,
                })

        return "Task incomplete: maximum tool calls reached."

    def ask(
        self,
        question: str,
        *,
        mode: Literal["direct", "rag"] = "direct",
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Unified entry point: direct chat or RAG chat.

        Two modes:
        - "direct": sends question straight to LLM (basic chat)
        - "rag": retrieves relevant chunks first, then sends them with the
          question to the LLM (answer grounded in documents)

        Args:
            question: The user's question.
            mode: "direct" delegates to chat(); "rag" invokes the LCEL chain.
            history: Optional override for the RAG chain's history placeholder.
        """
        # mode == "direct": route through self.chat(), print response
        # mode == "rag":
        #   Step 1: Obtain answer + docs from rag.chain.answer
        #   Step 2: If no docs returned, print REFUSAL_STRING + sources notice
        #   Step 3: Otherwise, print the answer text
        #   Step 4: Show deduplicated source citations via self._print_sources
        ...

    def _print_sources(self, docs: List[Any]) -> None:
        """Print deduplicated (filename, page) citations."""
        seen: set[tuple[str, str]] = set()
        lines: list[str] = []
        for doc in docs:
            meta = getattr(doc, "metadata", {}) or {}
            source = str(meta.get("source", "unknown"))
            page = meta.get("page")
            page_str = f"p. {int(page) + 1}" if isinstance(page, (int, float)) else "p. --"
            key = (source, page_str)
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"{source} ({page_str})")
        if lines:
            print("Sources: " + ", ".join(lines))
        else:
            print("Sources: (none)")

    def bind_tools(self, tools: List[Dict[str, Any]]):
        """Register tool schemas for the next API call."""
        self.tools = tools

    def reset_conversation(self, new_system_prompt: Optional[str] = None):
        """Clear message history and reset."""
        if new_system_prompt:
            self.system_prompt = new_system_prompt
        self.history = [{"role": "system", "content": self.system_prompt}]

    def _build_messages(self) -> list:
        """Convert history dict to LangChain message objects."""
        messages = []
        for msg in self.history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                messages.append(SystemMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
            elif role == "user":
                messages.append(HumanMessage(content=content))
            # tool messages handled separately by the LLM
        return messages
