"""
chat/client.py
Modular chat client for CS 4603 PA-1.
Supports streaming, token budgeting, message history management, and multi-turn tool loops.
"""

from typing import List, Dict, Any, Generator, Optional, Union
import warnings
import tiktoken
import mlflow
import json

from utils.client import get_openai_client, DATABRICKS_ENDPOINT, CONTEXT_LIMITS
from .registry import PromptRegistry

from tools.definitions import TOOLS_LIST

class ChatClient:
    """
    Modular chat client supporting streaming, token budgeting, tool binding, 
    message history management, and prompt registry.
    """
    
    def __init__(
        self,
        model: str = None,
        system_prompt: str = "You are a helpful assistant.",
        prompts_dir: str = "prompts",
        budgeting: bool = True
    ):
        """
        Initializes the chat engine.
        
        Args:
            model: Model endpoint name (default: DATABRICKS_ENDPOINT).
            system_prompt: System message to set assistant behavior.
            prompts_dir: Path to directory containing YAML prompt templates.
            budgeting: Whether or not to enable Token Budgeting
        """
        # (Task 1.1)
        self.client = get_openai_client()
        self.model = model if model else DATABRICKS_ENDPOINT
        self.context_limit = CONTEXT_LIMITS.get(self.model, 32768)

        # Initialize prompt registry
        self.prompt_registry = PromptRegistry(prompts_dir)
        
        # (Task 1.3): Initialize self.history. Make sure the system prompt is the first message.
        self.system_prompt = system_prompt
        self.history = [{"role": "system", "content": self.system_prompt}]
        
        # (Task 1.4) Token tracking
        self.budgeting = budgeting

        self.cumulative_tokens = 0
        # self.token_warning_threshold = int(self.context_limit * 0.8)
        self.token_warning_threshold = 0.8


        # Initialize tiktoken for token counting
        try:
            self.encoding = tiktoken.encoding_for_model(self.model)
        except KeyError:
            self.encoding = tiktoken.get_encoding("cl100k_base")


        # (Task 4.1) Tool management
        self.tools = None
    
    @mlflow.trace(name="chat_engine_call", span_type="LLM")
    def chat(
        self,
        user_message: str,
        stream: bool = False,
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> Union[str, Generator[str, None, None]]:
        """
        Tasks 1.1, 1.2, 1.3, 1.4: Main Chat Loop & Streaming Support
        Sends user message to model, handles token budgeting, history management, and returns response.
        
        Args:
            user_message: The user's message.
            stream: If True, returns a generator yielding token strings.
                   If False, returns the complete response as a single string.
            tools: Optional list of tool definitions to pass to the API.
                  If None, uses self.tools (from bind_tools).
        
        Returns:
            If stream=False: Complete response string.
            If stream=True: Generator yielding token strings as they arrive.
        """

        # TODO (Task 1.1): 
        # Append user_message to self.history
        # Add user message to history
        if user_message:
            self.history.append({"role": "user", "content": user_message})

        if self.budgeting:
            self._check_token_budget()

        # Make the standard API call
        api_params = {
            "model": self.model,
            "messages": self.history,
        }

        # Prepare API call parameters
        
        # Extract and append assistant response to history

        # Return the assistant's text response
        active_tools = tools if tools is not None else self.tools
        if active_tools:
            api_params["tools"] = active_tools
        # --------


        # TODO (Task 1.2): 
        # Branch logic. If stream=True, return self._stream_response()

        if stream:
            return self._stream_response(api_params)

        # --------

        response = self.client.chat.completions.create(**api_params)
        response_message = response.choices[0].message

        # TODO (Task 1.3): 
        # Save the assistant's reply back to history
        assistant_msg: Dict[str, Any] = {
            "role": "assistant",
        }
        if response_message.content is not None:
            assistant_msg["content"] = response_message.content
        if response_message.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                } for tc in response_message.tool_calls
            ]

        self.history.append(assistant_msg)

        # --------
        
        # TODO (Task 1.4): 
        # Call budget check before making API request
        # Update self.cumulative_tokens with prompt & completion usage
        if self.budgeting and getattr(response, "usage", None):
            self.cumulative_tokens += response.usage.prompt_tokens + response.usage.completion_tokens

        # --------
        return response_message.content if response_message.content else ""   

    def _truncate_history(self, target_tokens: int):
        """
        Task 1.3: Message History Management
        Truncates the oldest non-system messages to fit within the token budget.
        
        Args:
            target_tokens: Target token count to truncate to.
        """
        # TODO (Task 1.3):
        # Implement FIFO truncation.
        # Hint: Never remove self.history[0] (the system prompt).
        # Hint: Loop until _estimate_history_tokens() <= target_tokens.
        while self._estimate_history_tokens() > target_tokens and len(self.history) > 1:
            self.history.pop(1)        

    def _check_token_budget(self):
        """
        Task 1.4: Token Budgeting
        Emits a UserWarning at 80% of context limit and truncates proactively if needed.
        """
        if not self.budgeting:
            return
        
        # TODO (Task 1.4):
        # Token Calculation
        usage_pct = self._estimate_history_tokens() / self.context_limit if self.context_limit else 0

        # UserWarning
        # Example of user warning:
        # warning_msg = "This is a warning"
        # warnings.warn(warning_msg, UserWarning)
        if usage_pct >= self.token_warning_threshold:
            warning_msg = f"Token budget warning: {self._estimate_history_tokens()}/{self.context_limit} tokens used ({usage_pct:.2%})"
            warnings.warn(warning_msg, UserWarning)

            self._truncate_history(int(self.context_limit * 0.8))
        


    # TODO (Task 4.1):
    def bind_tools(self, tools: List[Dict[str, Any]]):
        """
        Task 4.1: Tool Binding Hooks
        Registers tool schemas to be passed to the LLM in the next API call.
        
        Args:
            tools: List of tool definitions in OpenAI function-calling format.
        """
        self.tools = tools

    # TODO (Task 4.2): 
    def chat_loop(self, user_message: str, max_turns: int = 5, tools = TOOLS_LIST) -> str:
        """
        Task 4.2: Multi-Turn Tool Loop
        Continuously calls the model and executes tools until a text answer is generated.
        """        
        # Map: Tools to their defintions
        # Bind Tools

        # Implement the multi-turn loop.
        # Hint 1: Call self.chat() without streaming.
        # Hint 2: Check the latest history entry. If it contains 'tool_calls', loop.
        # Hint 3: Extract arguments, call the python function from tool_map, and append using add_tool_result().
        # Hint 4: Stop and return the content when a normal text answer arrives.
        # Hint 5: Return a clear fallback string if max_turns is reached without finishing.
        from tools.definitions import calculator, document_lookup
        tool_map = {
            "calculator": calculator,
            "document_lookup": document_lookup
        }
        
        self.bind_tools(tools)
        current_input = user_message
        
        for turn in range(max_turns):

            response = self.chat(current_input if turn == 0 else "", stream=False)
            latest_msg = self.history[-1]

            if latest_msg.get("tool_calls"):

                for tool_call in latest_msg["tool_calls"]:
                    name = tool_call["function"]["name"]
                    tool_id = tool_call["id"]

                    try:
                        args = json.loads(tool_call["function"]["arguments"])
                    except Exception as e:
                        self.add_tool_result(tool_id, f"Argument parse error: {e}")
                        continue

                    if name in tool_map:
                        try:
                            result = tool_map[name](**args)
                            self.add_tool_result(tool_id, str(result))
                        except Exception as e:
                            self.add_tool_result(tool_id, f"Tool error: {e}")
                    else:
                        self.add_tool_result(tool_id, f"Unknown tool: {name}")

                current_input = ""
            else:
                return response

        return "Task incomplete: maximum tool calls reached."

    # ------------------------------------------------------------------
    # Optional Helpers (Provided)
    # ------------------------------------------------------------------
    
    def _stream_response(self, api_params: Dict[str, Any]) -> Generator[str, None, None]:
        """Handles streaming API response, yielding token strings and updating history."""
        stream_payload = self.client.chat.completions.create(**api_params, stream=True)

        full_text = ""
        
        for chunk in stream_payload:
            if chunk.choices:
                token = chunk.choices[0].delta
                if token.content:
                    full_text += token.content
                    yield token.content
        
        self.history.append({"role": "assistant", "content": full_text})
        
        if self.budgeting:
            prompt_cost = self._estimate_history_tokens() - self.get_token_count(full_text)
            completion_cost = self.get_token_count(full_text)
            self.cumulative_tokens += prompt_cost + completion_cost
        

    def add_tool_result(self, tool_call_id: str, result: str):
        """Appends a tool execution result to the conversation history."""
        self.history.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": result
        })

    def get_token_count(self, text: str) -> int:
        """Returns exact token count using tiktoken, or a rough heuristic fallback."""
        if not text:
            return 0
        try:
            return len(self.encoding.encode(text))
        except Exception:
            return max(1, len(text) // 4)

    def _estimate_history_tokens(self) -> int:
        """Estimates total token count of the current message history."""
        total = 0
        for msg in self.history:
            total += 4
            total += self.get_token_count(msg.get("content") or "")
        return total

    def reset_conversation(self, new_system_prompt: Optional[str] = None):
        """Clears message history (retaining system prompt) and resets token count."""
        if new_system_prompt is not None:
            self.system_prompt = new_system_prompt
        self.history = [{"role": "system", "content": self.system_prompt}]
        self.cumulative_tokens = 0
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Returns a snapshot of current conversation metrics."""
        return {
            "message_count": len(self.history),
            "estimated_tokens": self._estimate_history_tokens(),
            "cumulative_tokens_used": self.cumulative_tokens,
            "context_limit": self.context_limit,
            "context_usage_percent": (self._estimate_history_tokens() / self.context_limit) * 100,
            "available_prompts": self.prompt_registry.list_prompts(),
        }
    
    def use_prompt_template(
        self,
        prompt_name: str,
        stream: bool = False,
        tools: Optional[List[Dict[str, Any]]] = None,
        loop: bool = False,
        **template_vars
    ) -> Union[str, Generator[str, None, None]]:
        """Convenience method to render a registered template and send it to the model."""
        rendered_prompt = self.prompt_registry.render(prompt_name, **template_vars)
        if not loop:
            return self.chat(rendered_prompt, stream=stream, tools=tools)
        else:
            self.bind_tools(tools)
            return self.chat_loop(rendered_prompt)