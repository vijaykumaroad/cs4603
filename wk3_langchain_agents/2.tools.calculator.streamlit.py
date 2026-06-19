import streamlit as st
from langchain_common import bootstrap_notebook
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.agents import create_agent
from langchain_core.tools import tool
from typing import Union

Number = Union[int, float]

STEP_BY_STEP_INSTRUCTION = (
    "You are a calculator assistant that must use available arithmetic tools. "
    "Always provide a step-by-step solution with numbered steps and then a final line "
    "in the exact format: Final Answer: <value>."
)

CONCISE_INSTRUCTION = (
    "You are a calculator assistant that must use available arithmetic tools. "
    "Respond with only the final numeric answer. No explanation."
)

@tool
def add_numbers(a: Number, b: Number) -> float:
    """Add two numbers together."""
    return float(a + b)


@tool
def subtract_numbers(a: Number, b: Number) -> float:
    """Subtract two numbers."""
    return float(a - b)


@tool
def multiply_numbers(a: Number, b: Number) -> float:
    """Multiply two numbers."""
    return float(a * b)


@tool
def divide_numbers(a: Number, b: Number) -> float:
    """Divide a by b."""
    if float(b) == 0.0:
        raise ValueError("Cannot divide by zero.")
    return float(a / b)


def initialize_session_state():
    """Initialize Streamlit session state."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "response_mode" not in st.session_state:
        st.session_state.response_mode = "Step-by-step"
    if "agent" not in st.session_state:
        # Bootstrap LLM on first load
        try:
            DATABRICKS_TOKEN, DATABRICKS_HOST, DATABRICKS_MODEL, (llm, llm_noreason), databricks_embeddings = bootstrap_notebook()
            tools = [add_numbers, subtract_numbers, multiply_numbers, divide_numbers]
            st.session_state.agent = create_agent(llm_noreason, tools)
            st.session_state.model_name = DATABRICKS_MODEL
        except Exception as e:
            st.error(f"Failed to initialize LLM: {e}")
            st.stop()


def main():
    st.set_page_config(page_title="LangChain Chat", layout="wide")
    st.title("💬 Calculator Agent")
    
    initialize_session_state()
    
    # Display model info
    col1, col2 = st.columns([3, 1])
    with col2:
        st.caption(f"Model: {st.session_state.model_name}")
    
    # Display chat history
    st.subheader("Chat History")
    chat_container = st.container(height=400, border=True)
    
    with chat_container:
        for message in st.session_state.messages:
            if isinstance(message, HumanMessage):
                with st.chat_message("user"):
                    st.markdown(message.content)
            elif isinstance(message, AIMessage):
                with st.chat_message("assistant"):
                    st.markdown(message.content)
    
    # Chat input
    st.subheader("Send a Message")
    user_input = st.text_area(
        "Your message:",
        placeholder="Type your question or message here...",
        height=100,
        label_visibility="collapsed"
    )
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        send_button = st.button("Send", type="primary", use_container_width=True)
    
    with col2:
        clear_button = st.button("Clear Chat", use_container_width=True)
    
    # Handle send button
    if send_button and user_input.strip():
        # Add user message to history
        user_message = HumanMessage(content=user_input)
        st.session_state.messages.append(user_message)

        system_instruction = (
            STEP_BY_STEP_INSTRUCTION
            if st.session_state.response_mode == "Step-by-step"
            else CONCISE_INSTRUCTION
        )
        request_messages = [SystemMessage(content=system_instruction)] + st.session_state.messages
        
        # Show loading state and get LLM response
        with st.spinner("Generating response..."):
            try:
                response = st.session_state.agent.invoke({"messages": request_messages})
                ai_message = AIMessage(content=response["messages"][-1].content)
                st.session_state.messages.append(ai_message)
                st.rerun()
            except Exception as e:
                st.error(f"Error generating response: {e}")
    
    # Handle clear button
    if clear_button:
        st.session_state.messages = []
        st.rerun()
    
    # Sidebar info
    with st.sidebar:
        st.subheader("Response Style")
        st.session_state.response_mode = st.radio(
            "Choose response format",
            options=["Step-by-step", "Concise"],
            index=0 if st.session_state.response_mode == "Step-by-step" else 1,
        )

        st.subheader("About")
        st.markdown("""
        This is a simple chat client powered by:
        - **LangChain**: LLM orchestration framework
        - **Databricks**: Model serving endpoints
        - **Streamlit**: Web UI
        
        ### Features
        - Real-time chat with LLM
        - Message history tracking
        - Easy conversation clearing
        """)


if __name__ == "__main__":
    main()
