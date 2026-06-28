"""
utils/client.py
Shared Databricks LLM client and pgvector connection string.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

DATABRICKS_TOKEN: str = os.environ.get("DATABRICKS_TOKEN", "")
DATABRICKS_HOST: str = os.environ.get("DATABRICKS_HOST", "")
DATABRICKS_ENDPOINT: str = os.environ.get(
    "DATABRICKS_ENDPOINT", "databricks-meta-llama-3-3-70b-instruct"
)

PGVECTOR_CONNECTION_STRING: str = os.environ.get(
    "CS4603_PG_URL",
    "postgresql+psycopg://langchain:langchain!@localhost:5432/cs4603_vectordb",
)


def get_chat_llm(model: str | None = None, **kwargs) -> ChatOpenAI:
    """ChatOpenAI pointed at Databricks.

    For LCEL pipelines (``prompt | llm | parser``).
    Use the raw OpenAI SDK for direct API calls.
    """
    if not DATABRICKS_TOKEN or not DATABRICKS_HOST:
        raise ValueError(
            "DATABRICKS_TOKEN and DATABRICKS_HOST must be set in .env file."
        )
    return ChatOpenAI(
        model=model or DATABRICKS_ENDPOINT,
        api_key=DATABRICKS_TOKEN,
        base_url=f"{DATABRICKS_HOST}/serving-endpoints",
        **kwargs,
    )
