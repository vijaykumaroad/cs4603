"""
rag/router.py
Query routing: logical routing (LLM-based classification) and
semantic routing (embedding similarity).
"""

from __future__ import annotations

from typing import Literal

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from pydantic import BaseModel, Field
from databricks_langchain.embeddings import DatabricksEmbeddings
from langchain_community.utils.math import cosine_similarity


# ---------------------------------------------------------------------------
# Shared prompt templates for all routing domains
# ---------------------------------------------------------------------------

SEMANTIC_ROUTER_PROMPTS: dict[str, str] = {
    "physics": (
        "You are a very smart physics professor. You are great at "
        "answering questions about physics in a concise and easy-to-understand manner. "
        "When you don't know the answer to a question, you admit that you don't know.\n\n"
        "Here is a question:\n{question}"
    ),
    "math": (
        "You are a very good mathematician. You are great at answering "
        "math questions. You are so good because you are able to break down hard "
        "problems into their component parts, answer the component parts, and then "
        "put them together to answer the broader question.\n\n"
        "Here is a question:\n{question}"
    ),
    "cs": (
        "You are a knowledgeable computer science professor. You are great at "
        "answering questions about algorithms, data structures, programming languages, "
        "and software engineering. You explain concepts clearly with examples.\n\n"
        "Here is a question:\n{question}"
    ),
}


# ---------------------------------------------------------------------------
# Logical routing: LLM classifies query via structured output
# ---------------------------------------------------------------------------


class RouteQuery(BaseModel):
    """Route a user query to the most relevant data source."""

    datasource: Literal["physics", "math", "cs"] = Field(
        ...,
        description=(
            "Given a user question, ..."
        ),
    )


LOGICAL_ROUTE_SYSTEM: str = (
    ...
)


def choose_route(result: RouteQuery) -> PromptTemplate:
    """Map a RouteQuery result to the corresponding prompt template."""
    # Extract datasource from result, lowercase it
    # Look up the matching template from SEMANTIC_ROUTER_PROMPTS
    # Return PromptTemplate.from_template(template)
    ...


def build_logical_router(llm=None):
    """Build a logical routing chain.

    Returns an LCEL chain that routes the query to the most relevant domain
    prompt using LLM-based classification, then generates a response.
    """
    # Default llm to get_chat_llm() if None
    # Build system+human prompt via ChatPromptTemplate.from_messages
    # Build a function that: classifies the query via structured output (RouteQuery), maps the result to a PromptTemplate via choose_route, formats the PromptTemplate with the original question, and returns the formatted string
    # Return an LCEL chain that runs the classification function, then pipes through the LLM and StrOutputParser
    ...


# ---------------------------------------------------------------------------
# Semantic routing (Ch3 p84-86): embed prompts, cosine similarity
# ---------------------------------------------------------------------------


class SemanticRouter:
    """Route queries to the most relevant prompt template using embedding similarity."""

    def __init__(self, prompts: dict[str, str], embeddings: DatabricksEmbeddings) -> None:
        # Store prompts and embeddings
        # Extract domain keys and template values
        # Embed all prompt templates via embeddings.embed_documents
        ...

    def route(self, query: str) -> str:
        """Return the domain key with highest cosine similarity to the query."""
        # Embed query via embeddings.embed_query
        # Compute cosine_similarity between query embedding and prompt embeddings
        # Return domain key with highest similarity score
        ...

    def get_prompt(self, query: str) -> PromptTemplate:
        """Return the PromptTemplate for the matched domain."""
        # Get domain via self.route(query)
        # Return PromptTemplate.from_template(self.prompts[domain])
        ...


def build_semantic_router(
    embeddings: DatabricksEmbeddings = None,
    llm=None,
    prompts: dict[str, str] | None = None,
):
    """Build a semantic routing chain: get prompt from router -> give to llm -> parse output.

    Returns an LCEL chain that routes the query to the most similar domain prompt,
    then generates a response using the LLM.
    """
    # Default embeddings to get_embeddings() if None
    # Default llm to get_chat_llm() if None
    # Default prompts to SEMANTIC_ROUTER_PROMPTS if None
    # Create SemanticRouter with prompts and embeddings
    # Build an LCEL chain that extracts the question from the input dict, passes it to router.get_prompt, formats the returned PromptTemplate with the question via format_prompt, then pipes through the LLM and StrOutputParser
    ...
