"""
utils/client.py
Shared Databricks + MLflow client utilities for PA-1.
"""
import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

DATABRICKS_TOKEN = ...
DATABRICKS_HOST = ...
DATABRICKS_ENDPOINT = ...

# Define common context limits for token budgeting
CONTEXT_LIMITS = {
    ...
}

def get_openai_client() -> OpenAI:
    """
    Initializes and returns an OpenAI client configured for Databricks Model Serving.
    """

    return NotImplementedError()