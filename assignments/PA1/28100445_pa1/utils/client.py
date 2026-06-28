"""
utils/client.py
Shared Databricks + MLflow client utilities for PA-1.
"""
import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

DATABRICKS_TOKEN = os.environ.get("DATABRICKS_TOKEN")
DATABRICKS_HOST = os.environ.get("DATABRICKS_HOST")
DATABRICKS_ENDPOINT = os.environ.get("DATABRICKS_ENDPOINT")
MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI")

if MLFLOW_URI:
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_URI)

# Define common context limits for token budgeting
CONTEXT_LIMITS = {
    "databricks-meta-llama-3-3-70b-instruct": 128000,
    "databricks-meta-llama-3-1-70b-instruct": 128000,
    "databricks-meta-llama-3-1-8b-instruct": 128000,
    "databricks-gpt-oss-120b": 128000,
    "databricks-qwen35-122b-a10b": 128000,
    "databricks-gemini-3-5-flash": 1000000,
    "databricks-dbrx-instruct": 32768,
    "databricks-mixtral-8x7b-instruct": 32768,
}

def get_openai_client() -> OpenAI:
    """
    Initializes and returns an OpenAI client configured for Databricks Model Serving.
    """
    if not DATABRICKS_TOKEN or not DATABRICKS_HOST:
        raise ValueError("Missing DATABRICKS_TOKEN or DATABRICKS_HOST. Check your .env file")
    base_url = f"{DATABRICKS_HOST}/serving-endpoints"
    return OpenAI(api_key=DATABRICKS_TOKEN, base_url=base_url)

def get_context_limit(model_name: str) -> int:
    #return the context length limit for a given model endpoint name 
    if not model_name:
        return 32768
    return CONTEXT_LIMITS.get(model_name, 32768)  