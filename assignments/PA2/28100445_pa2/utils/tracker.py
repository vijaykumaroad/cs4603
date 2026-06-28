"""
utils/tracker.py
Wrappers and utilities for MLflow experiment tracking and tracing.
"""
import mlflow
import os
from dotenv import load_dotenv

load_dotenv()

DATABRICKS_EMAIL = "27100315@lums.edu.pk"

def setup_mlflow(experiment_name: str):
    """
    Sets the tracking URI and initializes an MLflow experiment.
    """

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    mlflow.set_tracking_uri(tracking_uri)

    if tracking_uri == "databricks" or "databricks.com" in tracking_uri:
        if not experiment_name.startswith("/"):
            experiment_name = f"/Users/{DATABRICKS_EMAIL}/{experiment_name}"

    mlflow.set_experiment(experiment_name)
    print(f"MLflow experiment set: {experiment_name} at {tracking_uri}")

def trace_llm_call(func):
    """
    A decorator that wraps a function with an MLflow span of type 'LLM'.
    """

    import functools
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        @mlflow.trace(name=func.__name__, span_type="LLM")
        def _traced():
            return func(*args, **kwargs)
        return _traced()
        
    return wrapper