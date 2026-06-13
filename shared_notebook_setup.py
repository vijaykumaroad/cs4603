"""Shared setup helpers for Databricks + OpenAI notebooks."""

from dataclasses import dataclass
import os

from dotenv import load_dotenv
from openai import OpenAI


@dataclass(frozen=True)
class DatabricksConfig:
    token: str
    host: str
    endpoint: str


class MissingEnvironmentVariableError(ValueError):
    """Raised when one or more required environment variables are missing."""


def get_databricks_config(validate: bool = True) -> DatabricksConfig:
    """Load Databricks environment variables and return a typed config object."""
    load_dotenv()

    token = os.environ.get("DATABRICKS_TOKEN", "")
    host = os.environ.get("DATABRICKS_HOST", "")
    endpoint = os.environ.get("DATABRICKS_ENDPOINT", "")

    if validate:
        missing = [
            name
            for name, value in {
                "DATABRICKS_TOKEN": token,
                "DATABRICKS_HOST": host,
                "DATABRICKS_ENDPOINT": endpoint,
            }.items()
            if not value
        ]
        if missing:
            missing_text = ", ".join(missing)
            raise MissingEnvironmentVariableError(
                f"Missing required environment variable(s): {missing_text}"
            )

    return DatabricksConfig(token=token, host=host, endpoint=endpoint)


def create_databricks_client(config: DatabricksConfig) -> OpenAI:
    """Create an OpenAI client configured for Databricks model serving endpoints."""
    return OpenAI(
        api_key=config.token,
        base_url=f"{config.host}/serving-endpoints",
    )


def bootstrap_notebook(validate: bool = True):
    """Return notebook-ready variables: token, host, endpoint, and configured client."""
    config = get_databricks_config(validate=validate)
    client = create_databricks_client(config)
    return config.token, config.host, config.endpoint, client


if __name__ == "__main__":
    DATABRICKS_TOKEN, DATABRICKS_HOST, DATABRICKS_ENDPOINT, client = bootstrap_notebook()
