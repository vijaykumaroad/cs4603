1. create python environment following steps in uv-environment.md
2. Signup for your free databricks account from https://www.databricks.com
3. create .env file in the root directory with the following values:
    DATABRICKS_TOKEN=""
    DATABRICKS_HOST="https://<your-workspace>.databricks.com"
    DATABRICKS_ENDPOINT="databricks-llama-4-maverick"
4. Generate DATABRICKS_TOKEN from developer settings under your databricks workspace.
5. You can use any of the endpoints as retreived in the first notebook (1.tokens.ipynb)
6. To start mlflow server: mlflow ui --port 5000