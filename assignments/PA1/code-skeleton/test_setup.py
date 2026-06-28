"""
test_setup.py
Run this script to verify your Databricks API access and MLflow tracking server.
"""

import mlflow
from utils.client import get_openai_client, DATABRICKS_ENDPOINT
from utils.tracker import setup_mlflow

def main():
    print("1. Connecting to MLflow...")
    setup_mlflow("pa1-setup-test")
    
    print("\n2. Initializing Databricks OpenAI Client...")
    try:
        client = get_openai_client()
        print(f"[SUCCESS] Client initialized successfully targeting endpoint: {DATABRICKS_ENDPOINT}")
    except Exception as e:
        print(f"[ERROR] Failed to initialize client: {e}")
        return

    print("\n3. Running a test trace and model call...")
    
    # We wrap the test call in an active MLflow run
    with mlflow.start_run(run_name="setup_validation"):
        @mlflow.trace(name="test_completion", span_type="LLM")
        def call_model():
            response = client.chat.completions.create(
                model=DATABRICKS_ENDPOINT,
                messages=[{"role": "user", "content": "Respond with exactly: 'Setup is complete!'"}]
            )
            return response.choices[0].message.content
            
        try:
            result = call_model()
            print(f"\nModel Response: {result}")
            print(f"[SUCCESS] API Call successful!")
            
            trace_id = mlflow.get_last_active_trace_id()
            run_id = mlflow.active_run().info.run_id
            
            print(f"[SUCCESS] MLflow Run Logged: {run_id}")
            print(f"[SUCCESS] MLflow Trace Logged: {trace_id}")
            print("\nSetup verification passed. You are ready to start PA-1!")
            
        except Exception as e:
            print(f"[ERROR] API call failed: {e}")

if __name__ == "__main__":
    main()