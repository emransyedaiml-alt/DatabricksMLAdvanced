# Databricks notebook source
# MAGIC %md
# MAGIC # Stage C · 01 — Serve the Agent with MLflow `ResponsesAgent`
# MAGIC
# MAGIC *Reference: databricks/app-templates (`agent-*` templates).*
# MAGIC
# MAGIC ## 1. Lesson objective
# MAGIC Wrap the Stage B investigation agent in MLflow's `ResponsesAgent` interface, test it locally, log it to MLflow,
# MAGIC register it in Unity Catalog, and (optionally) deploy it to a Model Serving endpoint.
# MAGIC
# MAGIC ## 2. Business problem
# MAGIC A notebook demo is not a product. Other teams need to call the agent from apps, AI Playground or a REST API, with
# MAGIC versioning, permissions and tracing. A standard interface makes that possible without custom glue code.
# MAGIC
# MAGIC ## 3. Simple explanation from scratch
# MAGIC `ResponsesAgent` is MLflow's standard shape for agents. It follows the OpenAI **Responses API**:
# MAGIC - **input**: a list of messages, e.g. `[{"role": "user", "content": "Why did … fail?"}]`
# MAGIC - **output**: a list of output items (here, one text message), plus optional `custom_outputs` (we return the root
# MAGIC   causes, evidence references and approval requests).
# MAGIC
# MAGIC Because every Databricks agent speaks this shape, Model Serving, AI Playground, the Review App and Databricks Apps
# MAGIC can all call our agent the same way. Our class `EikaResponsesAgent` (`src/agents/responses_agent.py`) simply passes
# MAGIC the last user message to the orchestrator.
# MAGIC
# MAGIC ## 4. Architecture
# MAGIC ```
# MAGIC EikaResponsesAgent.predict(request) → Orchestrator.investigate(request text) → tools → answer + custom_outputs
# MAGIC        ↓ mlflow.pyfunc.log_model (code + config + data)  → Unity Catalog model version → (optional) serving endpoint
# MAGIC ```

# COMMAND ----------

# MAGIC %pip install -q "mlflow>=3.1" databricks-agents pypdf==5.1.0
# MAGIC %restart_python

# COMMAND ----------

import os, sys
notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
REPO_ROOT = "/Workspace" + notebook_path.split("/notebooks/")[0]
sys.path.insert(0, REPO_ROOT); os.chdir(REPO_ROOT)
dbutils.widgets.dropdown("profile", "learning", ["learning", "enterprise"], "Profile")
dbutils.widgets.text("catalog", "main", "UC catalog for the model")
dbutils.widgets.text("schema", "eika", "UC schema for the model")
dbutils.widgets.dropdown("deploy_endpoint", "no", ["no", "yes"], "Deploy a serving endpoint (billable)")
PROFILE = dbutils.widgets.get("profile")
MODEL_NAME = f"{dbutils.widgets.get('catalog')}.{dbutils.widgets.get('schema')}.eika_investigation_agent"
print("Model will be registered as:", MODEL_NAME)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Step-by-step implementation
# MAGIC ### 5.1 Call the agent through the ResponsesAgent interface (no logging yet)
# MAGIC This is exactly the request shape a serving endpoint receives.

# COMMAND ----------

import json
from mlflow.types.responses import ResponsesAgentRequest
from src.agents.responses_agent import EikaResponsesAgent

agent = EikaResponsesAgent(profile=PROFILE, role="analyst")
request = ResponsesAgentRequest(input=[{"role": "user", "content": "Why did yesterday's customer ingestion pipeline fail?"}])
response = agent.predict(request)
payload = response.model_dump()                       # ResponsesAgentResponse is a pydantic model
print(payload["output"][0]["content"][0]["text"])
print("\ncustom_outputs:", json.dumps(payload.get("custom_outputs"), default=str)[:600])

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.2 Log the agent to MLflow and register it in Unity Catalog
# MAGIC - **Models from code:** MLflow 3 does not pickle a `ResponsesAgent` object. We log the file
# MAGIC   `src/agents/agent_model.py`, which builds the agent and calls `mlflow.models.set_model(...)`; `model_config`
# MAGIC   passes the profile and role into it.
# MAGIC - `code_paths` packages our `src/`, `config/` and `data/` folders with the model, so it runs anywhere.
# MAGIC - `resources` declares the endpoints the agent calls. Model Serving then grants the endpoint's identity access to
# MAGIC   them automatically (no tokens in the model).

# COMMAND ----------

import mlflow
from mlflow.models.resources import DatabricksServingEndpoint
from src.utils.config import load_config

cfg = load_config(PROFILE)
user = spark.sql("SELECT current_user()").first()[0]
mlflow.set_experiment(f"/Users/{user}/eika_agent_serving")
mlflow.set_registry_uri("databricks-uc")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {dbutils.widgets.get('catalog')}.{dbutils.widgets.get('schema')}")

input_example = {"input": [{"role": "user", "content": "Why did yesterday's customer ingestion pipeline fail?"}]}
with mlflow.start_run(run_name="eika_responses_agent"):
    info = mlflow.pyfunc.log_model(
        name="eika_agent",
        python_model=f"{REPO_ROOT}/src/agents/agent_model.py",     # models from code: MLflow logs this file
        model_config={"profile": PROFILE, "role": "analyst"},
        code_paths=[f"{REPO_ROOT}/src", f"{REPO_ROOT}/config", f"{REPO_ROOT}/data"],
        pip_requirements=["PyYAML>=6.0", "requests>=2.31", "pypdf>=4.2", "numpy>=1.26", "mlflow>=3.1", "databricks-sdk>=0.40"],
        resources=[DatabricksServingEndpoint(endpoint_name=cfg["llm"]["endpoint"]),
                   DatabricksServingEndpoint(endpoint_name=cfg["embedding"]["endpoint"])],
        input_example=input_example,
        registered_model_name=MODEL_NAME,
    )
print("Logged:", info.model_uri, "| registered version:", info.registered_model_version)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.3 Load the logged model back and call it
# MAGIC This proves the packaged copy (not the workspace files) works.

# COMMAND ----------

loaded = mlflow.pyfunc.load_model(info.model_uri)
out = loaded.predict(input_example)
print(json.dumps(out, default=str)[:800])

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.4 Optional: deploy to a Model Serving endpoint (billable)
# MAGIC `databricks.agents.deploy` creates a serving endpoint with tracing and a Review App. Leave the widget on **no** to skip.

# COMMAND ----------

if dbutils.widgets.get("deploy_endpoint") == "yes":
    import datetime as dt
    from databricks import agents
    from databricks.sdk import WorkspaceClient
    endpoint_name = "agents_" + MODEL_NAME.replace(".", "-")          # the name agents.deploy gives the endpoint
    try:
        deployment = agents.deploy(MODEL_NAME, info.registered_model_version, scale_to_zero=True)
        endpoint_name = deployment.endpoint_name
        print("Endpoint:", endpoint_name, "| Review App:", deployment.review_app_url)
    except ValueError as exc:   # agents.deploy can give up while the container is still building
        print("agents.deploy reported:", exc, "- waiting for the endpoint instead.")
    # building the container takes 15-30 minutes the first time
    ep = WorkspaceClient().serving_endpoints.wait_get_serving_endpoint_not_updating(endpoint_name, timeout=dt.timedelta(minutes=50))
    print("Endpoint state:", ep.state.ready, "| config update:", ep.state.config_update)
    if ep.state.ready and ep.state.ready.value == "READY":
        reply = WorkspaceClient().api_client.do("POST", f"/serving-endpoints/{endpoint_name}/invocations", body=input_example)
        print(reply["output"][0]["content"][0]["text"][:800])
else:
    print("Skipped deployment. Set 'Deploy a serving endpoint' to yes to create one.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.5 Clean up (stops the billing)
# MAGIC The endpoint is kept so you can demo it. When you are done, set **Delete the serving endpoint** to yes and run this cell.

# COMMAND ----------

dbutils.widgets.dropdown("cleanup", "no", ["no", "yes"], "Delete the serving endpoint")
if dbutils.widgets.get("cleanup") == "yes":
    from databricks import agents
    for d in agents.list_deployments():
        if d.model_name == MODEL_NAME:
            agents.delete_deployment(model_name=MODEL_NAME, model_version=d.model_version)
            print("Deleted deployment of version", d.model_version)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Explanation of important code
# MAGIC - `EikaResponsesAgent` builds the orchestrator **lazily**, on the first request, so logging the model does not
# MAGIC   need workspace access.
# MAGIC - `custom_outputs` carries structured results (root causes, evidence refs, approval requests, synthetic-data flag)
# MAGIC   next to the human-readable answer.
# MAGIC - `resources=[DatabricksServingEndpoint(...)]` is how a served model gets permission to call other endpoints.
# MAGIC
# MAGIC ## 7. Execute the example — run the cells above.
# MAGIC ## 8. Expected output
# MAGIC The investigation answer (schema mismatch for the synthetic customer-ingestion failure) from both the in-notebook
# MAGIC agent and the reloaded model; a registered model version in Unity Catalog under `<catalog>.<schema>.eika_investigation_agent`.
# MAGIC
# MAGIC ## 9. Common errors
# MAGIC | Error | Fix |
# MAGIC |---|---|
# MAGIC | `agents.deploy`: `deployment ... is in a failed state` or `currently updating` | The container is still building; the cell then waits for the endpoint. Check **Serving → endpoint → Events** if it ends NOT_READY. |
# MAGIC | `Failed to serialize Python model` | Pass the file path `src/agents/agent_model.py` as `python_model`, not an agent object. |
# MAGIC | `ImportError: ResponsesAgent` | MLflow is older than 3.x: keep the `%pip install "mlflow>=3.1"` cell. |
# MAGIC | `PERMISSION_DENIED` creating the schema or model | Use a catalog/schema where you have CREATE SCHEMA / CREATE MODEL. |
# MAGIC | Reloaded model cannot find `src` | `code_paths` must point at the `src` folder itself. |
# MAGIC
# MAGIC ## 10. Real-world production considerations
# MAGIC Use models-from-code (`python_model="agent.py"` with `mlflow.models.set_model`) for full reproducibility, enable
# MAGIC inference tables on the endpoint, and promote versions with aliases (`champion`) after the gates in notebook 03 pass.
# MAGIC
# MAGIC ## 11. Interview questions
# MAGIC 1. *Why standardise on ResponsesAgent?* One interface for serving, playground, evaluation and apps.
# MAGIC 2. *How does a served agent authenticate to other endpoints?* Declared `resources` give its service identity scoped access.
# MAGIC 3. *What goes in `custom_outputs`?* Structured data clients need besides text: citations, evidence, flags.
