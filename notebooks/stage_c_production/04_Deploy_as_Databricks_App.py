# Databricks notebook source
# MAGIC %md
# MAGIC # Stage C · 04 — Deploy the Chat App as a Databricks App
# MAGIC
# MAGIC *Reference: databricks/app-templates (`streamlit-chatbot-app`, app resources with `valueFrom`).*
# MAGIC
# MAGIC ## 1. Lesson objective
# MAGIC Deploy `app/app.py` (the Streamlit chat UI for Stage A and Stage B) as a Databricks App, with its model-endpoint
# MAGIC permissions declared as app resources, then open it, and stop it when done.
# MAGIC
# MAGIC ## 2. Business problem
# MAGIC Employees should use the assistant in a browser, signed in with their Databricks identity, without anyone running
# MAGIC a server or handling tokens.
# MAGIC
# MAGIC ## 3. Simple explanation from scratch
# MAGIC A **Databricks App** runs a web app (Streamlit, Gradio, Dash, Flask…) inside the workspace. It has its own
# MAGIC **service principal**. Instead of granting that identity access by hand, the app declares **resources**
# MAGIC (`llm-endpoint`, `embedding-endpoint`) and Databricks grants `CAN_QUERY` on them. `app.yaml` reads the endpoint names
# MAGIC with `valueFrom`, and our config picks them up through `EIKA__LLM__ENDPOINT` / `EIKA__EMBEDDING__ENDPOINT`.
# MAGIC
# MAGIC ## 4. Two ways to deploy
# MAGIC | Way | Command |
# MAGIC |---|---|
# MAGIC | Asset Bundle (recommended, CI/CD) | `databricks bundle deploy -t dev` then `databricks bundle run eika_app -t dev` |
# MAGIC | From this notebook (SDK) | the cells below |
# MAGIC
# MAGIC **Cost:** a running app bills for its compute. The last cell stops it.

# COMMAND ----------

# MAGIC %pip install -q "databricks-sdk>=0.40"
# MAGIC %restart_python

# COMMAND ----------

import os, sys
notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
REPO_ROOT = "/Workspace" + notebook_path.split("/notebooks/")[0]
sys.path.insert(0, REPO_ROOT); os.chdir(REPO_ROOT)
dbutils.widgets.text("app_name", "eika-assistant-dev", "App name (lowercase, hyphens)")
dbutils.widgets.text("llm_endpoint", "databricks-meta-llama-3-3-70b-instruct", "Chat endpoint")
dbutils.widgets.text("embedding_endpoint", "databricks-bge-large-en", "Embedding endpoint")
dbutils.widgets.dropdown("deploy", "no", ["no", "yes"], "Create and deploy the app (billable)")
APP = dbutils.widgets.get("app_name")
print("Source code path:", REPO_ROOT, "| app:", APP)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Step-by-step implementation
# MAGIC ### 5.1 Look at the app entry point

# COMMAND ----------

print(open(f"{REPO_ROOT}/app.yaml").read())

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.2 Create the app with its resources, then deploy the source code

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.apps import (App, AppDeployment, AppResource, AppResourceServingEndpoint,
                                         AppResourceServingEndpointServingEndpointPermission as Perm)

w = WorkspaceClient()
if dbutils.widgets.get("deploy") == "yes":
    resources = [
        AppResource(name="llm-endpoint", serving_endpoint=AppResourceServingEndpoint(
            name=dbutils.widgets.get("llm_endpoint"), permission=Perm.CAN_QUERY)),
        AppResource(name="embedding-endpoint", serving_endpoint=AppResourceServingEndpoint(
            name=dbutils.widgets.get("embedding_endpoint"), permission=Perm.CAN_QUERY)),
    ]
    existing = {a.name for a in w.apps.list()}
    if APP not in existing:
        w.apps.create_and_wait(app=App(name=APP, description="Enterprise Intelligent Knowledge Assistant",
                                       resources=resources))
        print("Created app", APP)
    from time import monotonic, sleep

    deadline = monotonic() + 600
    start_requested = False
    while True:
        app = w.apps.get(APP)
        compute = app.compute_status
        state = getattr(compute.state, "value", compute.state)
        if state == "ACTIVE":
            break
        if state == "STOPPED" and not start_requested:
            w.apps.start(name=APP)
            start_requested = True
        elif state not in {"STARTING", "STOPPING", "STOPPED"}:
            raise RuntimeError(f"App {APP} compute is {state}: {compute.message}")
        if monotonic() >= deadline:
            raise TimeoutError(f"App {APP} compute did not become ACTIVE: {compute.message}")
        sleep(5)

    deployment = w.apps.deploy_and_wait(app_name=APP, app_deployment=AppDeployment(source_code_path=REPO_ROOT))
    app = w.apps.get(APP)
    print("Deployment status:", deployment.status.state, "|", deployment.status.message)
    print("Open the app:", app.url)
else:
    print("Skipped. Set 'Create and deploy the app' to yes.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.3 Check the app's identity and status

# COMMAND ----------

try:
    app = w.apps.get(APP)
    print("State:", app.app_status.state if app.app_status else None, "| compute:", app.compute_status.state if app.compute_status else None)
    print("Service principal:", app.service_principal_name)
    print("Resources:", [(r.name, r.serving_endpoint.name) for r in (app.resources or []) if r.serving_endpoint])
except Exception as exc:  # noqa: BLE001
    print("App not found yet:", type(exc).__name__)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.4 Try it
# MAGIC Open the URL printed above. In **Knowledge Q&A**, ask *"How should failed ingestion jobs be restarted?"*; in
# MAGIC **Pipeline investigation**, ask *"Why did yesterday's customer ingestion pipeline fail?"* (learning data is synthetic).
# MAGIC The first request builds the knowledge index inside the app, so it takes longer.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.5 Stop or delete the app (stops the billing)

# COMMAND ----------

dbutils.widgets.dropdown("teardown", "none", ["none", "stop", "delete"], "After testing")
action = dbutils.widgets.get("teardown")
if action == "stop":
    w.apps.stop_and_wait(name=APP); print("Stopped", APP)
elif action == "delete":
    w.apps.delete(name=APP); print("Deleted", APP)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Explanation of important code
# MAGIC - `AppResource(... permission=CAN_QUERY)` is the notebook equivalent of the `resources:` block in `databricks.yml`.
# MAGIC - `deploy_and_wait(source_code_path=REPO_ROOT)` snapshots the project folder; the app installs `requirements.txt`
# MAGIC   and runs the `command` in `app.yaml`.
# MAGIC - The app runs as its own service principal: grant it SELECT on any Unity Catalog tables it reads (enterprise).
# MAGIC
# MAGIC ## 7. Execute the example — run the cells with *deploy* = yes, test, then teardown = stop.
# MAGIC ## 8. Expected output
# MAGIC `Deployment status: SUCCEEDED`, an app URL, state RUNNING, the two endpoint resources listed, and a working chat UI.
# MAGIC
# MAGIC ## 9. Common errors
# MAGIC | Error | Fix |
# MAGIC |---|---|
# MAGIC | `Apps are not enabled` | Ask an admin to enable Databricks Apps for the workspace. |
# MAGIC | Deployment FAILED, `ModuleNotFoundError` | Check `requirements.txt`; see the app's **Logs** tab. |
# MAGIC | Deployment FAILED, `Error installing packages` | `requirements.txt` has conflicting pins. Keep it to the app's runtime packages; notebook and test extras live in `requirements-dev.txt`. |
# MAGIC | App shows `llm_auth` errors | The endpoint resource is missing or named differently from the config. |
# MAGIC | Name rejected | Use lowercase letters, numbers and hyphens only. |
# MAGIC
# MAGIC ## 10. Real-world production considerations
# MAGIC Deploy through the bundle in CI/CD, use on-behalf-of-user authorization if answers must respect each user's own
# MAGIC permissions, keep the app stateless, and monitor it with the app logs and MLflow traces.
# MAGIC
# MAGIC ## 11. Interview questions
# MAGIC 1. *How does a Databricks App authenticate to model endpoints?* Its service principal, with access granted through declared resources.
# MAGIC 2. *Service principal vs on-behalf-of-user?* SP: same access for everyone; OBO: each request uses the signed-in user's permissions.
