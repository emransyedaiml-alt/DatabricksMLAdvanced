# Databricks notebook source
# MAGIC %md
# MAGIC # 00 - Create the EIKA Job
# MAGIC
# MAGIC Run this notebook to create the Lakeflow job that runs the whole project on **serverless** compute, in order:
# MAGIC
# MAGIC | Stage | Notebooks | What it does |
# MAGIC |---|---|---|
# MAGIC | A - RAG | `stage_a_rag/01` to `07` | Ingest documents, chunk, embed, search, answer with citations, evaluate |
# MAGIC | B - Agent | `stage_b_agentic_ai/01` to `07` | Tools, monitoring, knowledge and data-quality agents, orchestration, evaluation |
# MAGIC | C - Production | `stage_c_production/01` to `04` | ResponsesAgent, chunks table, evaluation gates, app (not deployed) |
# MAGIC
# MAGIC **Before you run it**
# MAGIC - Keep this notebook in the project's top folder, next to `notebooks/` and `src/`. It finds them relative to itself.
# MAGIC - Set **UC catalog** to a catalog you can create a schema in. This notebook stops early if the catalog does not
# MAGIC   exist. In Databricks Free Edition, use `workspace`.
# MAGIC
# MAGIC **What it costs**: serverless compute and Foundation Model API calls. Billable Stage C resources are enabled by default and can include a serving endpoint, Vector Search resources, and a Databricks App. Set **Enable billable Stage C resources** to `no` to skip their creation. Promotion stays off.
# MAGIC The job starts by default. Set **Start the job after creating it** to `no` to save the job without starting it.
# MAGIC
# MAGIC Running it again updates the job instead of creating a duplicate.

# COMMAND ----------

dbutils.widgets.dropdown("profile", "learning", ["learning", "enterprise"], "Profile")
dbutils.widgets.text("catalog", "main", "UC catalog for Stage C")
dbutils.widgets.text("schema", "eika", "UC schema for Stage C")
dbutils.widgets.text("name_suffix", "", "Job name suffix (optional, e.g. your initials)")
dbutils.widgets.dropdown("run_now", "yes", ["no", "yes"], "Start the job after creating it")

dbutils.widgets.dropdown("enable_billable", "yes", ["no", "yes"], "Enable billable Stage C resources")

# COMMAND ----------

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
here = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
ROOT = here.rsplit("/", 1)[0]
NB = f"{ROOT}/notebooks"
PROFILE = dbutils.widgets.get("profile")
CATALOG = dbutils.widgets.get("catalog").strip()
SCHEMA = dbutils.widgets.get("schema").strip()
SUFFIX = dbutils.widgets.get("name_suffix").strip()
RUN_NOW = dbutils.widgets.get("run_now") == "yes"
BILLABLE = dbutils.widgets.get("enable_billable")

try:
    w.workspace.get_status(NB)
except Exception:
    raise FileNotFoundError(f"{NB} was not found. Keep this notebook in the project's top folder.")
try:
    w.api_client.do("GET", f"/api/2.1/unity-catalog/catalogs/{CATALOG}")
except Exception:
    names = sorted(c["name"] for c in w.api_client.do("GET", "/api/2.1/unity-catalog/catalogs").get("catalogs", []))
    raise ValueError(f"Catalog '{CATALOG}' was not found. Set 'UC catalog' to one of: {', '.join(names)}")
print("Project folder:", ROOT)
print("Stage C catalog and schema:", f"{CATALOG}.{SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Define the tasks
# MAGIC Each task waits for the one before it, because Stage B reuses Stage A's index and Stage C reuses both.

# COMMAND ----------

uc = {"catalog": CATALOG, "schema": SCHEMA}

items = [
    ("a1_architecture_setup", "stage_a_rag/01_Architecture_and_Setup", {}),
    ("a2_document_ingestion", "stage_a_rag/02_Document_Ingestion", {}),
    ("a3_text_chunking", "stage_a_rag/03_Text_Chunking", {}),
    ("a4_embedding_generation", "stage_a_rag/04_Embedding_Generation", {}),
    ("a5_vector_search", "stage_a_rag/05_Vector_Search", {}),
    ("a6_rag_pipeline", "stage_a_rag/06_RAG_Pipeline", {}),
    ("a7_rag_evaluation", "stage_a_rag/07_RAG_Evaluation", {}),
    ("b1_agent_architecture", "stage_b_agentic_ai/01_Agentic_AI_Architecture", {}),
    ("b2_tool_calling", "stage_b_agentic_ai/02_Tool_Calling", {}),
    ("b3_monitoring_agent", "stage_b_agentic_ai/03_Monitoring_Agent", {}),
    ("b4_knowledge_agent", "stage_b_agentic_ai/04_Knowledge_Agent", {}),
    ("b5_data_quality_agent", "stage_b_agentic_ai/05_Data_Quality_Agent", {}),
    ("b6_agent_orchestration", "stage_b_agentic_ai/06_Agent_Orchestration", {}),
    ("b7_agent_evaluation", "stage_b_agentic_ai/07_Agent_Evaluation", {}),
    ("c1_responses_agent", "stage_c_production/01_Serve_Agent_with_ResponsesAgent", dict(uc, deploy_endpoint=BILLABLE, cleanup="no")),
    ("c2_vector_search_mcp", "stage_c_production/02_Vector_Search_and_MCP_Retrieval", dict(uc, create_resources=BILLABLE, cleanup="no")),
    ("c3_evaluation_gates", "stage_c_production/03_Evaluation_Gates_and_Promotion", {"promote": "no"}),
    ("c4_databricks_app", "stage_c_production/04_Deploy_as_Databricks_App", {"deploy": BILLABLE, "teardown": "none"}),
]

def chain(items):
    """No cluster on a task means serverless. Each task depends on the previous one."""
    tasks, prev = [], None
    for key, path, params in items:
        task = {"task_key": key, "timeout_seconds": 3600,
                "notebook_task": {"notebook_path": f"{NB}/{path}", "source": "WORKSPACE", "base_parameters": params}}
        if prev:
            task["depends_on"] = [{"task_key": prev}]
        tasks.append(task)
        prev = key
    return tasks

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Create or update the job

# COMMAND ----------

name = "EIKA - GenAI and Agentic AI Full Run" + (f" - {SUFFIX}" if SUFFIX else "")
settings = {
    "name": name,
    "description": "Runs Enterprise Intelligent Knowledge Assistant stages A, B and C in order. Billable Stage C resources: " + BILLABLE + ".",
    "tags": {"project": "eika", "course": "swsacademy"},
    "max_concurrent_runs": 1,
    "parameters": [{"name": "profile", "default": PROFILE}],
    "tasks": chain(items),
}

jobs = w.api_client.do("GET", "/api/2.2/jobs/list", query={"name": name}).get("jobs", [])   # REST: works with any SDK version
if jobs:
    job_id = jobs[0]["job_id"]
    w.api_client.do("POST", "/api/2.2/jobs/reset", body={"job_id": job_id, "new_settings": settings})
    print(f"Updated job {job_id}: {name}")
else:
    job_id = w.api_client.do("POST", "/api/2.2/jobs/create", body=settings)["job_id"]
    print(f"Created job {job_id}: {name}")
print(f"Tasks: {len(items)}. Open it: {w.config.host}/jobs/{job_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Start the job (optional)
# MAGIC The job starts unless **Start the job after creating it** is set to `no`.

# COMMAND ----------

if RUN_NOW:
    run_id = w.api_client.do("POST", "/api/2.2/jobs/run-now", body={"job_id": job_id})["run_id"]
    print(f"Started run {run_id}: {w.config.host}/jobs/{job_id}/runs/{run_id}")
else:
    print("Job not started. Set 'run_now' to yes, or use Run now on the job page.")

dbutils.notebook.exit(str(job_id))
