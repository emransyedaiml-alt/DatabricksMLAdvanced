# Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'src'` | Notebook moved outside the project, or imported alone from a DBC | Import the whole project folder; keep `notebooks/` inside it |
| `Reading PDF files needs the 'pypdf' package` | Setup cell skipped | Run the `%pip install pypdf` cell first |
| `No Databricks credentials` | Running locally without credentials | Set `DATABRICKS_HOST` and `DATABRICKS_TOKEN`, or a CLI profile |
| `HTTP 404` from an endpoint | Endpoint name differs in this workspace | Override `EIKA__EMBEDDING__ENDPOINT` / `EIKA__LLM__ENDPOINT` |
| `error: llm_auth` / `HTTP 403` | No CAN QUERY permission on the endpoint | Ask an admin to grant it |
| `error: llm_rate_limit` | Pay-per-token limits | Lower `embedding.batch_size`, retry later, or use provisioned throughput |
| `vector dimension … != index dimension` | Embedding model changed | Rebuild the index (Stage A 05, *Rebuild* = `yes`) |
| Every question gets "I don't have enough information" | Threshold too high for the model, or empty index | Lower `retrieval.similarity_threshold`; check `diagnostics.top_score` and the index count |
| Irrelevant passages returned | Threshold too low | Raise the threshold; review chunk size |
| Agent: `unknown_pipeline` | Pipeline name not recognised | Use a name in `security.approved_pipelines` |
| Agent: `Not authorised …` | Role lacks the tool, or table/pipeline not approved | Adjust `security.role_tools` / approved lists (deliberately) |
| Agent llm mode: `llm_bad_request` | Endpoint does not support tool calling | Use Llama 3.3 70B Instruct or a Claude endpoint |
| Agent llm mode: `step_limit_reached` | Model kept calling tools | Raise `agent.max_steps` or improve tool descriptions |
| `[CANNOT_DETERMINE_TYPE]` when displaying results | `spark.createDataFrame` cannot type a column that is `None` in every row | Display Python rows with `pd.DataFrame(...)`; use Spark only for tables you save |
| `No module named 'mlflow'` on serverless | Serverless notebooks do not include MLflow | Keep the `%pip install "mlflow>=3.1"` cell |
| `cannot import name 'Sentinel' from 'typing_extensions'` | Packages were upgraded by `%pip` but Python still holds the old ones | Run `%restart_python` straight after `%pip install` |
| `Failed to serialize Python model` when logging the agent | MLflow 3 logs `ResponsesAgent` from code, not as a pickled object | Pass `python_model="src/agents/agent_model.py"` and `model_config` |
| MCP `asyncio.run() cannot be called from a running event loop` | Notebooks already run an event loop | `import nest_asyncio; nest_asyncio.apply()` before using `DatabricksMCPClient` |
| `Databricks managed MCP servers are not enabled` | The Managed MCP preview is off in the workspace | An admin enables **Managed MCP Servers** under workspace settings → Previews |
| `cannot import name 'agents' from 'databricks'` | `databricks-agents` is not installed | Keep it in the `%pip install` cell of Stage C 01 |
| App deployment `Error installing packages` | Conflicting pins in `requirements.txt` (e.g. an old `databricks-vectorsearch` with MLflow 3) | Keep `requirements.txt` to app runtime packages; extras go in `requirements-dev.txt` |
| `Synthetic monitoring data not found` | `data/` missing from the import | Re-import the whole project |
| Vector Search endpoint creation fails | Feature not enabled or no permission | Use the learning profile, or ask an admin |
| Bundle deploy fails on `apps` | Databricks Apps not enabled | Remove the `apps` resource or enable Apps |
