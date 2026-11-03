"""Serve the investigation agent through MLflow's ``ResponsesAgent`` interface (OpenAI Responses API shape).

This is the interface used by Databricks' current agent templates (databricks/app-templates, agent-* folders), so the
agent can be logged with MLflow, deployed to Model Serving or a Databricks App, and used from AI Playground.

Logging uses MLflow "models from code" (see ``src/agents/agent_model.py``):

    import mlflow
    with mlflow.start_run():
        mlflow.pyfunc.log_model(name="eika_agent", python_model="src/agents/agent_model.py",
                                model_config={"profile": "enterprise", "role": "analyst"},
                                code_paths=["src", "config", "data"], pip_requirements=["pyyaml", "requests", "pypdf"])
"""
from __future__ import annotations

import uuid

from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse

from src.security.authorization import Principal


class EikaResponsesAgent(ResponsesAgent):
    """Wraps the orchestrator: the last user message is the investigation request."""

    def __init__(self, profile: str = "learning", role: str = "analyst") -> None:
        self.profile, self.role = profile, role
        self._agent = None

    def _orchestrator(self):
        if self._agent is None:                 # build lazily so model logging does not need workspace access
            from src.assistant import load_agent
            from src.utils.config import load_config
            self._agent = load_agent(load_config(self.profile))
        return self._agent

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        user_messages = [m for m in request.input if getattr(m, "role", None) == "user"]
        text = user_messages[-1].content if user_messages else ""
        if isinstance(text, list):              # content parts
            text = " ".join(part.get("text", "") for part in text if isinstance(part, dict))
        result = self._orchestrator().investigate(str(text), Principal("serving.user", (self.role,)))
        item = self.create_text_output_item(text=result.answer, id=str(uuid.uuid4()))
        return ResponsesAgentResponse(output=[item], custom_outputs={
            "root_causes": result.root_causes, "evidence_refs": [e["ref"] for e in result.evidence],
            "approval_requests": result.approval_requests, "synthetic_data": result.synthetic_data, "error": result.error})
