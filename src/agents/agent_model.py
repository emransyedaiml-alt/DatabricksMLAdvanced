"""MLflow "models from code" entry point for ``EikaResponsesAgent``.

MLflow 3 does not pickle ``ResponsesAgent`` objects. Instead, this file is logged as the model: MLflow runs it when the
model is loaded, and ``set_model`` tells MLflow which object answers requests. The profile and role come from the
``model_config`` passed to ``log_model`` (the defaults below are used while developing).

    mlflow.pyfunc.log_model(name="eika_agent", python_model="src/agents/agent_model.py",
                            model_config={"profile": "learning", "role": "analyst"},
                            code_paths=["src", "config", "data"], ...)
"""
import mlflow
from mlflow.models import ModelConfig

from src.agents.responses_agent import EikaResponsesAgent

_config = ModelConfig(development_config={"profile": "learning", "role": "analyst"})
mlflow.models.set_model(EikaResponsesAgent(profile=_config.get("profile"), role=_config.get("role")))
