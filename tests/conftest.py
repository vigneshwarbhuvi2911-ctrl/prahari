from __future__ import annotations

import pytest

try:
    import langgraph.graph  # noqa: F401
    REAL_LANGGRAPH = True
except ImportError:
    from tests import langgraph_shim
    langgraph_shim.install()
    REAL_LANGGRAPH = False

from prahari.llm import LLMClient
from prahari.pipeline import Engine
from prahari.sim import SimConfig, simulate


@pytest.fixture(scope="session")
def small_ds():
    return simulate(SimConfig.small())


@pytest.fixture(scope="session")
def engine() -> Engine:
    return Engine.build(SimConfig.small(), log=lambda *_: None)


class ScriptedLLM(LLMClient):
    """Stands in for a real model: scripted JSON per step; records prompts."""

    def __init__(self, responses: dict):
        super().__init__("openai", "gpt-4.1-mini", "gpt-4.1")
        self.responses, self.calls = responses, []

    def complete_json(self, system, user, *, tier, step, usage, max_tokens=700):
        self.calls.append({"step": step, "tier": tier, "user": user})
        usage.add(step, self.model_for(tier), len(user) // 4, 80)
        r = self.responses.get(step, {})
        r = r(user) if callable(r) else r
        if isinstance(r, Exception):
            raise r
        return r


def first_alert(engine: Engine, scenario: str):
    return next((t for t in engine.alert_txns() if t.scenario == scenario), None)
