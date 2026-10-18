"""Settings from environment variables (see .env.example)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _env(k: str, d: str = "") -> str:
    return os.environ.get(k, d)


@dataclass
class Settings:
    artifacts: Path = field(default_factory=lambda: Path(_env("PRAHARI_ARTIFACTS", str(ROOT / "var" / "artifacts"))))
    scale: str = field(default_factory=lambda: _env("PRAHARI_SCALE", "full"))      # full | small
    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "auto"))
    openai_small: str = field(default_factory=lambda: _env("OPENAI_MODEL_SMALL", "gpt-4.1-mini"))
    openai_large: str = field(default_factory=lambda: _env("OPENAI_MODEL_LARGE", "gpt-4.1"))
    azure_small: str = field(default_factory=lambda: _env("AZURE_OPENAI_DEPLOYMENT_SMALL", "gpt-4.1-mini"))
    azure_large: str = field(default_factory=lambda: _env("AZURE_OPENAI_DEPLOYMENT_LARGE", "gpt-4.1"))
    bedrock_small: str = field(default_factory=lambda: _env("BEDROCK_MODEL_SMALL", "amazon.nova-lite-v1:0"))
    bedrock_large: str = field(default_factory=lambda: _env("BEDROCK_MODEL_LARGE", "amazon.nova-pro-v1:0"))
    aws_region: str = field(default_factory=lambda: _env("AWS_REGION", "us-east-1"))
    llm_timeout_s: float = 45.0

    def resolved_provider(self) -> str:
        p = self.llm_provider.lower()
        if p == "auto":
            return "openai" if _env("OPENAI_API_KEY") else "mock"
        return p

    def models(self) -> tuple[str, str]:
        p = self.resolved_provider()
        return {"openai": (self.openai_small, self.openai_large), "azure": (self.azure_small, self.azure_large),
                "bedrock": (self.bedrock_small, self.bedrock_large)}.get(p, ("mock", "mock"))
