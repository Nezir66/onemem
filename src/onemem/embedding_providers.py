from __future__ import annotations

import json
import math
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol

from .embeddings import DIMENSIONS, embed


class EmbeddingProvider(Protocol):
    name: str
    dimensions: int
    model: str

    def embed(self, text: str) -> list[float]:
        ...


@dataclass(slots=True)
class HashEmbeddingProvider:
    dimensions: int = DIMENSIONS
    name: str = "hash"
    model: str = "hash-token-v1"

    def embed(self, text: str) -> list[float]:
        return embed(text, dimensions=self.dimensions)


@dataclass(slots=True)
class GeminiEmbeddingProvider:
    api_key: str
    model: str = "gemini-embedding-001"
    dimensions: int = 768
    base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    name: str = "gemini"

    @classmethod
    def from_env(cls) -> "GeminiEmbeddingProvider":
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY is required for Gemini embeddings")
        return cls(
            api_key=api_key,
            model=os.environ.get("ONEMEM_GEMINI_EMBEDDING_MODEL", "gemini-embedding-001"),
            dimensions=int(os.environ.get("ONEMEM_EMBEDDING_DIMENSIONS", "768")),
            base_url=os.environ.get("ONEMEM_GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta").rstrip("/"),
        )

    def embed(self, text: str) -> list[float]:
        payload = {
            "model": f"models/{self.model}",
            "content": {"parts": [{"text": text}]},
            "outputDimensionality": self.dimensions,
        }
        request = urllib.request.Request(
            f"{self.base_url}/models/{self.model}:embedContent",
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": self.api_key,
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Gemini embedding API error {exc.code}: {detail}") from exc
        return normalize_vector(extract_embedding_values(data))


def provider_from_env() -> EmbeddingProvider:
    provider = os.environ.get("ONEMEM_EMBEDDING_PROVIDER", "hash").strip().lower()
    if provider == "hash":
        return HashEmbeddingProvider(dimensions=int(os.environ.get("ONEMEM_EMBEDDING_DIMENSIONS", str(DIMENSIONS))))
    if provider == "gemini":
        return GeminiEmbeddingProvider.from_env()
    raise ValueError(f"Unknown embedding provider: {provider}")


def extract_embedding_values(data: dict) -> list[float]:
    if isinstance(data.get("embedding"), dict) and isinstance(data["embedding"].get("values"), list):
        return [float(value) for value in data["embedding"]["values"]]
    if isinstance(data.get("embeddings"), list) and data["embeddings"]:
        first = data["embeddings"][0]
        if isinstance(first, dict) and isinstance(first.get("values"), list):
            return [float(value) for value in first["values"]]
    raise RuntimeError("Gemini embedding response did not contain embedding values")


def normalize_vector(values: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0:
        return values
    return [value / norm for value in values]

