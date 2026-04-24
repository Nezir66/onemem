"""Manual smoke test for the Gemini embedding provider.

Run with a real API key:

    export GEMINI_API_KEY=...
    python3 scripts/gemini_smoke_test.py

Success: prints the configured dimensions, vector norms near 1.0, and cosine
similarities between a few sample texts. Exits 1 if the environment is missing
the key or the API rejects the request.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from onemem.embedding_providers import GeminiEmbeddingProvider  # noqa: E402
from onemem.embeddings import cosine  # noqa: E402
from onemem.env import load_default_env  # noqa: E402


SAMPLES = [
    "The user prefers short, technical answers.",
    "Nezir bought a silver Honda Civic in February 2023.",
    "OneMem stores canonical memory in Markdown files.",
]


def main() -> int:
    load_default_env()
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        print("set GEMINI_API_KEY or GOOGLE_API_KEY in your environment or .env", file=sys.stderr)
        return 1

    provider = GeminiEmbeddingProvider.from_env()
    print(f"provider: {provider.name} model={provider.model} dimensions={provider.dimensions}")

    vectors: list[list[float]] = []
    for text in SAMPLES:
        vector = provider.embed(text)
        norm = sum(component * component for component in vector) ** 0.5
        assert len(vector) == provider.dimensions, (
            f"unexpected dimension {len(vector)} != {provider.dimensions}"
        )
        print(f"  len={len(vector):4d} norm={norm:.4f} text={text!r}")
        vectors.append(vector)

    print("cosine(sample0, sample1) =", round(cosine(vectors[0], vectors[1]), 4))
    print("cosine(sample0, sample2) =", round(cosine(vectors[0], vectors[2]), 4))
    print("cosine(sample1, sample2) =", round(cosine(vectors[1], vectors[2]), 4))
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
