from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from onemem.embedding_providers import HashEmbeddingProvider, extract_embedding_values, normalize_vector
from onemem.runtime import MemoryRuntime


class CountingEmbeddingProvider(HashEmbeddingProvider):
    def __init__(self) -> None:
        super().__init__(dimensions=16)
        self.calls = 0

    def embed(self, text: str) -> list[float]:
        self.calls += 1
        return super().embed(text)


class EmbeddingProviderTest(unittest.TestCase):
    def test_hash_provider_uses_configured_dimensions(self) -> None:
        provider = HashEmbeddingProvider(dimensions=16)

        self.assertEqual(len(provider.embed("semantic memory")), 16)

    def test_extracts_gemini_embedding_response_shapes(self) -> None:
        self.assertEqual(extract_embedding_values({"embedding": {"values": [1, 2]}}), [1.0, 2.0])
        self.assertEqual(extract_embedding_values({"embeddings": [{"values": [3, 4]}]}), [3.0, 4.0])

    def test_normalizes_vectors(self) -> None:
        self.assertEqual(normalize_vector([3.0, 4.0]), [0.6, 0.8])

    def test_sidecar_records_embedding_provider_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = MemoryRuntime(Path(tmp) / "memory", embedding_provider=HashEmbeddingProvider(dimensions=16))
            runtime.init()

            with sqlite3.connect(Path(tmp) / "memory" / ".sidecar" / "index.sqlite3") as conn:
                metadata = dict(conn.execute("SELECT key, value FROM metadata"))

            self.assertEqual(metadata["embedding_provider"], "hash")
            self.assertEqual(metadata["embedding_model"], "hash-token-v1")
            self.assertEqual(metadata["embedding_dimensions"], "16")

    def test_sidecar_caches_embeddings_across_rebuilds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            provider = CountingEmbeddingProvider()
            runtime = MemoryRuntime(Path(tmp) / "memory", embedding_provider=provider)
            runtime.init()
            runtime.capture("Nora prefers concise summaries.", source="test")
            first_calls = provider.calls

            runtime.rebuild_index()
            second_calls = provider.calls

            self.assertEqual(first_calls, second_calls)


if __name__ == "__main__":
    unittest.main()
