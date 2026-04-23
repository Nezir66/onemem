from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import onemem
from onemem.chatbot import MemoryChatbot, build_prompt, extract_output_text
from onemem.markdown_store import MarkdownStore
from onemem.runtime import MemoryRuntime


class FakeChatClient:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return "I will remember that Nora prefers concise summaries."


class MemoryFlowTest(unittest.TestCase):
    def test_public_package_api_exports_runtime(self) -> None:
        self.assertIs(onemem.MemoryRuntime, MemoryRuntime)
        self.assertTrue(hasattr(onemem, "MemoryChatbot"))

    def test_capture_flush_retrieve_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = MemoryRuntime(Path(tmp) / "memory")
            runtime.init()

            episode = runtime.capture(
                "Nora prefers concise technical summaries for the OneMem project. "
                "OneMem stores canonical memory in Markdown files.",
                source="test",
                session="onemem",
                salience=0.8,
            )

            self.assertEqual(episode.kind, "episode")
            self.assertTrue((Path(tmp) / "memory" / "episodes").exists())

            outcome = runtime.flush()
            self.assertGreaterEqual(outcome["facts"], 2)
            self.assertGreaterEqual(outcome["concepts"], 1)
            self.assertEqual(outcome["summaries"], 1)

            result = runtime.retrieve("What does Nora prefer for OneMem?", limit=5)
            context = result.context()

            self.assertIn("Nora prefers concise technical summaries", context)
            self.assertTrue(any(memory.kind == "fact" for memory in result.memories))
            self.assertTrue(any("onemem" in memory.concept_refs for memory in result.memories))

    def test_sidecar_rebuilds_from_canonical_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "memory"
            runtime = MemoryRuntime(root)
            runtime.init()
            runtime.capture(
                "The retrieval layer treats vectors as candidate generators, not truth.",
                source="test",
                session="architecture",
            )
            runtime.flush()

            sidecar = root / ".sidecar" / "index.sqlite3"
            sidecar.unlink()

            runtime.rebuild_index()
            result = runtime.retrieve("candidate generators truth", limit=3)

            self.assertTrue(sidecar.exists())
            self.assertIn("vectors as candidate generators", result.context())

    def test_invalidation_deprecates_without_deleting_truth(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = MemoryRuntime(Path(tmp) / "memory")
            runtime.init()
            runtime.capture("The project color is green.", source="test")
            runtime.flush()

            fact_id = next(node.id for node in runtime.store.nodes_by_kind("fact"))
            invalidated = runtime.invalidate(fact_id, reason="color changed")
            result = runtime.retrieve("project color green", limit=5)

            self.assertEqual(invalidated.status, "deprecated")
            self.assertIsNotNone(invalidated.valid_to)
            self.assertIn("color changed", runtime.store.get(fact_id).body)
            self.assertNotIn(fact_id, {memory.id for memory in result.memories})

    def test_markdown_store_is_source_of_truth(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = MemoryRuntime(Path(tmp) / "memory")
            runtime.init()
            runtime.capture("Source of truth lives in readable Markdown.", source="test")

            nodes = MarkdownStore(Path(tmp) / "memory").all_nodes()

            self.assertEqual(len(nodes), 1)
            self.assertEqual(nodes[0].kind, "episode")
            self.assertIn("readable Markdown", nodes[0].body)

    def test_chatbot_retrieves_memory_and_captures_turn(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = MemoryRuntime(Path(tmp) / "memory")
            runtime.init()
            runtime.capture("Nora prefers concise summaries.", source="test", session="profile")
            runtime.flush()
            client = FakeChatClient()
            bot = MemoryChatbot(runtime, client, auto_flush=False)

            answer = bot.ask("What should you remember about Nora?")

            self.assertIn("Nora", client.prompts[0])
            self.assertIn("concise summaries", client.prompts[0])
            self.assertIn("I will remember", answer)
            episodes = runtime.store.nodes_by_kind("episode")
            self.assertTrue(any("Assistant answered" in episode.body for episode in episodes))

    def test_openai_response_text_extraction(self) -> None:
        self.assertEqual(extract_output_text({"output_text": "hello"}), "hello")
        self.assertEqual(
            extract_output_text(
                {
                    "output": [
                        {"content": [{"type": "output_text", "text": "hello again"}]},
                    ]
                }
            ),
            "hello again",
        )

    def test_prompt_includes_memory_and_history(self) -> None:
        prompt = build_prompt(
            "What now?",
            "Memory says: vectors are candidate generators.",
            [("user", "Hi"), ("assistant", "Hello")],
        )

        self.assertIn("vectors are candidate generators", prompt)
        self.assertIn("user: Hi", prompt)
        self.assertIn("What now?", prompt)


if __name__ == "__main__":
    unittest.main()
