from __future__ import annotations

import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path

from onemem.maintenance import MaintenanceWorker
from onemem.models import MemoryNode, utc_now
from onemem.runtime import MemoryRuntime


def _iso_days_ago(days: int) -> str:
    return (
        (datetime.now(UTC) - timedelta(days=days))
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


class MaintenanceTest(unittest.TestCase):
    def _runtime(self, tmp: str) -> MemoryRuntime:
        runtime = MemoryRuntime(Path(tmp) / "memory")
        runtime.init()
        return runtime

    def _write_episode(
        self,
        runtime: MemoryRuntime,
        *,
        episode_id: str = "episode_test",
        body: str = "An old episode.",
        created_days_ago: int = 60,
        salience: float = 0.3,
        pinned: bool = False,
        status: str = "ephemeral",
    ) -> MemoryNode:
        created_at = _iso_days_ago(created_days_ago)
        node = MemoryNode(
            id=episode_id,
            kind="episode",
            title="old",
            body=body,
            status=status,
            confidence=0.6,
            salience=salience,
            pinned=pinned,
            created_at=created_at,
            updated_at=created_at,
            valid_from=created_at,
            source_refs=["manual"],
            concept_refs=["test"],
        )
        runtime.store.write(node)
        return node

    def test_pinned_episode_is_not_archived(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = self._runtime(tmp)
            self._write_episode(runtime, episode_id="episode_pinned", pinned=True)

            MaintenanceWorker(runtime.store).run()

            node = runtime.store.get("episode_pinned")
            self.assertIsNotNone(node)
            self.assertFalse(node.archived)

    def test_core_episode_is_not_archived(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = self._runtime(tmp)
            self._write_episode(runtime, episode_id="episode_core", status="core")

            MaintenanceWorker(runtime.store).run()

            node = runtime.store.get("episode_core")
            self.assertIsNotNone(node)
            self.assertFalse(node.archived)

    def test_old_low_salience_episode_is_archived(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = self._runtime(tmp)
            self._write_episode(runtime, episode_id="episode_stale", created_days_ago=45, salience=0.2)

            outcome = MaintenanceWorker(runtime.store).run(episode_ttl_days=30)
            runtime.rebuild_index()

            self.assertEqual(outcome["archived"], 1)
            # after archive, retrieval must not surface it
            result = runtime.retrieve("old episode", limit=5)
            self.assertNotIn("episode_stale", {memory.id for memory in result.memories})

    def test_deprecated_fact_is_not_retrievable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = self._runtime(tmp)
            runtime.capture("The project color is green.", source="test", session="colors")
            runtime.flush()
            fact_id = next(node.id for node in runtime.store.nodes_by_kind("fact"))
            runtime.invalidate(fact_id, reason="color changed")

            result = runtime.retrieve("project color green", limit=5)

            self.assertNotIn(fact_id, {memory.id for memory in result.memories})

    def test_candidate_fact_salience_decays(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = self._runtime(tmp)
            now = utc_now()
            fact = MemoryNode(
                id="fact_candidate",
                kind="fact",
                title="candidate",
                body="This is a candidate fact.",
                status="candidate",
                confidence=0.6,
                salience=0.5,
                created_at=now,
                updated_at=now,
                source_refs=["episode_whatever"],
                concept_refs=["candidate"],
            )
            runtime.store.write(fact)

            MaintenanceWorker(runtime.store).run()
            updated = runtime.store.get("fact_candidate")

            self.assertIsNotNone(updated)
            self.assertLess(updated.salience, 0.5)
            self.assertGreaterEqual(updated.salience, 0.1)

    def test_hypothesis_expires_to_deprecated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = self._runtime(tmp)
            created_at = _iso_days_ago(20)
            node = MemoryNode(
                id="fact_hypothesis",
                kind="fact",
                title="speculation",
                body="The user might like dark mode.",
                status="hypothesis",
                confidence=0.5,
                salience=0.4,
                created_at=created_at,
                updated_at=created_at,
                source_refs=["episode_whatever"],
                concept_refs=["preferences"],
            )
            runtime.store.write(node)

            outcome = MaintenanceWorker(runtime.store).run(hypothesis_ttl_days=14)
            refreshed = runtime.store.get("fact_hypothesis")

            self.assertEqual(outcome["deprecated"], 1)
            self.assertEqual(refreshed.status, "deprecated")
            self.assertIsNotNone(refreshed.valid_to)

    def test_archived_episode_is_moved_and_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = self._runtime(tmp)
            self._write_episode(runtime, episode_id="episode_gone", created_days_ago=60, salience=0.2)

            MaintenanceWorker(runtime.store).run(episode_ttl_days=30)
            runtime.rebuild_index()

            archive_hits = list((Path(tmp) / "memory" / "archive").rglob("*.md"))
            self.assertEqual(len(archive_hits), 1)
            ids = {memory.id for memory in runtime.retrieve("episode", limit=10).memories}
            self.assertNotIn("episode_gone", ids)


if __name__ == "__main__":
    unittest.main()
