from __future__ import annotations

from datetime import UTC, datetime

from .markdown_store import MarkdownStore
from .models import MemoryNode
from .retrieval import parse_time


class MaintenanceWorker:
    def __init__(self, store: MarkdownStore) -> None:
        self.store = store

    def run(self, *, episode_ttl_days: int = 30, hypothesis_ttl_days: int = 14) -> dict[str, int]:
        archived = 0
        deprecated = 0
        now = datetime.now(UTC)
        referenced = self._referenced_episode_ids()

        for node in self.store.all_nodes():
            if node.pinned or node.status == "core":
                continue
            if node.kind == "episode" and node.id not in referenced:
                if (now - parse_time(node.created_at)).days >= episode_ttl_days and node.salience < 0.5:
                    self._archive(node)
                    archived += 1
            if node.status == "candidate" and node.kind == "fact":
                node.salience = max(0.1, round(node.salience * 0.97, 4))
                node.updated_at = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
                self.store.write(node)
            if node.status == "hypothesis":
                if (now - parse_time(node.created_at)).days >= hypothesis_ttl_days:
                    node.status = "deprecated"
                    node.valid_to = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
                    self.store.write(node)
                    deprecated += 1

        return {"archived": archived, "deprecated": deprecated}

    def _referenced_episode_ids(self) -> set[str]:
        refs: set[str] = set()
        for node in self.store.all_nodes():
            if node.kind in {"fact", "summary"}:
                refs.update(ref for ref in node.source_refs if ref.startswith("episode_"))
        return refs

    def _archive(self, node: MemoryNode) -> None:
        node.archived = True
        self.store.archive(node)

