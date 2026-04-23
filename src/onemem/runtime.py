from __future__ import annotations

from pathlib import Path

from .consolidator import SimpleConsolidator
from .embedding_providers import EmbeddingProvider
from .index import SidecarIndex
from .markdown_store import MarkdownStore
from .models import MemoryNode, utc_now
from .retrieval import RetrievalOrchestrator, RetrievalResult
from .text import stable_hash, title_from_body


class MemoryRuntime:
    def __init__(self, root: Path | str = "memory", embedding_provider: EmbeddingProvider | None = None) -> None:
        self.store = MarkdownStore(root)
        self.index = SidecarIndex(Path(root) / ".sidecar" / "index.sqlite3", embedding_provider=embedding_provider)

    def init(self) -> None:
        self.store.ensure()
        self.index.init()

    def capture(
        self,
        observation: str,
        *,
        source: str = "manual",
        session: str = "default",
        salience: float = 0.5,
    ) -> MemoryNode:
        now = utc_now()
        episode_id = f"episode_{now.replace(':', '').replace('-', '').replace('Z', '')}_{stable_hash(observation, 8)}"
        node = MemoryNode(
            id=episode_id,
            kind="episode",
            title=title_from_body(observation),
            body=observation,
            status="ephemeral",
            confidence=0.65,
            salience=salience,
            created_at=now,
            updated_at=now,
            valid_from=now,
            source_refs=[source],
            concept_refs=[session],
        )
        self.store.write(node)
        self.index.upsert(node)
        return node

    def flush(self) -> dict[str, int]:
        result = SimpleConsolidator(self.store).consolidate()
        self.rebuild_index()
        return {
            "facts": len(result.facts),
            "concepts": len(result.concepts),
            "summaries": 1 if result.summary else 0,
        }

    def rebuild_index(self) -> None:
        self.index.rebuild(self.store)

    def retrieve(self, query: str, *, limit: int = 8, include_hypotheses: bool = False) -> RetrievalResult:
        return RetrievalOrchestrator(self.index).retrieve(
            query,
            limit=limit,
            include_hypotheses=include_hypotheses,
        )

    def invalidate(self, node_id: str, *, reason: str = "manual invalidation") -> MemoryNode:
        node = self.store.get(node_id)
        if node is None:
            raise ValueError(f"Unknown memory node: {node_id}")
        now = utc_now()
        node.status = "deprecated"
        node.valid_to = now
        node.updated_at = now
        node.body = f"{node.body}\n\nInvalidated: {reason}".strip()
        self.store.write(node)
        self.index.upsert(node)
        return node
