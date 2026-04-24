from __future__ import annotations

from pathlib import Path

from .consolidator import SimpleConsolidator
from .dedupe import DedupeEngine, MergeCandidate
from .embedding_providers import EmbeddingProvider
from .index import SidecarIndex
from .markdown_store import MarkdownStore
from .models import AppliedOperation, MemoryNode, MemoryOperation, utc_now
from .operations import MemoryOperationManager
from .reader import EvidenceReader, ReaderAnswer
from .retrieval import RetrievalOrchestrator, RetrievalResult
from .summaries import LayeredSummaryBuilder
from .temporal import parse_event_date
from .text import stable_hash, title_from_body


class MemoryRuntime:
    def __init__(self, root: Path | str = "memory", embedding_provider: EmbeddingProvider | None = None) -> None:
        self.root = Path(root)
        self.store = MarkdownStore(root)
        self.index = SidecarIndex(self.root / ".sidecar" / "index.sqlite3", embedding_provider=embedding_provider)
        self.audit_log = self.root / ".sidecar" / "operations.jsonl"

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
        event_date: str | None = None,
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
            valid_from=_resolve_event_date(event_date, now),
            source_refs=[source],
            concept_refs=[session],
        )
        self.store.write(node)
        self.index.upsert(node)
        return node

    def flush(self) -> dict[str, int]:
        result = SimpleConsolidator(self.store).consolidate()
        layered = LayeredSummaryBuilder(self.store).refresh()
        self.rebuild_index()
        return {
            "facts": len(result.facts),
            "concepts": len(result.concepts),
            "summaries": 1 if result.summary else 0,
            "layered_summaries": len(layered),
        }

    def rebuild_index(self) -> None:
        self.index.rebuild(self.store)

    def retrieve(
        self,
        query: str,
        *,
        limit: int = 8,
        include_hypotheses: bool = False,
        reference_date: str | None = None,
    ) -> RetrievalResult:
        return RetrievalOrchestrator(self.index).retrieve(
            query,
            limit=limit,
            include_hypotheses=include_hypotheses,
            reference_date=reference_date,
        )

    def invalidate(self, node_id: str, *, reason: str = "manual invalidation") -> MemoryNode:
        applied = self.apply_operations(
            [MemoryOperation(op="INVALIDATE", node_id=node_id, reason=reason)]
        )
        changed_id = applied[0].changed_ids[0]
        node = self.store.get(changed_id)
        assert node is not None
        return node

    def answer(
        self,
        query: str,
        *,
        limit: int = 8,
        reference_date: str | None = None,
    ) -> ReaderAnswer:
        result = self.retrieve(query, limit=limit, reference_date=reference_date)
        return EvidenceReader(index=self.index).answer(result)

    def apply_operations(self, operations: list[MemoryOperation]) -> list[AppliedOperation]:
        manager = MemoryOperationManager(self.store, self.index, audit_log=self.audit_log)
        applied = manager.apply_many(operations)
        self.rebuild_index()
        return applied

    def merge_candidates(self, *, limit: int = 25) -> list[MergeCandidate]:
        return DedupeEngine(self.store, self.index).candidates(limit=limit)

    def approve_merge(self, candidate_id: str) -> AppliedOperation:
        engine = DedupeEngine(self.store, self.index)
        operation = engine.operation_for(candidate_id)
        applied = self.apply_operations([operation])[0]
        engine.mark_resolved(candidate_id)
        return applied

    def refresh_summaries(self) -> list[MemoryNode]:
        summaries = LayeredSummaryBuilder(self.store).refresh()
        self.rebuild_index()
        return summaries

    def inspect(self, node_id: str) -> MemoryNode:
        node = self.store.get(node_id)
        if node is None:
            raise ValueError(f"Unknown memory node: {node_id}")
        return node

    def list_nodes(self, *, kind: str | None = None, include_archive: bool = False) -> list[MemoryNode]:
        nodes = self.store.all_nodes(include_archive=include_archive)
        if kind:
            nodes = [node for node in nodes if node.kind == kind]
        return sorted(nodes, key=lambda node: (node.kind, node.updated_at, node.id), reverse=True)

    def graph_neighbors(self, node_id: str, *, limit: int = 20) -> list[dict[str, object]]:
        return self.index.graph_neighbor_details(node_id, limit=limit)

    def record_feedback(self, node_id: str, signal: str) -> MemoryNode:
        node = self.inspect(node_id)
        updates = self._feedback_updates(node, signal)
        operation = MemoryOperation(
            op="UPDATE",
            node_id=node.id,
            updates=updates,
            reason=f"feedback:{signal.lower().strip()}",
        )
        self.apply_operations([operation])
        return self.inspect(node_id)

    def _feedback_updates(self, node: MemoryNode, signal: str) -> dict[str, object]:
        normalized = signal.lower().strip()
        if normalized in {"used", "confirmed", "confirm"}:
            return {
                "salience": min(1.0, round(node.salience + 0.08, 4)),
                "confidence": min(0.98, round(node.confidence + 0.04, 4)),
            }
        if normalized in {"corrected", "wrong", "failed"}:
            updates: dict[str, object] = {
                "salience": max(0.1, round(node.salience - 0.12, 4)),
                "confidence": max(0.1, round(node.confidence - 0.15, 4)),
            }
            if normalized in {"corrected", "wrong"} and node.status == "candidate":
                updates["status"] = "hypothesis"
            return updates
        if normalized == "pin":
            return {"pinned": True}
        if normalized == "unpin":
            return {"pinned": False}
        raise ValueError(f"Unsupported feedback signal: {signal}")


def _resolve_event_date(event_date: str | None, fallback: str) -> str:
    if not event_date:
        return fallback
    return parse_event_date(event_date) or event_date
