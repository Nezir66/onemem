from __future__ import annotations

from dataclasses import dataclass

from .embeddings import cosine
from .index import SidecarIndex
from .markdown_store import MarkdownStore
from .models import MemoryNode, MemoryOperation, utc_now
from .text import normalize_text, stable_hash, tokenize

MERGE_THRESHOLD = 0.82
EMBEDDING_WEIGHT = 0.45
TOKEN_WEIGHT = 0.35
CONCEPT_WEIGHT = 0.20


@dataclass(slots=True)
class MergeCandidate:
    id: str
    source_id: str
    target_id: str
    score: float
    reason: str


class DedupeEngine:
    def __init__(self, store: MarkdownStore, index: SidecarIndex | None = None) -> None:
        self.store = store
        self.index = index

    def candidates(self, *, limit: int = 25, persist: bool = True) -> list[MergeCandidate]:
        facts = [
            node
            for node in self.store.nodes_by_kind("fact")
            if node.status != "deprecated" and not node.archived
        ]
        embeddings = self._embeddings_for(facts)
        candidates: list[MergeCandidate] = []
        for index, left in enumerate(facts):
            for right in facts[index + 1 :]:
                score, reason = self._score_pair(left, right, embeddings)
                if score < MERGE_THRESHOLD:
                    continue
                source, target = self._order(left, right)
                candidate_id = f"merge_{stable_hash(source.id + ':' + target.id, 10)}"
                candidates.append(
                    MergeCandidate(
                        id=candidate_id,
                        source_id=source.id,
                        target_id=target.id,
                        score=round(score, 4),
                        reason=reason,
                    )
                )
        candidates.sort(key=lambda item: item.score, reverse=True)
        trimmed = candidates[:limit]
        if persist and self.index is not None:
            now = utc_now()
            for candidate in trimmed:
                self.index.save_merge_candidate(
                    candidate_id=candidate.id,
                    source_id=candidate.source_id,
                    target_id=candidate.target_id,
                    score=candidate.score,
                    reason=candidate.reason,
                    created_at=now,
                )
        return trimmed

    def operation_for(self, candidate_id: str) -> MemoryOperation:
        if self.index is not None:
            row = self.index.load_merge_candidate(candidate_id)
            if row is not None:
                return MemoryOperation(
                    op="MERGE",
                    node_id=str(row["source_id"]),
                    target_id=str(row["target_id"]),
                    reason=f"dedupe candidate {candidate_id}: {row['reason']}",
                    confidence=float(row["score"]),
                )
        for candidate in self.candidates(limit=500, persist=False):
            if candidate.id == candidate_id:
                return MemoryOperation(
                    op="MERGE",
                    node_id=candidate.source_id,
                    target_id=candidate.target_id,
                    reason=f"dedupe candidate {candidate.id}: {candidate.reason}",
                    confidence=candidate.score,
                )
        raise ValueError(f"Unknown merge candidate: {candidate_id}")

    def mark_resolved(self, candidate_id: str) -> None:
        if self.index is not None:
            self.index.mark_merge_candidate(candidate_id, "applied")

    def _order(self, left: MemoryNode, right: MemoryNode) -> tuple[MemoryNode, MemoryNode]:
        if right.confidence > left.confidence or right.salience > left.salience:
            return left, right
        return right, left

    def _embeddings_for(self, nodes: list[MemoryNode]) -> dict[str, list[float]]:
        if self.index is None:
            return {}
        return self.index.embeddings_for([node.id for node in nodes])

    def _score_pair(
        self,
        left: MemoryNode,
        right: MemoryNode,
        embeddings: dict[str, list[float]],
    ) -> tuple[float, str]:
        if normalize_text(left.body) == normalize_text(right.body):
            return 1.0, "exact normalized match"
        left_tokens = set(tokenize(left.body))
        right_tokens = set(tokenize(right.body))
        if not left_tokens or not right_tokens:
            return 0.0, "insufficient lexical signal"
        token_overlap = len(left_tokens & right_tokens) / len(left_tokens | right_tokens)
        concept_union = set(left.concept_refs) | set(right.concept_refs)
        concept_overlap = (
            len(set(left.concept_refs) & set(right.concept_refs)) / len(concept_union)
            if concept_union
            else 0.0
        )
        embedding_score = 0.0
        left_vec = embeddings.get(left.id)
        right_vec = embeddings.get(right.id)
        if left_vec and right_vec:
            embedding_score = max(0.0, cosine(left_vec, right_vec))
        score = (
            embedding_score * EMBEDDING_WEIGHT
            + token_overlap * TOKEN_WEIGHT
            + concept_overlap * CONCEPT_WEIGHT
        )
        reason_parts: list[str] = []
        if embedding_score >= 0.85:
            reason_parts.append("near-duplicate embeddings")
        elif embedding_score >= 0.6:
            reason_parts.append("similar embeddings")
        if token_overlap >= 0.6:
            reason_parts.append("high lexical overlap")
        if concept_overlap >= 0.5:
            reason_parts.append("shared concepts")
        reason = ", ".join(reason_parts) or "combined similarity"
        return score, reason
