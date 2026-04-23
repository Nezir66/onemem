from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime

from .index import SidecarIndex
from .text import tokenize


@dataclass(slots=True)
class RankedMemory:
    id: str
    kind: str
    title: str
    body: str
    status: str
    score: float
    source_refs: list[str]
    concept_refs: list[str]
    debug: dict[str, float | bool] | None = None


@dataclass(slots=True)
class RetrievalResult:
    query: str
    memories: list[RankedMemory]

    def context(self, *, include_debug: bool = False) -> str:
        lines: list[str] = []
        for memory in self.memories:
            concepts = ", ".join(memory.concept_refs) or "none"
            header = f"[{memory.kind}:{memory.status}:{memory.id} | score={memory.score:.3f} | concepts={concepts}]"
            if include_debug and memory.debug:
                debug = ", ".join(f"{key}={value}" for key, value in memory.debug.items())
                header = f"{header}\n  debug: {debug}"
            lines.append(f"{header}\n{memory.body}")
        return "\n\n".join(lines)


class RetrievalOrchestrator:
    def __init__(self, index: SidecarIndex) -> None:
        self.index = index

    def retrieve(self, query: str, *, limit: int = 8, include_hypotheses: bool = False) -> RetrievalResult:
        candidates: dict[str, tuple[sqlite3.Row, float, float, float]] = {}

        for row in self.index.fts_search(self._fts_query(query), limit=30):
            candidates[row["id"]] = (row, max(0.0, -float(row["lexical_score"])), 0.0, 0.0)

        for row, vector_score in self.index.vector_search(query, limit=30):
            old = candidates.get(row["id"], (row, 0.0, 0.0, 0.0))
            candidates[row["id"]] = (row, old[1], max(old[2], vector_score), old[3])

        seed_ids = list(candidates.keys())[:12]
        neighbor_ids = self.index.graph_neighbors(seed_ids, hops=2)
        for row in self.index.get_rows(neighbor_ids):
            old = candidates.get(row["id"], (row, 0.0, 0.0, 0.0))
            candidates[row["id"]] = (row, old[1], old[2], max(old[3], 0.08))

        ranked = [
            self._rank(row, lexical_score, vector_score, graph_boost)
            for row, lexical_score, vector_score, graph_boost in candidates.values()
            if self._is_retrievable(row, include_hypotheses)
        ]
        ranked.sort(key=lambda item: (self._kind_budget_rank(item.kind), item.score), reverse=True)
        return RetrievalResult(query=query, memories=self._apply_type_budget(ranked, limit))

    def _rank(self, row: sqlite3.Row, lexical_score: float, vector_score: float, graph_boost: float) -> RankedMemory:
        confidence = float(row["confidence"])
        salience = float(row["salience"])
        recency = self._recency_score(row["updated_at"])
        pinned = 0.12 if row["pinned"] else 0.0
        semantic_score = max(vector_score, graph_boost)
        score = (
            semantic_score * 0.38
            + min(1.0, lexical_score) * 0.22
            + salience * 0.18
            + confidence * 0.14
            + recency * 0.08
            + pinned
        )
        return RankedMemory(
            id=row["id"],
            kind=row["kind"],
            title=row["title"],
            body=row["body"],
            status=row["status"],
            score=round(score, 6),
            source_refs=json.loads(row["source_refs"]),
            concept_refs=json.loads(row["concept_refs"]),
            debug={
                "lexical_score": round(min(1.0, lexical_score), 6),
                "vector_score": round(vector_score, 6),
                "graph_boost": round(graph_boost, 6),
                "salience": round(salience, 6),
                "confidence": round(confidence, 6),
                "recency": round(recency, 6),
                "pinned": bool(row["pinned"]),
            },
        )

    def _is_retrievable(self, row: sqlite3.Row, include_hypotheses: bool) -> bool:
        if row["archived"] or row["status"] == "deprecated":
            return False
        if row["status"] == "hypothesis" and not include_hypotheses:
            return False
        valid_to = row["valid_to"]
        if valid_to:
            return parse_time(valid_to) >= datetime.now(UTC)
        return True

    def _apply_type_budget(self, ranked: list[RankedMemory], limit: int) -> list[RankedMemory]:
        budgets = {"fact": 15, "episode": 5, "summary": 3, "concept": 5}
        used = dict.fromkeys(budgets, 0)
        selected: list[RankedMemory] = []
        for item in ranked:
            if len(selected) >= limit:
                break
            if used[item.kind] >= budgets.get(item.kind, limit):
                continue
            used[item.kind] += 1
            selected.append(item)
        selected.sort(key=lambda item: (self._kind_budget_rank(item.kind), item.score), reverse=True)
        return selected

    def _kind_budget_rank(self, kind: str) -> int:
        return {"fact": 4, "episode": 3, "summary": 2, "concept": 1}.get(kind, 0)

    def _recency_score(self, timestamp: str) -> float:
        age_days = max(0.0, (datetime.now(UTC) - parse_time(timestamp)).total_seconds() / 86400)
        return math.exp(-age_days / 30)

    def _fts_query(self, query: str) -> str:
        terms = tokenize(query)
        return " OR ".join(terms) if terms else query


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))
