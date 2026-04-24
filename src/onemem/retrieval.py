from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime

from .index import SidecarIndex
from .temporal import TemporalIntent, detect_temporal_intent, parse_event_date
from .text import tokenize

TEMPORAL_WEIGHT = 0.12

RELATION_TYPE_WEIGHTS: dict[str, float] = {
    "supports": 1.0,
    "contradicts": 0.9,
    "derives_from": 0.85,
    "summarizes": 0.8,
    "mentions_concept": 0.7,
    "about": 0.65,
    "related_to": 0.5,
    "merged_into": 0.0,
}
DEFAULT_RELATION_WEIGHT = 0.5
PPR_DECAY = 0.6


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

    def retrieve(
        self,
        query: str,
        *,
        limit: int = 8,
        include_hypotheses: bool = False,
        reference_date: str | None = None,
    ) -> RetrievalResult:
        candidates: dict[str, tuple[sqlite3.Row, float, float, float]] = {}

        for row in self.index.fts_search(self._fts_query(query), limit=30):
            candidates[row["id"]] = (row, max(0.0, -float(row["lexical_score"])), 0.0, 0.0)

        for row, vector_score in self.index.vector_search(query, limit=30):
            old = candidates.get(row["id"], (row, 0.0, 0.0, 0.0))
            candidates[row["id"]] = (row, old[1], max(old[2], vector_score), old[3])

        seed_ids = list(candidates.keys())[:12]
        neighbor_edges = self.index.graph_neighbor_edges(seed_ids, hops=2)
        neighbor_boosts = self._personalized_boosts(candidates, neighbor_edges)
        for row in self.index.get_rows(set(neighbor_edges.keys())):
            old = candidates.get(row["id"], (row, 0.0, 0.0, 0.0))
            typed_boost = self._typed_relation_boost(row["kind"], neighbor_edges.get(row["id"], []))
            diffused = neighbor_boosts.get(row["id"], 0.0)
            candidates[row["id"]] = (row, old[1], old[2], max(old[3], typed_boost, diffused))

        intent = detect_temporal_intent(query)
        reference_iso = parse_event_date(reference_date) or reference_date
        if intent.is_temporal:
            fallback = self.index.temporal_candidates(
                prefer_earliest=intent.prefer_earliest,
                prefer_latest=intent.prefer_latest,
                before=reference_iso if intent.before else None,
                after=reference_iso if intent.after else None,
                limit=10,
            )
            for row in fallback:
                if row["id"] not in candidates:
                    candidates[row["id"]] = (row, 0.0, 0.0, 0.0)
        retrievable = [
            entry for entry in candidates.values() if self._is_retrievable(entry[0], include_hypotheses)
        ]
        temporal_scores = self._temporal_scores(retrievable, intent, reference_iso)
        ranked = [
            self._rank(row, lexical_score, vector_score, graph_boost, temporal_scores.get(row["id"], 0.0), intent)
            for row, lexical_score, vector_score, graph_boost in retrievable
        ]
        ranked.sort(key=lambda item: (self._kind_budget_rank(item.kind), item.score), reverse=True)
        return RetrievalResult(query=query, memories=self._apply_type_budget(ranked, limit))

    def _rank(
        self,
        row: sqlite3.Row,
        lexical_score: float,
        vector_score: float,
        graph_boost: float,
        temporal_score: float,
        intent: TemporalIntent,
    ) -> RankedMemory:
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
            + temporal_score * TEMPORAL_WEIGHT
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
                "temporal_score": round(temporal_score, 6),
                "temporal_query": intent.is_temporal,
                "valid_from": row["valid_from"] or "",
                "pinned": bool(row["pinned"]),
            },
        )

    def _temporal_scores(
        self,
        entries: list[tuple[sqlite3.Row, float, float, float]],
        intent: TemporalIntent,
        reference_iso: str | None,
    ) -> dict[str, float]:
        if not intent.is_temporal:
            return {}
        dated: list[tuple[str, datetime]] = []
        for row, *_ in entries:
            vf = row["valid_from"]
            if not vf:
                continue
            try:
                dated.append((row["id"], parse_time(vf)))
            except ValueError:
                continue
        if not dated:
            return {}

        reference_dt: datetime | None = None
        if reference_iso:
            try:
                reference_dt = parse_time(reference_iso)
            except ValueError:
                reference_dt = None

        if intent.before and reference_dt:
            dated = [item for item in dated if item[1] <= reference_dt]
        elif intent.after and reference_dt:
            dated = [item for item in dated if item[1] >= reference_dt]
        if not dated:
            return {}

        if intent.prefer_earliest and not intent.prefer_latest:
            dated.sort(key=lambda item: item[1])
        elif intent.prefer_latest and not intent.prefer_earliest:
            dated.sort(key=lambda item: item[1], reverse=True)
        elif reference_dt:
            dated.sort(key=lambda item: abs((item[1] - reference_dt).total_seconds()))
        else:
            dated.sort(key=lambda item: item[1], reverse=True)

        total = max(1, len(dated) - 1)
        return {node_id: 1.0 - index / total for index, (node_id, _) in enumerate(dated)}

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

    def _relation_bonus(self, kind: str) -> float:
        return {"fact": 0.1, "episode": 0.08, "summary": 0.06, "concept": 0.04}.get(kind, 0.04)

    def _typed_relation_boost(self, kind: str, edges: list[tuple[str, float]]) -> float:
        if not edges:
            return self._relation_bonus(kind)
        best = max(
            RELATION_TYPE_WEIGHTS.get(relation_type, DEFAULT_RELATION_WEIGHT) * weight
            for relation_type, weight in edges
        )
        return self._relation_bonus(kind) * best

    def _personalized_boosts(
        self,
        seeds: dict[str, tuple[sqlite3.Row, float, float, float]],
        neighbor_edges: dict[str, list[tuple[str, float]]],
    ) -> dict[str, float]:
        if not neighbor_edges:
            return {}
        seed_strengths: dict[str, float] = {}
        for node_id, (_, lexical, vector, _) in seeds.items():
            seed_strengths[node_id] = max(min(1.0, lexical), vector)
        diffused: dict[str, float] = {}
        for neighbor_id, edges in neighbor_edges.items():
            if not edges:
                continue
            best_edge_strength = max(
                RELATION_TYPE_WEIGHTS.get(relation_type, DEFAULT_RELATION_WEIGHT) * weight
                for relation_type, weight in edges
            )
            if best_edge_strength <= 0:
                continue
            seed_strength = max(seed_strengths.values()) if seed_strengths else 0.0
            diffused[neighbor_id] = PPR_DECAY * best_edge_strength * seed_strength
        return diffused

    def _recency_score(self, timestamp: str) -> float:
        age_days = max(0.0, (datetime.now(UTC) - parse_time(timestamp)).total_seconds() / 86400)
        return math.exp(-age_days / 30)

    def _fts_query(self, query: str) -> str:
        terms = tokenize(query)
        return " OR ".join(terms) if terms else query


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))
