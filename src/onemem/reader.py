from __future__ import annotations

import re
from dataclasses import dataclass, field

from .index import SidecarIndex
from .retrieval import RankedMemory, RetrievalResult, parse_time
from .temporal import detect_temporal_intent
from .text import tokenize

WEAK_EVIDENCE_THRESHOLD = 0.22
MULTI_HOP_MARKERS = ("and", "then", "after", "before", "because", "und", "dann", "danach", "weil")
OPEN_REASONING_MARKERS = ("why", "warum", "explain", "erklär", "how come", "opinion", "meinung")


@dataclass(slots=True)
class ReaderAnswer:
    answer: str
    abstained: bool
    memory_ids: list[str] = field(default_factory=list)
    source_refs: list[str] = field(default_factory=list)
    confidence: float = 0.0
    reason: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "answer": self.answer,
            "abstained": self.abstained,
            "memory_ids": self.memory_ids,
            "source_refs": self.source_refs,
            "confidence": self.confidence,
            "reason": self.reason,
        }


class EvidenceReader:
    """Deterministic reader for temporal, counting, multi-hop, and evidence-grounded answers."""

    def __init__(
        self,
        *,
        weak_threshold: float = WEAK_EVIDENCE_THRESHOLD,
        index: SidecarIndex | None = None,
    ) -> None:
        self.weak_threshold = weak_threshold
        self.index = index

    def answer(self, result: RetrievalResult) -> ReaderAnswer:
        evidence = self._usable_evidence(result.memories)
        if not evidence:
            return self._abstain("no relevant memory found")
        if evidence[0].score < self.weak_threshold:
            return self._abstain("weak evidence only")
        if self._needs_reasoning_beyond_context(result.query, evidence):
            return self._abstain("answer requires reasoning beyond retrieved context")

        intent = detect_temporal_intent(result.query)
        if intent.is_temporal:
            temporal = self._temporal_answer(result.query, evidence)
            if temporal is not None:
                return temporal

        if self._is_counting_query(result.query):
            return self._count_answer(result.query, evidence)

        if self._is_multi_hop_query(result.query):
            multi = self._multi_hop_answer(result.query, evidence)
            if multi is not None:
                return multi

        return self._extractive_answer(result.query, evidence)

    def _usable_evidence(self, memories: list[RankedMemory]) -> list[RankedMemory]:
        return [
            memory
            for memory in memories
            if memory.status != "deprecated" and memory.kind in {"fact", "episode", "summary"}
        ]

    def _temporal_answer(self, query: str, evidence: list[RankedMemory]) -> ReaderAnswer | None:
        dated: list[tuple[RankedMemory, object]] = []
        for memory in evidence:
            date_value = (memory.debug or {}).get("valid_from")
            if not isinstance(date_value, str) or not date_value:
                continue
            try:
                dated.append((memory, parse_time(date_value)))
            except ValueError:
                continue
        if not dated:
            return None

        intent = detect_temporal_intent(query)
        dated.sort(key=lambda item: item[1], reverse=not intent.prefer_earliest)
        chosen = dated[0][0]
        prefix = "The earliest remembered event is" if intent.prefer_earliest else "The most recent remembered event is"
        return self._answer_from_memories(
            f"{prefix}: {chosen.body}",
            [chosen],
            reason="temporal evidence comparison",
        )

    def _count_answer(self, query: str, evidence: list[RankedMemory]) -> ReaderAnswer:
        query_terms = set(tokenize(query))
        relevant = [
            memory
            for memory in evidence
            if query_terms & set(tokenize(f"{memory.title} {memory.body}"))
        ]
        selected = relevant or evidence
        return self._answer_from_memories(
            f"I found {len(selected)} relevant remembered item(s).",
            selected[:8],
            reason="simple evidence count",
        )

    def _extractive_answer(self, query: str, evidence: list[RankedMemory]) -> ReaderAnswer:
        query_terms = set(tokenize(query))
        scored = sorted(
            evidence,
            key=lambda memory: (
                len(query_terms & set(tokenize(f"{memory.title} {memory.body}"))),
                memory.score,
            ),
            reverse=True,
        )
        selected = scored[:3]
        conflict = self._conflict_pair(selected)
        if conflict is not None:
            ids = ", ".join(memory.id for memory in conflict)
            return self._abstain(f"conflicting evidence: {ids}")
        sentence = selected[0].body
        return self._answer_from_memories(sentence, selected, reason="grounded extractive answer")

    def _is_multi_hop_query(self, query: str) -> bool:
        lowered = query.lower()
        tokens = set(tokenize(lowered))
        return sum(1 for marker in MULTI_HOP_MARKERS if marker in tokens) >= 1 and len(tokens) >= 6

    def _multi_hop_answer(self, query: str, evidence: list[RankedMemory]) -> ReaderAnswer | None:
        if self.index is None or not evidence:
            return None
        seed = evidence[0]
        neighbor_rows = self.index.graph_neighbor_details(seed.id, limit=5)
        if not neighbor_rows:
            return None
        neighbor_ids = {row["id"] for row in neighbor_rows if row["status"] != "deprecated"}
        supporting = [memory for memory in evidence if memory.id in neighbor_ids][:2]
        if not supporting:
            return None
        combined = [seed, *supporting]
        sentence = " ".join(memory.body for memory in combined)
        return self._answer_from_memories(
            sentence,
            combined,
            reason="multi-hop evidence chain via graph neighbors",
        )

    def _conflict_pair(self, memories: list[RankedMemory]) -> list[RankedMemory] | None:
        if len(memories) < 2:
            return None
        negation_re = re.compile(r"\b(no|not|never|nicht|kein|keine|keinen)\b")
        for index, left in enumerate(memories):
            for right in memories[index + 1 :]:
                if not self._share_topic(left, right):
                    continue
                left_negated = bool(negation_re.search(left.body.lower()))
                right_negated = bool(negation_re.search(right.body.lower()))
                if left_negated != right_negated:
                    return [left, right]
        return None

    def _share_topic(self, left: RankedMemory, right: RankedMemory) -> bool:
        return bool(set(left.concept_refs) & set(right.concept_refs))

    def _needs_reasoning_beyond_context(self, query: str, evidence: list[RankedMemory]) -> bool:
        lowered = query.lower()
        if not any(marker in lowered for marker in OPEN_REASONING_MARKERS):
            return False
        query_terms = set(tokenize(lowered)) - _STOPWORDS
        if not query_terms:
            return False
        top = evidence[:3]
        overlap = max(
            (len(query_terms & set(tokenize(f"{memory.title} {memory.body}"))) for memory in top),
            default=0,
        )
        return overlap == 0

    def _is_counting_query(self, query: str) -> bool:
        lowered = query.lower()
        return any(marker in lowered for marker in ["how many", "wie viele", "count", "anzahl"])

    def _answer_from_memories(self, text: str, memories: list[RankedMemory], *, reason: str) -> ReaderAnswer:
        assert memories
        memory_ids = [memory.id for memory in memories]
        source_refs = sorted({ref for memory in memories for ref in memory.source_refs})
        confidence = min(0.95, max(memory.score for memory in memories))
        return ReaderAnswer(
            answer=text,
            abstained=False,
            memory_ids=memory_ids,
            source_refs=source_refs,
            confidence=round(confidence, 4),
            reason=reason,
        )

    def _abstain(self, reason: str) -> ReaderAnswer:
        return ReaderAnswer(
            answer="I do not have a reliable memory for that.",
            abstained=True,
            confidence=0.0,
            reason=reason,
        )


_STOPWORDS: set[str] = {
    "the", "a", "an", "is", "are", "was", "were", "to", "of", "and", "or", "i",
    "my", "me", "you", "your", "in", "on", "at", "for", "with", "about",
    "der", "die", "das", "ein", "eine", "und", "oder", "ich", "mein", "mich",
}
