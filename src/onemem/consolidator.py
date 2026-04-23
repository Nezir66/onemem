from __future__ import annotations

from collections import Counter

from .markdown_store import MarkdownStore
from .models import MemoryNode, Relation, utc_now
from .text import normalize_text, split_sentences, stable_hash, title_from_body, tokenize


class ConsolidationResult:
    def __init__(self, facts: list[MemoryNode], concepts: list[MemoryNode], summary: MemoryNode | None) -> None:
        self.facts = facts
        self.concepts = concepts
        self.summary = summary


class SimpleConsolidator:
    """Deterministic V1 consolidator for local, inspectable operation."""

    def __init__(self, store: MarkdownStore) -> None:
        self.store = store

    def consolidate(self) -> ConsolidationResult:
        episodes = [node for node in self.store.nodes_by_kind("episode") if not node.archived]
        if not episodes:
            return ConsolidationResult([], [], None)

        concepts = self._build_concepts(episodes)
        facts = self._build_facts(episodes, concepts)
        summary = self._build_summary(episodes, facts, concepts)

        for node in [*concepts, *facts, summary]:
            if node is not None:
                self.store.write(node)

        return ConsolidationResult(facts, concepts, summary)

    def _build_concepts(self, episodes: list[MemoryNode]) -> list[MemoryNode]:
        counter: Counter[str] = Counter()
        labels: list[str] = []
        for episode in episodes:
            counter.update(tokenize(f"{episode.title} {episode.body}"))
            labels.extend(self._session_labels(episode))
            labels.extend(self._inferred_labels(episode.body))
        concept_terms = self._unique([*labels, *[term for term, count in counter.most_common(12) if count >= 1]])[:8]

        concepts: list[MemoryNode] = []
        now = utc_now()
        for term in concept_terms:
            concept_id = f"concept_{stable_hash(term, 10)}"
            concepts.append(
                MemoryNode(
                    id=concept_id,
                    kind="concept",
                    title=term,
                    body=f"Concept anchor for memories related to `{term}`.",
                    status="stable",
                    confidence=0.8,
                    salience=min(1.0, 0.6 if term in labels else 0.45 + counter[term] * 0.08),
                    created_at=self._existing_created_at(concept_id, now),
                    updated_at=now,
                    concept_refs=[term],
                )
            )
        return concepts

    def _build_facts(self, episodes: list[MemoryNode], concepts: list[MemoryNode]) -> list[MemoryNode]:
        concept_terms = [concept.title for concept in concepts]
        grouped: dict[str, list[MemoryNode]] = {}
        sentence_by_id: dict[str, str] = {}

        for episode in episodes:
            for sentence in split_sentences(episode.body):
                fact_id = f"fact_{stable_hash(sentence)}"
                grouped.setdefault(fact_id, []).append(episode)
                sentence_by_id[fact_id] = sentence

        now = utc_now()
        facts: list[MemoryNode] = []
        for fact_id, sources in sorted(grouped.items()):
            sentence = sentence_by_id[fact_id]
            refs = sorted({source.id for source in sources})
            matched_concepts = self._match_concepts(sentence, concept_terms)
            evidence_count = len(refs)
            status = "stable" if evidence_count >= 2 else "candidate"
            relations = [
                Relation(target_id=f"concept_{stable_hash(concept, 10)}", type="mentions_concept", weight=0.8)
                for concept in matched_concepts
            ]
            facts.append(
                MemoryNode(
                    id=fact_id,
                    kind="fact",
                    title=title_from_body(sentence),
                    body=sentence,
                    status=status,
                    confidence=min(0.95, 0.58 + evidence_count * 0.14),
                    salience=min(1.0, 0.45 + len(matched_concepts) * 0.08 + evidence_count * 0.06),
                    created_at=self._existing_created_at(fact_id, now),
                    updated_at=now,
                    valid_from=min(source.created_at for source in sources),
                    source_refs=refs,
                    concept_refs=matched_concepts,
                    relations=relations,
                )
            )
        return facts

    def _build_summary(
        self,
        episodes: list[MemoryNode],
        facts: list[MemoryNode],
        concepts: list[MemoryNode],
    ) -> MemoryNode:
        now = utc_now()
        top_concepts = ", ".join(concept.title for concept in concepts[:5]) or "uncategorized"
        fact_lines = "\n".join(f"- {fact.body}" for fact in facts[:8])
        body = (
            f"Consolidated from {len(episodes)} episode(s).\n\n"
            f"Main concepts: {top_concepts}.\n\n"
            f"Key facts:\n{fact_lines if fact_lines else '- No durable facts extracted yet.'}"
        )
        return MemoryNode(
            id="summary_current",
            kind="summary",
            title="Current memory summary",
            body=body,
            status="stable",
            confidence=0.75,
            salience=0.7,
            created_at=self._existing_created_at("summary_current", now),
            updated_at=now,
            source_refs=sorted({episode.id for episode in episodes}),
            concept_refs=[concept.title for concept in concepts[:5]],
            relations=[
                Relation(target_id=fact.id, type="summarizes", weight=0.7)
                for fact in facts[:8]
            ],
        )

    def _match_concepts(self, sentence: str, concept_terms: list[str]) -> list[str]:
        tokens = set(tokenize(sentence))
        inferred = set(self._inferred_labels(sentence))
        matched = [term for term in concept_terms if term in tokens or term in inferred]
        return matched[:5] or (concept_terms[:1] if concept_terms else ["uncategorized"])

    def _existing_created_at(self, node_id: str, fallback: str) -> str:
        existing = self.store.get(node_id)
        return existing.created_at if existing else fallback

    def _session_labels(self, episode: MemoryNode) -> list[str]:
        return [
            label
            for label in episode.concept_refs
            if label and label not in {"default", "chat"} and not label.startswith("longmemeval")
        ]

    def _inferred_labels(self, text: str) -> list[str]:
        normalized = normalize_text(text)
        labels: list[str] = []
        if any(marker in normalized for marker in ["mein name ist", "my name is", "user asked"]):
            labels.append("user_profile")
        if any(marker in normalized for marker in ["bevorzug", "prefer", "antwort", "answer style", "kurze antwort"]):
            labels.append("answer_style")
        if any(marker in normalized for marker in ["markdown", "sqlite", "sidecar", "source of truth", "canonical"]):
            labels.append("memory_backend")
        if any(marker in normalized for marker in ["onemem", "retrieval", "embedding", "vector", "graph"]):
            labels.append("project_architecture")
        if any(marker in normalized for marker in [" before ", " after ", " first ", " last ", "how many days", "date:"]):
            labels.append("temporal_event")
        if "longmemeval session" in normalized:
            labels.append("benchmark_evidence")
        return labels

    def _unique(self, values: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for value in values:
            if value not in seen:
                seen.add(value)
                result.append(value)
        return result
