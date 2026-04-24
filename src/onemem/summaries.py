from __future__ import annotations

from collections import defaultdict

from .markdown_store import MarkdownStore
from .models import MemoryNode, Relation, utc_now
from .text import slugify, stable_hash, title_from_body


class LayeredSummaryBuilder:
    def __init__(self, store: MarkdownStore) -> None:
        self.store = store

    def refresh(self) -> list[MemoryNode]:
        nodes = [node for node in self.store.all_nodes() if not node.archived and node.status != "deprecated"]
        facts = [node for node in nodes if node.kind == "fact"]
        episodes = [node for node in nodes if node.kind == "episode"]
        summaries: list[MemoryNode] = []
        summaries.extend(self._session_summaries(episodes))
        summaries.extend(self._topic_summaries(facts))
        profile = self._profile_summary(facts)
        if profile:
            summaries.append(profile)
        project = self._project_summary(facts)
        if project:
            summaries.append(project)
        for summary in summaries:
            self.store.write(summary)
        return summaries

    def _session_summaries(self, episodes: list[MemoryNode]) -> list[MemoryNode]:
        grouped: dict[str, list[MemoryNode]] = defaultdict(list)
        for episode in episodes:
            for label in episode.concept_refs or ["default"]:
                grouped[label].append(episode)
        return [
            self._summary(
                layer="session",
                key=session,
                title=f"Session summary: {session}",
                items=items,
            )
            for session, items in sorted(grouped.items())
            if session not in {"default", "chat"} and items
        ]

    def _topic_summaries(self, facts: list[MemoryNode]) -> list[MemoryNode]:
        grouped: dict[str, list[MemoryNode]] = defaultdict(list)
        for fact in facts:
            for concept in fact.concept_refs or ["uncategorized"]:
                grouped[concept].append(fact)
        return [
            self._summary(
                layer="topic",
                key=topic,
                title=f"Topic summary: {topic}",
                items=items,
            )
            for topic, items in sorted(grouped.items())
            if len(items) >= 2
        ]

    def _profile_summary(self, facts: list[MemoryNode]) -> MemoryNode | None:
        items = [fact for fact in facts if "user_profile" in fact.concept_refs or "answer_style" in fact.concept_refs]
        if not items:
            return None
        return self._summary(layer="profile", key="user", title="User profile summary", items=items)

    def _project_summary(self, facts: list[MemoryNode]) -> MemoryNode | None:
        items = [fact for fact in facts if "project_architecture" in fact.concept_refs or "memory_backend" in fact.concept_refs]
        if not items:
            return None
        return self._summary(layer="project", key="onemem", title="Project summary: OneMem", items=items)

    def _summary(self, *, layer: str, key: str, title: str, items: list[MemoryNode]) -> MemoryNode:
        now = utc_now()
        summary_id = self._summary_id(layer, key)
        existing = self.store.get(summary_id)
        top_items = sorted(items, key=lambda item: (item.salience, item.confidence), reverse=True)[:10]
        cited = sorted({ref for item in top_items for ref in [item.id, *item.source_refs]})
        lines = "\n".join(f"- {item.body}" for item in top_items)
        body = (
            f"Layer: {layer}\n"
            f"Derived from {len(items)} source item(s). Facts and episodes remain the truth layer.\n\n"
            f"{lines}"
        )
        return MemoryNode(
            id=summary_id,
            kind="summary",
            title=title_from_body(title),
            body=body,
            status="stable",
            confidence=0.72,
            salience=min(0.9, 0.5 + len(top_items) * 0.04),
            created_at=existing.created_at if existing else now,
            updated_at=now,
            source_refs=cited,
            concept_refs=[layer, key],
            relations=[Relation(target_id=item.id, type="summarizes", weight=0.65) for item in top_items],
        )

    def _summary_id(self, layer: str, key: str) -> str:
        if (layer, key) in {("profile", "user"), ("project", "onemem")}:
            return f"summary_{layer}_{key}"
        readable = slugify(key)
        if len(readable) <= 32:
            return f"summary_{layer}_{readable}"
        return f"summary_{layer}_{stable_hash(key, 10)}"
