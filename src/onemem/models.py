from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

NodeKind = Literal["episode", "fact", "concept", "summary"]
NodeStatus = Literal["ephemeral", "candidate", "stable", "core", "deprecated", "hypothesis"]


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass(slots=True)
class Relation:
    target_id: str
    type: str
    weight: float = 1.0


@dataclass(slots=True)
class MemoryNode:
    id: str
    kind: NodeKind
    title: str
    body: str
    status: NodeStatus
    confidence: float = 0.7
    salience: float = 0.5
    pinned: bool = False
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    valid_from: str | None = None
    valid_to: str | None = None
    source_refs: list[str] = field(default_factory=list)
    entity_refs: list[str] = field(default_factory=list)
    concept_refs: list[str] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)
    archived: bool = False

    def metadata(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "title": self.title,
            "status": self.status,
            "confidence": self.confidence,
            "salience": self.salience,
            "pinned": self.pinned,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
            "source_refs": self.source_refs,
            "entity_refs": self.entity_refs,
            "concept_refs": self.concept_refs,
            "relations": [
                {
                    "target_id": relation.target_id,
                    "type": relation.type,
                    "weight": relation.weight,
                }
                for relation in self.relations
            ],
            "archived": self.archived,
        }

    @classmethod
    def from_parts(cls, metadata: dict[str, Any], body: str) -> "MemoryNode":
        relations = [
            Relation(
                target_id=str(item["target_id"]),
                type=str(item["type"]),
                weight=float(item.get("weight", 1.0)),
            )
            for item in metadata.get("relations", [])
        ]
        return cls(
            id=str(metadata["id"]),
            kind=metadata["kind"],
            title=str(metadata.get("title", metadata["id"])),
            body=body,
            status=metadata["status"],
            confidence=float(metadata.get("confidence", 0.7)),
            salience=float(metadata.get("salience", 0.5)),
            pinned=bool(metadata.get("pinned", False)),
            created_at=str(metadata.get("created_at") or utc_now()),
            updated_at=str(metadata.get("updated_at") or utc_now()),
            valid_from=metadata.get("valid_from"),
            valid_to=metadata.get("valid_to"),
            source_refs=list(metadata.get("source_refs", [])),
            entity_refs=list(metadata.get("entity_refs", [])),
            concept_refs=list(metadata.get("concept_refs", [])),
            relations=relations,
            archived=bool(metadata.get("archived", False)),
        )
