from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .index import SidecarIndex
from .markdown_store import MarkdownStore
from .models import AppliedOperation, MemoryNode, MemoryOperation, NodeStatus, Relation, utc_now
from .retrieval import parse_time

STATUSES: set[str] = {"ephemeral", "candidate", "stable", "core", "deprecated", "hypothesis"}
UPDATABLE_STATUSES: set[str] = STATUSES - {"deprecated"}
PROMOTION: dict[str, NodeStatus] = {
    "ephemeral": "candidate",
    "candidate": "stable",
    "stable": "core",
    "hypothesis": "candidate",
}
DEMOTION: dict[str, NodeStatus] = {
    "core": "stable",
    "stable": "candidate",
    "candidate": "hypothesis",
}


@dataclass(slots=True)
class ValidationError:
    operation: MemoryOperation
    message: str


class MemoryOperationManager:
    """Validates proposed memory operations and applies them deterministically."""

    def __init__(
        self,
        store: MarkdownStore,
        index: SidecarIndex | None = None,
        *,
        audit_log: Path | None = None,
    ) -> None:
        self.store = store
        self.index = index
        self.audit_log = audit_log

    def apply_many(self, operations: list[MemoryOperation]) -> list[AppliedOperation]:
        applied: list[AppliedOperation] = []
        for operation in operations:
            self.validate(operation)
            result = self.apply(operation)
            self._record_audit(result)
            applied.append(result)
        return applied

    def _record_audit(self, applied: AppliedOperation) -> None:
        if self.audit_log is None:
            return
        self.audit_log.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "at": utc_now(),
            "op": applied.operation.op,
            "node_id": applied.operation.node_id,
            "target_id": applied.operation.target_id,
            "changed_ids": applied.changed_ids,
            "message": applied.message,
            "reason": applied.operation.reason,
        }
        with self.audit_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")

    def validate(self, operation: MemoryOperation) -> None:
        if not 0.0 <= operation.confidence <= 1.0:
            raise ValueError(f"{operation.op} confidence must be between 0 and 1")
        if operation.op == "ADD":
            if operation.node is None:
                raise ValueError("ADD requires a node")
            if operation.node.kind in {"fact", "summary"} and not operation.node.source_refs:
                raise ValueError("ADD requires provenance in source_refs for facts and summaries")
            if self.store.get(operation.node.id) is not None:
                raise ValueError(f"ADD target already exists: {operation.node.id}")
            if operation.node.status == "deprecated":
                raise ValueError("ADD may not create deprecated nodes")
            return

        if operation.node_id is None:
            raise ValueError(f"{operation.op} requires node_id")
        node = self._require_node(operation.node_id)

        if operation.op == "UPDATE":
            if node.status == "deprecated":
                raise ValueError("UPDATE may not silently overwrite deprecated nodes")
            if not operation.updates:
                raise ValueError("UPDATE requires updates")
            for key, value in operation.updates.items():
                if key in {"id", "kind", "created_at"}:
                    raise ValueError(f"UPDATE may not change immutable field: {key}")
                if key == "status":
                    if value not in UPDATABLE_STATUSES:
                        raise ValueError(f"UPDATE may not set status to {value!r}; use INVALIDATE")
                if key in {"confidence", "salience"}:
                    try:
                        numeric = float(value)
                    except (TypeError, ValueError) as exc:
                        raise ValueError(f"UPDATE {key} must be numeric") from exc
                    if not 0.0 <= numeric <= 1.0:
                        raise ValueError(f"UPDATE {key} must be between 0 and 1")
                if key in {"valid_from", "valid_to"} and value is not None:
                    try:
                        parse_time(str(value))
                    except ValueError as exc:
                        raise ValueError(f"UPDATE {key} must be ISO-8601") from exc
            return

        if operation.op == "INVALIDATE":
            if node.status == "deprecated":
                raise ValueError(f"Node is already deprecated: {node.id}")
            return

        if operation.op == "LINK":
            if not operation.target_id:
                raise ValueError("LINK requires target_id")
            if operation.relation_type is None:
                raise ValueError("LINK requires relation_type")
            self._require_node(operation.target_id)
            if operation.relation_weight <= 0:
                raise ValueError("LINK relation_weight must be positive")
            return

        if operation.op == "MERGE":
            if not operation.target_id:
                raise ValueError("MERGE requires target_id")
            target = self._require_node(operation.target_id)
            if node.id == target.id:
                raise ValueError("MERGE source and target must differ")
            if node.kind != target.kind:
                raise ValueError("MERGE requires nodes of the same kind")
            if node.status == "deprecated":
                raise ValueError("MERGE source is already deprecated")
            if target.status == "deprecated":
                raise ValueError("MERGE target may not be deprecated")
            return

        if operation.op in {"PROMOTE", "DEMOTE"}:
            if node.status == "deprecated":
                raise ValueError(f"{operation.op} may not change deprecated nodes")
            return

        raise ValueError(f"Unsupported operation: {operation.op}")

    def apply(self, operation: MemoryOperation) -> AppliedOperation:
        now = utc_now()
        if operation.op == "ADD":
            assert operation.node is not None
            node = operation.node
            if operation.source_refs:
                node.source_refs = sorted(set(node.source_refs) | set(operation.source_refs))
            node.updated_at = now
            self.store.write(node)
            return AppliedOperation(operation, [node.id], f"added {node.id}")

        assert operation.node_id is not None
        node = self._require_node(operation.node_id)

        if operation.op == "UPDATE":
            self._apply_updates(node, operation.updates)
            node.updated_at = now
            self.store.write(node)
            return AppliedOperation(operation, [node.id], f"updated {node.id}")

        if operation.op == "INVALIDATE":
            node.status = "deprecated"
            node.valid_to = now
            node.updated_at = now
            node.body = f"{node.body}\n\nInvalidated: {operation.reason}".strip()
            self.store.write(node)
            return AppliedOperation(operation, [node.id], f"invalidated {node.id}")

        if operation.op == "LINK":
            assert operation.target_id is not None
            relation = Relation(
                target_id=operation.target_id,
                type=operation.relation_type or "related_to",
                weight=operation.relation_weight,
            )
            node.relations = _replace_relation(node.relations, relation)
            node.updated_at = now
            self.store.write(node)
            return AppliedOperation(operation, [node.id], f"linked {node.id} to {operation.target_id}")

        if operation.op == "MERGE":
            assert operation.target_id is not None
            target = self._require_node(operation.target_id)
            target.source_refs = sorted(set(target.source_refs) | set(node.source_refs) | set(operation.source_refs))
            target.entity_refs = sorted(set(target.entity_refs) | set(node.entity_refs))
            target.concept_refs = sorted(set(target.concept_refs) | set(node.concept_refs))
            target.relations = _merge_relations(target.relations, node.relations)
            target.salience = max(target.salience, node.salience)
            target.confidence = max(target.confidence, min(0.98, node.confidence))
            target.updated_at = now
            node.status = "deprecated"
            node.valid_to = now
            node.updated_at = now
            node.relations = _replace_relation(
                node.relations,
                Relation(target_id=target.id, type="merged_into", weight=1.0),
            )
            node.body = f"{node.body}\n\nMerged into {target.id}: {operation.reason}".strip()
            self.store.write(target)
            self.store.write(node)
            if self.index is not None:
                self.index.record_alias(node.id, target.id)
            return AppliedOperation(operation, [target.id, node.id], f"merged {node.id} into {target.id}")

        if operation.op == "PROMOTE":
            node.status = PROMOTION.get(node.status, node.status)
            node.salience = min(1.0, round(node.salience + 0.08, 4))
            node.updated_at = now
            self.store.write(node)
            return AppliedOperation(operation, [node.id], f"promoted {node.id} to {node.status}")

        if operation.op == "DEMOTE":
            node.status = DEMOTION.get(node.status, node.status)
            node.salience = max(0.1, round(node.salience - 0.08, 4))
            node.updated_at = now
            self.store.write(node)
            return AppliedOperation(operation, [node.id], f"demoted {node.id} to {node.status}")

        raise ValueError(f"Unsupported operation: {operation.op}")

    def _require_node(self, node_id: str) -> MemoryNode:
        node = self.store.get(node_id)
        if node is None:
            raise ValueError(f"Unknown memory node: {node_id}")
        return node

    def _apply_updates(self, node: MemoryNode, updates: dict[str, Any]) -> None:
        for key, value in updates.items():
            if key == "status":
                if value not in UPDATABLE_STATUSES:
                    raise ValueError(f"UPDATE may not set status to {value!r}; use INVALIDATE")
                node.status = value
            elif key in {"title", "body"}:
                setattr(node, key, str(value))
            elif key in {"valid_from", "valid_to"}:
                if value is None:
                    setattr(node, key, None)
                else:
                    parse_time(str(value))
                    setattr(node, key, str(value))
            elif key in {"confidence", "salience"}:
                numeric = float(value)
                if not 0.0 <= numeric <= 1.0:
                    raise ValueError(f"UPDATE {key} must be between 0 and 1")
                setattr(node, key, numeric)
            elif key == "pinned":
                node.pinned = bool(value)
            elif key in {"source_refs", "entity_refs", "concept_refs"}:
                setattr(node, key, [str(item) for item in value])
            elif key == "archived":
                node.archived = bool(value)
            else:
                raise ValueError(f"Unsupported update field: {key}")


def _replace_relation(relations: list[Relation], relation: Relation) -> list[Relation]:
    kept = [
        item
        for item in relations
        if not (item.target_id == relation.target_id and item.type == relation.type)
    ]
    return [*kept, relation]


def _merge_relations(left: list[Relation], right: list[Relation]) -> list[Relation]:
    merged = list(left)
    for relation in right:
        existing = next(
            (item for item in merged if item.target_id == relation.target_id and item.type == relation.type),
            None,
        )
        if existing is None:
            merged.append(relation)
        else:
            existing.weight = max(existing.weight, relation.weight)
    return merged
