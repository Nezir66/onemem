from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .models import MemoryNode
from .text import slugify


class MarkdownStore:
    def __init__(self, root: Path | str = "memory") -> None:
        self.root = Path(root)

    def ensure(self) -> None:
        for relative in [
            "episodes",
            "facts",
            "concepts",
            "summaries",
            "archive",
            ".sidecar",
        ]:
            (self.root / relative).mkdir(parents=True, exist_ok=True)

    def write(self, node: MemoryNode) -> Path:
        self.ensure()
        target = self.path_for(node)
        target.parent.mkdir(parents=True, exist_ok=True)
        archive_root = self.root / "archive"
        for existing in self.root.rglob(f"{node.id}.md"):
            if existing == target:
                continue
            try:
                existing.relative_to(archive_root)
            except ValueError:
                existing.unlink()
        target.write_text(serialize_node(node), encoding="utf-8")
        return target

    def archive(self, node: MemoryNode) -> Path:
        current = self.find_path(node.id)
        filename = current.name if current is not None else f"{node.id}.md"
        target = self.root / "archive" / node.kind / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(serialize_node(node), encoding="utf-8")
        if current is not None and current != target:
            current.unlink()
        return target

    def read(self, path: Path) -> MemoryNode:
        metadata, body = parse_node(path.read_text(encoding="utf-8"))
        return MemoryNode.from_parts(metadata, body)

    def all_nodes(self, include_archive: bool = False) -> list[MemoryNode]:
        self.ensure()
        roots = [self.root / name for name in ["episodes", "facts", "concepts", "summaries"]]
        if include_archive:
            roots.append(self.root / "archive")
        nodes: list[MemoryNode] = []
        for base in roots:
            for path in sorted(base.rglob("*.md")):
                nodes.append(self.read(path))
        return nodes

    def nodes_by_kind(self, kind: str) -> list[MemoryNode]:
        return [node for node in self.all_nodes() if node.kind == kind]

    def get(self, node_id: str) -> MemoryNode | None:
        path = self.find_path(node_id)
        return self.read(path) if path else None

    def find_path(self, node_id: str) -> Path | None:
        self.ensure()
        for path in self.root.rglob("*.md"):
            try:
                metadata, _ = parse_node(path.read_text(encoding="utf-8"))
            except ValueError:
                continue
            if metadata.get("id") == node_id:
                return path
        return None

    def path_for(self, node: MemoryNode) -> Path:
        filename = f"{node.id}.md"
        if node.kind == "episode":
            date = node.created_at[:7].replace("-", "/")
            return self.root / "episodes" / date / filename
        if node.kind == "fact":
            concept = slugify(node.concept_refs[0] if node.concept_refs else "uncategorized")
            return self.root / "facts" / concept / filename
        if node.kind == "concept":
            return self.root / "concepts" / filename
        if node.kind == "summary":
            return self.root / "summaries" / filename
        raise ValueError(f"Unsupported node kind: {node.kind}")


def serialize_node(node: MemoryNode) -> str:
    lines = ["---"]
    for key, value in node.metadata().items():
        lines.append(f"{key}: {json.dumps(value, ensure_ascii=True)}")
    lines.extend(["---", "", node.body.strip(), ""])
    return "\n".join(lines)


def parse_node(raw: str) -> tuple[dict, str]:
    if not raw.startswith("---\n"):
        raise ValueError("Memory file is missing frontmatter")
    try:
        _, frontmatter, body = raw.split("---\n", 2)
    except ValueError as exc:
        raise ValueError("Memory file has invalid frontmatter") from exc
    metadata = {}
    for line in frontmatter.splitlines():
        if not line.strip():
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = json.loads(value.strip())
    return metadata, body.strip()


def write_many(store: MarkdownStore, nodes: Iterable[MemoryNode]) -> list[Path]:
    return [store.write(node) for node in nodes]

