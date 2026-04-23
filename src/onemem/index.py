from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .embeddings import cosine, embed
from .markdown_store import MarkdownStore
from .models import MemoryNode


class SidecarIndex:
    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)

    def connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS nodes (
                    id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    title TEXT NOT NULL,
                    body TEXT NOT NULL,
                    status TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    salience REAL NOT NULL,
                    pinned INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    valid_from TEXT,
                    valid_to TEXT,
                    source_refs TEXT NOT NULL,
                    concept_refs TEXT NOT NULL,
                    archived INTEGER NOT NULL,
                    embedding TEXT NOT NULL
                );
                CREATE VIRTUAL TABLE IF NOT EXISTS fts_nodes
                USING fts5(id UNINDEXED, title, body, kind UNINDEXED);
                CREATE TABLE IF NOT EXISTS edges (
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    type TEXT NOT NULL,
                    weight REAL NOT NULL,
                    PRIMARY KEY (source_id, target_id, type)
                );
                CREATE TABLE IF NOT EXISTS aliases (
                    alias TEXT PRIMARY KEY,
                    node_id TEXT NOT NULL
                );
                """
            )

    def rebuild(self, store: MarkdownStore) -> None:
        self.init()
        nodes = store.all_nodes(include_archive=True)
        with self.connect() as conn:
            conn.execute("DELETE FROM nodes")
            conn.execute("DELETE FROM fts_nodes")
            conn.execute("DELETE FROM edges")
            conn.execute("DELETE FROM aliases")
            for node in nodes:
                self.upsert_node(conn, node)

    def upsert(self, node: MemoryNode) -> None:
        self.init()
        with self.connect() as conn:
            self.upsert_node(conn, node)

    def upsert_node(self, conn: sqlite3.Connection, node: MemoryNode) -> None:
        body_for_embedding = f"{node.title}\n{node.body}\n{' '.join(node.concept_refs)}"
        conn.execute(
            """
            INSERT OR REPLACE INTO nodes
            (id, kind, title, body, status, confidence, salience, pinned, created_at,
             updated_at, valid_from, valid_to, source_refs, concept_refs, archived, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                node.id,
                node.kind,
                node.title,
                node.body,
                node.status,
                node.confidence,
                node.salience,
                int(node.pinned),
                node.created_at,
                node.updated_at,
                node.valid_from,
                node.valid_to,
                json.dumps(node.source_refs),
                json.dumps(node.concept_refs),
                int(node.archived),
                json.dumps(embed(body_for_embedding)),
            ),
        )
        conn.execute("DELETE FROM fts_nodes WHERE id = ?", (node.id,))
        conn.execute(
            "INSERT INTO fts_nodes (id, title, body, kind) VALUES (?, ?, ?, ?)",
            (node.id, node.title, node.body, node.kind),
        )
        conn.execute("DELETE FROM edges WHERE source_id = ?", (node.id,))
        for relation in node.relations:
            conn.execute(
                "INSERT OR REPLACE INTO edges (source_id, target_id, type, weight) VALUES (?, ?, ?, ?)",
                (node.id, relation.target_id, relation.type, relation.weight),
            )
        for concept in node.concept_refs:
            conn.execute(
                "INSERT OR IGNORE INTO aliases (alias, node_id) VALUES (?, ?)",
                (concept.lower(), node.id),
            )

    def fts_search(self, query: str, limit: int = 20) -> list[sqlite3.Row]:
        self.init()
        with self.connect() as conn:
            try:
                return list(
                    conn.execute(
                        """
                        SELECT n.*, bm25(fts_nodes) AS lexical_score
                        FROM fts_nodes
                        JOIN nodes n ON n.id = fts_nodes.id
                        WHERE fts_nodes MATCH ?
                          AND n.archived = 0
                        ORDER BY lexical_score
                        LIMIT ?
                        """,
                        (query, limit),
                    )
                )
            except sqlite3.OperationalError:
                return []

    def vector_search(self, query: str, limit: int = 20) -> list[tuple[sqlite3.Row, float]]:
        self.init()
        query_vector = embed(query)
        scored: list[tuple[sqlite3.Row, float]] = []
        with self.connect() as conn:
            for row in conn.execute("SELECT * FROM nodes WHERE archived = 0"):
                score = cosine(query_vector, json.loads(row["embedding"]))
                if score > 0:
                    scored.append((row, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:limit]

    def graph_neighbors(self, node_ids: list[str], hops: int = 1) -> set[str]:
        self.init()
        seen = set(node_ids)
        frontier = set(node_ids)
        with self.connect() as conn:
            for _ in range(hops):
                if not frontier:
                    break
                placeholders = ",".join("?" for _ in frontier)
                rows = conn.execute(
                    f"""
                    SELECT source_id, target_id FROM edges
                    WHERE source_id IN ({placeholders}) OR target_id IN ({placeholders})
                    """,
                    tuple(frontier) + tuple(frontier),
                )
                next_frontier: set[str] = set()
                for row in rows:
                    next_frontier.add(row["source_id"])
                    next_frontier.add(row["target_id"])
                next_frontier -= seen
                seen |= next_frontier
                frontier = next_frontier
        return seen - set(node_ids)

    def get_rows(self, node_ids: set[str]) -> list[sqlite3.Row]:
        if not node_ids:
            return []
        self.init()
        placeholders = ",".join("?" for _ in node_ids)
        with self.connect() as conn:
            return list(conn.execute(f"SELECT * FROM nodes WHERE id IN ({placeholders})", tuple(node_ids)))

