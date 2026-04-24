from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

from .embedding_providers import EmbeddingProvider, HashEmbeddingProvider
from .embeddings import cosine
from .markdown_store import MarkdownStore
from .models import MemoryNode


class SidecarIndex:
    def __init__(self, db_path: Path | str, embedding_provider: EmbeddingProvider | None = None) -> None:
        self.db_path = Path(db_path)
        self.embedding_provider = embedding_provider or HashEmbeddingProvider()

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
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS embedding_cache (
                    cache_key TEXT PRIMARY KEY,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    dimensions INTEGER NOT NULL,
                    text_hash TEXT NOT NULL,
                    embedding TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS merge_candidates (
                    id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    score REAL NOT NULL,
                    reason TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'open'
                );
                """
            )
            conn.execute(
                "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
                ("embedding_provider", self.embedding_provider.name),
            )
            conn.execute(
                "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
                ("embedding_model", self.embedding_provider.model),
            )
            conn.execute(
                "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
                ("embedding_dimensions", str(self.embedding_provider.dimensions)),
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
                json.dumps(self._embedding_for_text(conn, body_for_embedding)),
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
        if node.status == "deprecated":
            for relation in node.relations:
                if relation.type == "merged_into":
                    conn.execute(
                        "INSERT OR REPLACE INTO aliases (alias, node_id) VALUES (?, ?)",
                        (node.id, relation.target_id),
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
        scored: list[tuple[sqlite3.Row, float]] = []
        with self.connect() as conn:
            query_vector = self._embedding_for_text(conn, query)
            for row in conn.execute("SELECT * FROM nodes WHERE archived = 0"):
                score = cosine(query_vector, json.loads(row["embedding"]))
                if score > 0:
                    scored.append((row, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:limit]

    def temporal_candidates(
        self,
        *,
        prefer_earliest: bool = False,
        prefer_latest: bool = False,
        before: str | None = None,
        after: str | None = None,
        limit: int = 10,
    ) -> list[sqlite3.Row]:
        self.init()
        clauses = ["archived = 0", "valid_from IS NOT NULL"]
        params: list[object] = []
        if before is not None:
            clauses.append("valid_from <= ?")
            params.append(before)
        if after is not None:
            clauses.append("valid_from >= ?")
            params.append(after)
        if prefer_latest or before is not None:
            order = "DESC"
        elif prefer_earliest or after is not None:
            order = "ASC"
        else:
            order = "DESC"
        params.append(int(limit))
        with self.connect() as conn:
            return list(
                conn.execute(
                    f"SELECT * FROM nodes WHERE {' AND '.join(clauses)} ORDER BY valid_from {order} LIMIT ?",
                    tuple(params),
                )
            )

    def graph_neighbors(self, node_ids: list[str], hops: int = 1) -> set[str]:
        return set(self.graph_neighbor_edges(node_ids, hops=hops).keys())

    def graph_neighbor_edges(
        self, node_ids: list[str], hops: int = 1
    ) -> dict[str, list[tuple[str, float]]]:
        """Return reachable neighbors with the (type, weight) of the best edge found."""
        self.init()
        seen = set(node_ids)
        frontier = set(node_ids)
        edges_for: dict[str, list[tuple[str, float]]] = {}
        with self.connect() as conn:
            for _ in range(hops):
                if not frontier:
                    break
                placeholders = ",".join("?" for _ in frontier)
                rows = conn.execute(
                    f"""
                    SELECT source_id, target_id, type, weight FROM edges
                    WHERE source_id IN ({placeholders}) OR target_id IN ({placeholders})
                    """,
                    tuple(frontier) + tuple(frontier),
                )
                next_frontier: set[str] = set()
                for row in rows:
                    for node_id in (row["source_id"], row["target_id"]):
                        if node_id in seen:
                            continue
                        next_frontier.add(node_id)
                        edges_for.setdefault(node_id, []).append(
                            (str(row["type"]), float(row["weight"]))
                        )
                next_frontier -= seen
                next_frontier = self._limit_hubs(conn, next_frontier, max_degree=24)
                seen |= next_frontier
                frontier = next_frontier
        return {node_id: edges_for.get(node_id, []) for node_id in seen - set(node_ids)}

    def graph_neighbor_details(self, node_id: str, *, limit: int = 20) -> list[dict[str, object]]:
        self.init()
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT e.source_id, e.target_id, e.type, e.weight,
                       n.id, n.kind, n.title, n.status, n.salience
                FROM edges e
                JOIN nodes n ON n.id = CASE
                    WHEN e.source_id = ? THEN e.target_id
                    ELSE e.source_id
                END
                WHERE (e.source_id = ? OR e.target_id = ?)
                  AND n.archived = 0
                ORDER BY e.weight DESC, n.salience DESC
                LIMIT ?
                """,
                (node_id, node_id, node_id, limit),
            )
            return [
                {
                    "id": row["id"],
                    "kind": row["kind"],
                    "title": row["title"],
                    "status": row["status"],
                    "relation": row["type"],
                    "weight": row["weight"],
                    "direction": "out" if row["source_id"] == node_id else "in",
                }
                for row in rows
            ]

    def _limit_hubs(self, conn: sqlite3.Connection, node_ids: set[str], *, max_degree: int) -> set[str]:
        if not node_ids:
            return node_ids
        kept: set[str] = set()
        for node_id in node_ids:
            degree = conn.execute(
                "SELECT COUNT(*) AS degree FROM edges WHERE source_id = ? OR target_id = ?",
                (node_id, node_id),
            ).fetchone()["degree"]
            if int(degree) <= max_degree:
                kept.add(node_id)
        return kept

    def get_rows(self, node_ids: set[str]) -> list[sqlite3.Row]:
        if not node_ids:
            return []
        self.init()
        placeholders = ",".join("?" for _ in node_ids)
        with self.connect() as conn:
            return list(conn.execute(f"SELECT * FROM nodes WHERE id IN ({placeholders})", tuple(node_ids)))

    def record_alias(self, alias: str, node_id: str) -> None:
        self.init()
        with self.connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO aliases (alias, node_id) VALUES (?, ?)",
                (alias, node_id),
            )

    def resolve_alias(self, alias: str) -> str | None:
        self.init()
        with self.connect() as conn:
            row = conn.execute("SELECT node_id FROM aliases WHERE alias = ?", (alias,)).fetchone()
            return row["node_id"] if row else None

    def save_merge_candidate(
        self,
        *,
        candidate_id: str,
        source_id: str,
        target_id: str,
        score: float,
        reason: str,
        created_at: str,
    ) -> None:
        self.init()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO merge_candidates (id, source_id, target_id, score, reason, created_at, status)
                VALUES (?, ?, ?, ?, ?, ?, 'open')
                ON CONFLICT(id) DO UPDATE SET
                    source_id=excluded.source_id,
                    target_id=excluded.target_id,
                    score=excluded.score,
                    reason=excluded.reason
                """,
                (candidate_id, source_id, target_id, score, reason, created_at),
            )

    def load_merge_candidate(self, candidate_id: str) -> sqlite3.Row | None:
        self.init()
        with self.connect() as conn:
            return conn.execute(
                "SELECT * FROM merge_candidates WHERE id = ?", (candidate_id,)
            ).fetchone()

    def list_merge_candidates(self, *, status: str = "open", limit: int = 25) -> list[sqlite3.Row]:
        self.init()
        with self.connect() as conn:
            return list(
                conn.execute(
                    "SELECT * FROM merge_candidates WHERE status = ? ORDER BY score DESC LIMIT ?",
                    (status, int(limit)),
                )
            )

    def mark_merge_candidate(self, candidate_id: str, status: str) -> None:
        self.init()
        with self.connect() as conn:
            conn.execute(
                "UPDATE merge_candidates SET status = ? WHERE id = ?",
                (status, candidate_id),
            )

    def embeddings_for(self, node_ids: list[str]) -> dict[str, list[float]]:
        if not node_ids:
            return {}
        self.init()
        placeholders = ",".join("?" for _ in node_ids)
        with self.connect() as conn:
            rows = conn.execute(
                f"SELECT id, embedding FROM nodes WHERE id IN ({placeholders})",
                tuple(node_ids),
            )
            return {row["id"]: json.loads(row["embedding"]) for row in rows}

    def _embedding_for_text(self, conn: sqlite3.Connection, text: str) -> list[float]:
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        cache_key = ":".join(
            [
                self.embedding_provider.name,
                self.embedding_provider.model,
                str(self.embedding_provider.dimensions),
                text_hash,
            ]
        )
        cached = conn.execute("SELECT embedding FROM embedding_cache WHERE cache_key = ?", (cache_key,)).fetchone()
        if cached:
            return json.loads(cached["embedding"])
        vector = self.embedding_provider.embed(text)
        conn.execute(
            """
            INSERT OR REPLACE INTO embedding_cache
            (cache_key, provider, model, dimensions, text_hash, embedding)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                cache_key,
                self.embedding_provider.name,
                self.embedding_provider.model,
                self.embedding_provider.dimensions,
                text_hash,
                json.dumps(vector),
            ),
        )
        return vector
