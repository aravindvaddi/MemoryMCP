"""Memory store implementation with SQLite backend and semantic search capabilities."""

import json
import sqlite3
import struct
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class MemoryStore:
    """SQLite-based memory store with semantic search capabilities"""

    def __init__(self, agent_id: str, base_dir: Path):
        self.agent_id = agent_id
        self.db_path = base_dir / agent_id / "memories.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize the database schema matching MEMORY_DESIGN.md"""
        with closing(sqlite3.connect(self.db_path)) as conn:
            # Check if we need to migrate from old schema
            cursor = conn.execute("PRAGMA table_info(memories)")
            columns = {row[1] for row in cursor.fetchall()}

            if columns and "embedding" not in columns:
                # Migration needed - rename old table
                conn.execute("ALTER TABLE memories RENAME TO memories_old")

            # Create new schema
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    strength REAL DEFAULT 1.0,
                    access_count INTEGER DEFAULT 0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    last_accessed DATETIME DEFAULT CURRENT_TIMESTAMP,
                    context JSON,
                    metadata JSON
                )
            """)

            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_agent_id ON memories(agent_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON memories(created_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_strength ON memories(strength)")

            # Migrate old data if exists
            if "memories_old" in [row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]:
                # Note: Old memories won't have embeddings, so they'll be less useful
                conn.execute(
                    """
                    INSERT INTO memories (agent_id, content, embedding, created_at, metadata)
                    SELECT ?, content, X'00', timestamp, metadata
                    FROM memories_old
                """,
                    (self.agent_id,),
                )
                conn.execute("DROP TABLE memories_old")

            conn.commit()

    async def add_memory(
        self,
        content: str,
        embedding: np.ndarray,
        context: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Add a new memory with embedding to the store"""
        # Convert numpy array to binary blob
        embedding_blob = struct.pack("f" * 384, *embedding.tolist())

        with closing(sqlite3.connect(self.db_path)) as conn:
            cursor = conn.execute(
                """
                INSERT INTO memories (agent_id, content, embedding, strength, context, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    self.agent_id,
                    content,
                    embedding_blob,
                    metadata.get("importance", 1.0) if metadata else 1.0,
                    json.dumps(context or {}),
                    json.dumps(metadata or {}),
                ),
            )
            conn.commit()
            return cursor.lastrowid

    async def search_memories(
        self, query_embedding: np.ndarray, limit: int = 10, recency_weight: float = 0.3
    ) -> list[dict[str, Any]]:
        """Search memories using semantic similarity with relevance scoring"""
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row

            # Get all memories with embeddings
            cursor = conn.execute(
                """
                SELECT id, content, embedding, strength, access_count,
                       created_at, last_accessed, context, metadata
                FROM memories
                WHERE agent_id = ?
                ORDER BY created_at DESC
                LIMIT 1000
            """,
                (self.agent_id,),
            )

            memories = []
            current_time = datetime.utcnow()

            for row in cursor:
                # Deserialize embedding
                embedding_blob = row["embedding"]
                if len(embedding_blob) == 1:  # Skip placeholder embeddings
                    continue

                memory_embedding = np.array(struct.unpack("f" * 384, embedding_blob))

                # Calculate semantic similarity
                semantic_similarity = cosine_similarity(
                    query_embedding.reshape(1, -1), memory_embedding.reshape(1, -1)
                )[0][0]

                # Calculate recency score (exponential decay)
                created_at = datetime.fromisoformat(row["created_at"])
                days_old = (current_time - created_at).days
                recency_score = np.exp(-days_old / 30)  # 50% decay after 30 days

                # Calculate final score
                final_score = (semantic_similarity * (1 - recency_weight) + recency_score * recency_weight) * row[
                    "strength"
                ]

                memories.append(
                    {
                        "id": row["id"],
                        "content": row["content"],
                        "score": final_score,
                        "semantic_similarity": semantic_similarity,
                        "recency_score": recency_score,
                        "strength": row["strength"],
                        "access_count": row["access_count"],
                        "created_at": row["created_at"],
                        "context": json.loads(row["context"] or "{}"),
                        "metadata": json.loads(row["metadata"] or "{}"),
                    }
                )

                # Update access count and last accessed
                conn.execute(
                    """
                    UPDATE memories
                    SET access_count = access_count + 1,
                        last_accessed = CURRENT_TIMESTAMP,
                        strength = MIN(strength * 1.1, 1.0)
                    WHERE id = ?
                """,
                    (row["id"],),
                )

            conn.commit()

            # Sort by score and return top results
            memories.sort(key=lambda x: x["score"], reverse=True)
            return memories[:limit]

    def get_recent_memories(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get the most recent memories"""
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT id, content, strength, access_count,
                       created_at, last_accessed, context, metadata
                FROM memories
                WHERE agent_id = ?
                ORDER BY created_at DESC
                LIMIT ?
            """,
                (self.agent_id, limit),
            )

            return [dict(row) for row in cursor.fetchall()]

