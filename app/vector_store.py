"""
Vector store abstraction.

The RAG engine talks to a small collection interface so it can run on:

* **ChromaDB** -- embedded, on-disk (default, local development)
* **pgvector** -- PostgreSQL extension (production / AWS RDS / Azure PG)

Both implementations expose the same dict-shaped API that the engine already
uses: ``get`` / ``query`` / ``add`` / ``upsert`` / ``delete`` / ``count``.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Dict, List, Optional, Sequence

logger = logging.getLogger("retriva.vectorstore")

_SAFE_KEY = re.compile(r"^[A-Za-z0-9_]+$")


def _vector_literal(vector: Sequence[float]) -> str:
    return "[" + ",".join(f"{float(x):.7g}" for x in vector) + "]"


def _where_sql(where: Optional[dict], start: int = 0):
    """Translate a Chroma-style metadata filter into SQL + bound params."""
    params: Dict[str, object] = {}
    counter = [start]

    def build(node) -> str:
        if not node:
            return ""
        if "$and" in node:
            return "(" + " AND ".join(build(x) for x in node["$and"]) + ")"
        if "$or" in node:
            return "(" + " OR ".join(build(x) for x in node["$or"]) + ")"
        parts = []
        for key, value in node.items():
            if not _SAFE_KEY.match(str(key)):
                raise ValueError(f"Unsafe metadata key: {key!r}")
            name = f"p{counter[0]}"
            counter[0] += 1
            params[name] = str(value)
            parts.append(f"metadata->>'{key}' = :{name}")
        return "(" + " AND ".join(parts) + ")" if parts else ""

    return build(where), params


# --------------------------------------------------------------------------- #
# ChromaDB (embedded)
# --------------------------------------------------------------------------- #
class ChromaVectorCollection:
    def __init__(self, collection):
        self._collection = collection

    @property
    def raw(self):
        return self._collection

    def count(self) -> int:
        return self._collection.count()

    def get(self, where=None, include=None) -> Dict:
        data = self._collection.get(
            where=where, include=include or ["documents", "metadatas"]
        )
        return {
            "ids": data.get("ids") or [],
            "documents": data.get("documents") or [],
            "metadatas": data.get("metadatas") or [],
        }

    def query(self, query_embeddings, n_results, where=None) -> Dict:
        return self._collection.query(
            query_embeddings=query_embeddings, n_results=n_results, where=where
        )

    def add(self, ids, documents, metadatas, embeddings=None) -> None:
        kwargs = {"ids": ids, "documents": documents, "metadatas": metadatas}
        if embeddings is not None:
            kwargs["embeddings"] = embeddings
        self._collection.add(**kwargs)

    def upsert(self, ids, documents, metadatas, embeddings=None) -> None:
        kwargs = {"ids": ids, "documents": documents, "metadatas": metadatas}
        if embeddings is not None:
            kwargs["embeddings"] = embeddings
        self._collection.upsert(**kwargs)

    def delete(self, ids) -> None:
        if ids:
            self._collection.delete(ids=list(ids))

    def clear(self) -> None:
        existing = self.get()
        if existing["ids"]:
            self.delete(existing["ids"])


# --------------------------------------------------------------------------- #
# pgvector (PostgreSQL)
# --------------------------------------------------------------------------- #
class PgVectorCollection:
    def __init__(self, dsn: str, table: str, dim: int):
        from sqlalchemy import create_engine

        from app.database import normalize_database_url

        self.table = table
        self.dim = dim
        self.engine = create_engine(
            normalize_database_url(dsn), pool_pre_ping=True, future=True
        )
        self._ensure_schema()

    # -- schema ------------------------------------------------------------
    def _ensure_schema(self) -> None:
        with self.engine.begin() as conn:
            conn.exec_driver_sql("CREATE EXTENSION IF NOT EXISTS vector")
            conn.exec_driver_sql(
                f'CREATE TABLE IF NOT EXISTS "{self.table}" ('
                "id TEXT PRIMARY KEY, "
                f"embedding vector({self.dim}) NOT NULL, "
                "document TEXT, "
                "metadata JSONB NOT NULL DEFAULT '{}'::jsonb)"
            )
            try:
                conn.exec_driver_sql(
                    f'CREATE INDEX IF NOT EXISTS "{self.table}_hnsw" '
                    f'ON "{self.table}" USING hnsw (embedding vector_cosine_ops)'
                )
            except Exception as exc:  # noqa: BLE001 - HNSW is optional
                logger.warning("HNSW index not created for %s: %s", self.table, exc)

    # -- read --------------------------------------------------------------
    def count(self) -> int:
        from sqlalchemy import text

        with self.engine.begin() as conn:
            return int(
                conn.execute(text(f'SELECT count(*) FROM "{self.table}"')).scalar() or 0
            )

    def get(self, where=None, include=None) -> Dict:
        from sqlalchemy import text

        clause, params = _where_sql(where)
        sql = f'SELECT id, document, metadata FROM "{self.table}"'
        if clause:
            sql += f" WHERE {clause}"
        with self.engine.begin() as conn:
            rows = conn.execute(text(sql), params).fetchall()
        return {
            "ids": [r[0] for r in rows],
            "documents": [r[1] for r in rows],
            "metadatas": [r[2] for r in rows],
        }

    def query(self, query_embeddings, n_results, where=None) -> Dict:
        from sqlalchemy import text

        if not query_embeddings:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
        clause, params = _where_sql(where, start=1)
        params["emb"] = _vector_literal(query_embeddings[0])
        params["lim"] = int(n_results)
        sql = (
            f'SELECT id, document, metadata, embedding <=> CAST(:emb AS vector) AS distance '
            f'FROM "{self.table}"'
        )
        if clause:
            sql += f" WHERE {clause}"
        sql += " ORDER BY distance ASC LIMIT :lim"
        with self.engine.begin() as conn:
            rows = conn.execute(text(sql), params).fetchall()
        return {
            "ids": [[r[0] for r in rows]],
            "documents": [[r[1] for r in rows]],
            "metadatas": [[r[2] for r in rows]],
            "distances": [[float(r[3]) for r in rows]],
        }

    # -- write -------------------------------------------------------------
    def add(self, ids, documents, metadatas, embeddings=None) -> None:
        if embeddings is None:
            raise ValueError("pgvector backend requires embeddings on add()")
        self.upsert(ids, documents, metadatas, embeddings)

    def upsert(self, ids, documents, metadatas, embeddings=None) -> None:
        from sqlalchemy import text

        if embeddings is None:
            raise ValueError("pgvector backend requires embeddings on upsert()")
        sql = (
            f'INSERT INTO "{self.table}" (id, embedding, document, metadata) '
            "VALUES (:id, CAST(:emb AS vector), :doc, CAST(:meta AS jsonb)) "
            "ON CONFLICT (id) DO UPDATE SET "
            "embedding = EXCLUDED.embedding, "
            "document = EXCLUDED.document, "
            "metadata = EXCLUDED.metadata"
        )
        with self.engine.begin() as conn:
            for i, doc_id in enumerate(ids):
                conn.execute(
                    text(sql),
                    {
                        "id": str(doc_id),
                        "emb": _vector_literal(embeddings[i]),
                        "doc": documents[i],
                        "meta": json.dumps(metadatas[i] or {}),
                    },
                )

    def delete(self, ids) -> None:
        from sqlalchemy import bindparam, text

        ids = list(ids or [])
        if not ids:
            return
        with self.engine.begin() as conn:
            conn.execute(
                text(f'DELETE FROM "{self.table}" WHERE id IN :ids').bindparams(
                    bindparam("ids", expanding=True)
                ),
                {"ids": [str(i) for i in ids]},
            )

    def clear(self) -> None:
        with self.engine.begin() as conn:
            conn.exec_driver_sql(f'TRUNCATE TABLE "{self.table}"')


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def build_collection(config, name: str, dim: int):
    """Return a vector collection for the configured backend."""
    backend = (config.VECTOR_BACKEND or "chroma").strip().lower()
    if backend == "pgvector":
        dsn = config.PGVECTOR_URL or config.DATABASE_URL
        logger.info("Using pgvector backend (table=%s, dim=%d)", name, dim)
        return PgVectorCollection(dsn, name, dim)
    if backend == "chroma":
        import chromadb

        client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)
        return ChromaVectorCollection(client.get_or_create_collection(name=name))
    raise ValueError(f"Unknown VECTOR_BACKEND: {config.VECTOR_BACKEND!r}")
