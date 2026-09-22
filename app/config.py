"""
Retriva configuration.

Everything here is intentionally local-first: models are resolved from the
HuggingFace cache (``local_files_only``) and no network/API keys are required.
All values can be overridden through environment variables so the deployment
can be tuned without touching code.
"""

import os


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


class Settings:
    # --- Storage -----------------------------------------------------------
    CHROMA_DB_PATH: str = os.getenv("CHROMA_DB_PATH", "./retriva_chroma_db")
    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "retriva_financial_docs")
    MEMORY_COLLECTION_NAME: str = os.getenv(
        "MEMORY_COLLECTION_NAME", "retriva_chat_memory"
    )

    # --- Models (all resolved locally) ------------------------------------
    EMBEDDING_MODEL_NAME: str = os.getenv(
        "EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2"
    )
    RERANKER_MODEL_NAME: str = os.getenv(
        "RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    LLM_MODEL_NAME: str = os.getenv(
        "LLM_MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct"
    )
    # Never hit the network: the models are expected to live in the local cache.
    HF_LOCAL_ONLY: bool = _env_bool("HF_LOCAL_ONLY", True)

    # --- Retrieval ---------------------------------------------------------
    TOP_N_RETRIEVAL: int = _env_int("TOP_N_RETRIEVAL", 12)
    TOP_N_RERANKED: int = _env_int("TOP_N_RERANKED", 6)
    RRF_K: int = _env_int("RRF_K", 60)
    MAX_CONTEXT_CHUNKS: int = _env_int("MAX_CONTEXT_CHUNKS", 6)
    MAX_CHUNK_CHARS: int = _env_int("MAX_CHUNK_CHARS", 1800)

    # --- Generation --------------------------------------------------------
    MAX_NEW_TOKENS: int = _env_int("MAX_NEW_TOKENS", 384)
    TEMPERATURE: float = _env_float("TEMPERATURE", 0.1)
    TOP_P: float = _env_float("TOP_P", 0.9)
    TOP_K: int = _env_int("TOP_K", 50)

    # --- Agent -------------------------------------------------------------
    # Below this reranker score (ms-marco logits) we consider the retrieval
    # unreliable and ask the user a clarifying question instead of guessing.
    CLARIFICATION_THRESHOLD: float = _env_float("CLARIFICATION_THRESHOLD", -2.0)
    # When a query is not obviously about finance/documents we still probe the
    # knowledge base; above this score we ground the answer in the documents
    # (this is how uploaded files are found without keyword hints).
    RAG_CONFIDENCE_THRESHOLD: float = _env_float("RAG_CONFIDENCE_THRESHOLD", 0.5)
    # Uploaded documents are often summarised rather than keyword-queried, so
    # the cross-encoder scores them low. We still answer from an uploaded doc
    # as long as its score is above this (lower) bar.
    UPLOAD_CLARIFICATION_THRESHOLD: float = _env_float(
        "UPLOAD_CLARIFICATION_THRESHOLD", -8.0
    )
    # Optional LLM based planner/query-rewriter. Off by default because on CPU
    # it doubles latency; the deterministic planner is always active.
    AGENT_LLM_PLANNING: bool = _env_bool("AGENT_LLM_PLANNING", False)

    # --- Memory (cross-chat) ----------------------------------------------
    ENABLE_MEMORY: bool = _env_bool("ENABLE_MEMORY", True)
    MEMORY_TOP_K: int = _env_int("MEMORY_TOP_K", 4)
    MEMORY_MAX_DISTANCE: float = _env_float("MEMORY_MAX_DISTANCE", 1.35)
    MAX_HISTORY_TURNS: int = _env_int("MAX_HISTORY_TURNS", 6)

    # --- Multi tenancy -----------------------------------------------------
    # The knowledge base (finance filings + user uploads) is shared so every
    # user can query every document. Conversation memory however is ALWAYS
    # scoped to a single chat; the only cross-chat knowledge is the user's bio.
    SHARE_KNOWLEDGE_ACROSS_USERS: bool = _env_bool("SHARE_KNOWLEDGE_ACROSS_USERS", True)

    # --- Device ------------------------------------------------------------
    DEVICE: str = os.getenv("RETRIVA_DEVICE", "auto")  # auto | cpu | cuda
    TORCH_DTYPE: str = os.getenv("RETRIVA_TORCH_DTYPE", "float32")


# Force HuggingFace libraries to stay offline when local-only mode is on. All
# required models live in the local HF cache, so this guarantees the app never
# reaches out to the network at runtime.
_SETTINGS = Settings()
if _SETTINGS.HF_LOCAL_ONLY:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
