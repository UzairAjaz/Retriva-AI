"""
Retriva RAG engine (100% local).

Design
------
The engine is an *agent* with three tools:

* ``retrieve_documents`` -- hybrid retrieval (dense vector + BM25 fused with
  Reciprocal Rank Fusion) followed by a cross-encoder reranker. It powers all
  questions that can be answered from the knowledge base (SEC filings, tables
  and user-uploaded PDFs).
* ``answer_directly`` -- plain chat completion for general questions, so the
  assistant behaves like a normal chatbot (DeepSeek/ChatGPT style) when no
  document grounding is required.
* ``ask_clarification`` -- used when the request is too vague or when retrieval
  confidence is too low. Instead of hallucinating, Retriva asks a focused
  follow-up question.

A separate ChromaDB collection stores embeddings of past messages, giving each
chat long-term memory of its own history. Memory is strictly scoped to a single
chat: a conversation can never read another conversation's data. The only
knowledge shared across all of a user's chats is their profile/bio.
"""

import io
import json
import logging
import re
import threading
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional

import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from transformers import (
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
    pipeline,
)

from app.config import Settings
from app.vector_store import build_collection

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("retriva.engine")


# --------------------------------------------------------------------------- #
# Lexical helpers
# --------------------------------------------------------------------------- #
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> List[str]:
    return _TOKEN_RE.findall((text or "").lower())


COMPANY_TERMS = {
    "amazon", "apple", "google", "alphabet", "meta", "facebook",
    "microsoft", "nvidia", "tesla", "netflix", "intel", "amd",
}

FINANCE_TERMS = {
    "revenue", "revenues", "sales", "profit", "profits", "profitability",
    "loss", "losses", "earnings", "income", "expense", "expenses", "cost",
    "costs", "ebitda", "margin", "margins", "gross", "operating", "net",
    "cash", "debt", "liabilities", "assets", "equity", "shareholder",
    "shareholders", "dividend", "dividends", "stock", "share", "shares",
    "valuation", "expenditure", "capex", "guidance", "segment",
    "segments", "fiscal", "quarter", "quarterly", "annual", "yearly",
    "employees", "headcount", "growth", "buyback", "repurchase",
    "balance", "sheet", "cashflow", "financial", "financially", "finance",
    "investors", "roe", "roi", "eps",
}

COMPANY_SYNONYMS = {"alphabet": "google", "facebook": "meta"}

# Lightweight query expansion for retrieval. Small dense models and BM25 both
# benefit from knowing that "revenue" and "net sales" are the same thing.
QUERY_EXPANSIONS = {
    "revenue": "net sales total net sales",
    "revenues": "net sales total net sales",
    "sales": "net sales revenue",
    "profit": "net income operating income gross profit",
    "profits": "net income operating income",
    "profitability": "operating margin net margin",
    "income": "net income operating income",
    "loss": "net loss operating loss",
    "earnings": "net income diluted earnings per share",
    "expense": "operating expenses cost of sales research and development",
    "expenses": "operating expenses cost of sales research and development",
    "cost": "cost of sales operating expenses",
    "costs": "cost of sales operating expenses",
    "employees": "employees headcount human capital workforce",
    "headcount": "employees headcount human capital workforce",
    "debt": "long-term debt notes payable borrowings",
    "cash": "cash and cash equivalents free cash flow",
    "assets": "total assets",
    "liabilities": "total liabilities",
    "margin": "gross margin operating margin",
    "eps": "earnings per share diluted",
}


DOC_TERMS = {
    "document", "documents", "doc", "docs", "file", "files", "pdf", "upload",
    "uploaded", "report", "filing", "filings", "dataset",
    "policy", "policies", "guideline", "guidelines", "procedure", "procedures",
    "workflow", "manual", "handbook", "contract", "invoice",
}

FILING_RE = re.compile(r"\b(10-k|10-q|8-k|20-f|proxy|prospectus)\b")
YEAR_RE = re.compile(r"\b(20\d{2})\b")
QUARTER_RE = re.compile(r"\b(q[1-4]|first quarter|second quarter|third quarter|fourth quarter)\b")

STOPWORDS = {
    "what", "was", "were", "is", "are", "the", "a", "an", "of", "in", "for",
    "and", "or", "to", "on", "at", "by", "with", "from", "how", "much", "many",
    "did", "does", "do", "what's", "s", "tell", "me", "about", "please", "show",
    "give", "compare", "vs", "versus", "q1", "q2", "q3", "q4", "fiscal", "year",
    "total", "net",
}
GREETING_RE = re.compile(
    r"^\s*(hi|hey|hello|yo|howdy|hola|salam|assalamualaikum|good (morning|afternoon|evening)|"
    r"how are you|what'?s up|thanks|thank you|thx|bye|goodbye|ok|okay)\b"
)
FOLLOWUP_PREFIX_RE = re.compile(
    r"^\s*(what about|how about|and\b|what of|compare|vs\.?|versus|why|tell me more|"
    r"explain (that|this|it) more|elaborate|continue|go on|more details|also)\b"
)
FOLLOWUP_PRONOUNS = {
    "it", "its", "it's", "that", "this", "these", "those", "them", "they",
    "their", "he", "she", "his", "her", "earlier", "previous", "previously",
    "above", "before", "same", "there",
}
MEMORY_QUERY_RE = re.compile(
    r"(other (conversation|chat)|earlier|previously|we (discuss|discussed|talk|talked|said)|"
    r"did (i|we) (ask|talk|discuss|say)|our (conversation|chat)|chat history|"
    r"remember|last (time|chat|conversation)|mentioned)",
    re.IGNORECASE,
)
IDENTITY_RE = re.compile(
    r"(who am i|what(?:'s| is|s)?\s+my name|do you (?:know|remember|recall) my name|"
    r"what do you know about me|know about me|tell me about myself|do you remember me)",
    re.IGNORECASE,
)
REFUSAL_RE = re.compile(
    r"(cannot recall|can't recall|do not recall|don't recall|cannot remember|"
    r"can't remember|no (previous|earlier|prior) (conversation|chat|context)|"
    r"not aware of any|no record of|don't have (?:any )?(?:information|access) "
    r"(?:about|to) (?:previous|earlier|prior)|cannot access (?:previous|earlier|prior)|"
    r"no access to (?:previous|earlier|prior))",
    re.IGNORECASE,
)
UPLOAD_QUERY_RE = re.compile(
    r"(uploaded|ingested|i (?:uploaded|added|ingested)|my (?:own )?"
    r"(?:document|documents|doc|docs|file|files|pdf|pdfs)|the (?:document|documents|file|files) "
    r"(?:i|we) (?:uploaded|added|ingested))",
    re.IGNORECASE,
)


class _EventStoppingCriteria(StoppingCriteria):
    """Stops generation as soon as the event is set (client pressed Stop)."""

    def __init__(self, event: threading.Event):
        self._event = event

    def __call__(self, input_ids, scores, **kwargs) -> bool:  # noqa: ARG002
        return self._event.is_set()


class RetrivaEngine:
    """Local agentic RAG engine."""

    TOOLS = ("retrieve_documents", "answer_directly", "ask_clarification")
    def __init__(self, load_llm: bool = True):
        self.config = Settings()
        self.device = self._resolve_device(self.config.DEVICE)
        self.dtype = self._resolve_dtype(self.device)
        logger.info("Hardware detected: %s", self._hardware_summary())
        logger.info(
            "Initializing Retriva Engine (device=%s, dtype=%s)…",
            self.device,
            self.dtype,
        )

        self.embedding_model = HuggingFaceEmbeddings(
            model_name=self.config.EMBEDDING_MODEL_NAME,
            model_kwargs={"device": self.device},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.reranker = CrossEncoder(self.config.RERANKER_MODEL_NAME, device=self.device)

        self.tokenizer = None
        self.llm_pipeline = None
        if load_llm:
            self._load_llm()

        # Guards the in-memory index (id maps + BM25) during uploads.
        self._index_lock = threading.RLock()
        # The generation pipeline is not safe for concurrent calls; serialise.
        self._llm_lock = threading.Lock()

        self.collection = build_collection(
            self.config, self.config.COLLECTION_NAME, self.config.EMBEDDING_DIM
        )
        self.memory_collection = build_collection(
            self.config, self.config.MEMORY_COLLECTION_NAME, self.config.EMBEDDING_DIM
        )
        # Documents are split by a fast recursive splitter for uploads.
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=150, length_function=len
        )
        self._build_index()
        logger.info("Retriva Engine ready.")

    # ------------------------------------------------------------------ #
    # Setup helpers
    # ------------------------------------------------------------------ #
    def _resolve_device(self, requested: str) -> str:
        """Pick the best available device, honouring an explicit override.

        Priority for ``auto``: CUDA (NVIDIA) → MPS (Apple) → CPU. An explicit
        request for an unavailable device degrades gracefully to CPU so the
        container still starts (e.g. a GPU image running on a CPU node).
        """
        requested = (requested or "auto").strip().lower()
        cuda_available = torch.cuda.is_available()
        mps_backend = getattr(torch.backends, "mps", None)
        mps_available = bool(mps_backend and mps_backend.is_available())

        if requested in ("", "auto"):
            if cuda_available:
                return "cuda"
            if mps_available:
                return "mps"
            return "cpu"
        if requested.startswith("cuda") and not cuda_available:
            logger.warning("CUDA requested but unavailable – falling back to CPU.")
            return "cpu"
        if requested.startswith("mps") and not mps_available:
            logger.warning("MPS requested but unavailable – falling back to CPU.")
            return "cpu"
        return requested

    def _resolve_dtype(self, device: str):
        """Choose the compute dtype (float16/bfloat16 on GPU, float32 on CPU)."""
        requested = (self.config.TORCH_DTYPE or "auto").strip().lower()
        if requested not in ("", "auto"):
            dtype = getattr(torch, requested, None)
            if isinstance(dtype, torch.dtype):
                return dtype
            logger.warning("Unknown RETRIVA_TORCH_DTYPE=%r – using auto.", requested)
        if device.startswith("cuda"):
            try:
                if torch.cuda.is_bf16_supported():
                    return torch.bfloat16
            except Exception:  # noqa: BLE001
                pass
            return torch.float16
        return torch.float32

    def _llm_device_arg(self):
        """Device argument accepted by the transformers pipeline."""
        device = (self.config.LLM_DEVICE or "").strip() or self.device
        device = self._resolve_device(device)
        if device.startswith("cuda"):
            parts = device.split(":")
            return int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
        return device

    def _hardware_summary(self) -> str:
        parts = [f"torch={torch.__version__}"]
        if torch.cuda.is_available():
            try:
                name = torch.cuda.get_device_name(0)
            except Exception:  # noqa: BLE001
                name = "cuda"
            parts.append(f"cuda={torch.cuda.device_count()}x {name}")
        else:
            parts.append("cuda=unavailable")
        parts.append(f"device={self.device}")
        parts.append(f"dtype={self.dtype}")
        return " | ".join(parts)

    def _load_llm(self) -> None:
        logger.info("Loading local LLM: %s", self.config.LLM_MODEL_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.LLM_MODEL_NAME, local_files_only=self.config.HF_LOCAL_ONLY
        )
        device_arg = self._llm_device_arg()
        self.llm_pipeline = pipeline(
            "text-generation",
            model=self.config.LLM_MODEL_NAME,
            dtype=self.dtype,
            device=device_arg,
        )
        logger.info("LLM loaded on device=%s dtype=%s.", device_arg, self.dtype)

    def _build_index(self) -> None:
        """Load the vector store into memory and (re)build the BM25 index."""
        logger.info("Loading knowledge base (backend=%s) …", self.config.VECTOR_BACKEND)
        data = self.collection.get(include=["documents", "metadatas"])
        ids = data.get("ids") or []
        docs = data.get("documents") or []
        metas = data.get("metadatas") or []

        self.id_to_doc: Dict[str, str] = {}
        self.id_to_meta: Dict[str, dict] = {}
        for i, doc_id in enumerate(ids):
            self.id_to_doc[doc_id] = docs[i] or ""
            self.id_to_meta[doc_id] = metas[i] or {}
        self._rebuild_bm25()
        logger.info("Knowledge base contains %d chunks.", len(self.id_to_doc))

    def _rebuild_bm25(self) -> None:
        self._bm25_ids = list(self.id_to_doc.keys())
        corpus = [_tokens(self.id_to_doc[i]) for i in self._bm25_ids]
        self.bm25 = BM25Okapi(corpus) if corpus else None

    def _index_add(self, ids, docs, metas) -> None:
        with self._index_lock:
            for i, doc_id in enumerate(ids):
                self.id_to_doc[doc_id] = docs[i] or ""
                self.id_to_meta[doc_id] = metas[i] or {}
            self._rebuild_bm25()

    def _index_remove(self, ids) -> None:
        with self._index_lock:
            removed = False
            for doc_id in ids:
                if doc_id in self.id_to_doc:
                    del self.id_to_doc[doc_id]
                    self.id_to_meta.pop(doc_id, None)
                    removed = True
            if removed:
                self._rebuild_bm25()

    # ------------------------------------------------------------------ #
    # Agent planning
    # ------------------------------------------------------------------ #
    def _is_greeting(self, query_lower: str) -> bool:
        return bool(GREETING_RE.match(query_lower))

    def _is_finance_query(self, query_lower: str) -> bool:
        tokens = set(_tokens(query_lower))
        if tokens & COMPANY_TERMS:
            return True
        if tokens & FINANCE_TERMS:
            return True
        if FILING_RE.search(query_lower):
            return True
        # Multi-word financial phrases.
        for phrase in ("annual report", "quarterly report", "balance sheet",
                       "income statement", "cash flow", "market cap"):
            if phrase in query_lower:
                return True
        return False

    def _is_document_query(self, query_lower: str) -> bool:
        tokens = set(_tokens(query_lower))
        return bool(tokens & DOC_TERMS) or FILING_RE.search(query_lower) is not None

    def _is_memory_query(self, query: str) -> bool:
        query = query or ""
        return bool(MEMORY_QUERY_RE.search(query) or IDENTITY_RE.search(query))

    def _is_upload_query(self, query: str) -> bool:
        return bool(UPLOAD_QUERY_RE.search(query or ""))

    def _is_refusal(self, text: str) -> bool:
        return bool(REFUSAL_RE.search(text or ""))

    @staticmethod
    def _profile_tokens(profile: str) -> set:
        return {
            token
            for token in _tokens(profile or "")
            if len(token) >= 5 and token not in STOPWORDS
        }

    def _answer_uses_profile(self, answer: str, profile: str) -> bool:
        tokens = self._profile_tokens(profile)
        if not tokens:
            return True
        lowered = (answer or "").lower()
        return any(token in lowered for token in tokens)

    def _profile_fallback(self, profile: str) -> str:
        return "From your profile: " + (profile or "").strip()

    def _recall_summary(self, memory: List[Dict]) -> str:
        """Deterministic answer for 'what did we discuss' style questions."""
        user_messages = [
            (m.get("content") or "").strip()
            for m in reversed(memory)
            if m.get("role") == "user" and (m.get("content") or "").strip()
        ]
        if not user_messages:
            user_messages = [
                (m.get("content") or "").strip() for m in memory if m.get("content")
            ]
        seen = []
        for message in user_messages:
            snippet = message[:200]
            if snippet not in seen:
                seen.append(snippet)
        if not seen:
            return "I don't have anything recorded earlier in this chat yet."
        bullets = "\n".join(f"- {item}" for item in seen[:5])
        return (
            "Here's what you've asked me about earlier in this chat:\n" + bullets
        )

    def _is_followup(self, query_lower: str) -> bool:
        """True only for genuinely context-dependent/elliptical questions."""
        if FOLLOWUP_PREFIX_RE.match(query_lower):
            return True
        return bool(set(_tokens(query_lower)) & FOLLOWUP_PRONOUNS)

    def _entities_from_memory(self, memory: List[Dict]) -> List[str]:
        found: List[str] = []
        for item in memory:
            text = (item.get("content") or "").lower()
            for company in COMPANY_TERMS:
                if company in text and company not in found:
                    found.append(company)
        return found

    def _resolve_followup(self, query: str, memory: List[Dict]) -> str:
        """Make anaphoric follow-ups self contained using memory entities."""
        if not memory or self._detect_companies(query):
            return query
        query_lower = query.lower()
        should_resolve = self._is_followup(query_lower)
        if not should_resolve:
            # e.g. "what about profit?" -> short + finance term, inherits subject.
            should_resolve = (
                self._is_finance_query(query_lower) and len(_tokens(query)) <= 6
            )
        if not should_resolve:
            return query
        entities = self._entities_from_memory(memory)
        if not entities:
            return query
        prefix = " ".join(entities[:2])
        return f"{prefix} {query.strip()}"

    def _plan(self, query: str, memory: List[Dict]) -> Dict:
        query = (query or "").strip()
        query_lower = query.lower()
        if not query:
            return {
                "tool": "ask_clarification",
                "clarification_question": "What would you like to know?",
                "search_query": "",
                "explicit": False,
            }

        finance = self._is_finance_query(query_lower)
        document = self._is_document_query(query_lower)
        greeting = self._is_greeting(query_lower)
        followup = self._is_followup(query_lower)

        # 1. Explicit finance/document questions always search the knowledge base.
        if finance or document:
            return {
                "tool": "retrieve_documents",
                "search_query": self._resolve_followup(query, memory),
                "explicit": True,
            }

        # 2. Greetings and small talk are answered as normal chat.
        if greeting:
            return {"tool": "answer_directly", "search_query": query, "explicit": False}

        # 3. Elliptical follow-ups ("what about their profit?") use memory.
        if followup and memory:
            return {
                "tool": "retrieve_documents",
                "search_query": self._resolve_followup(query, memory),
                "explicit": False,
            }

        # 4. Truly empty/vague requests get a clarification instead of a guess.
        if len(_tokens(query)) <= 2:
            return {
                "tool": "ask_clarification",
                "clarification_question": (
                    "Could you tell me a little more about what you're looking for? "
                    "Adding a company, metric or document name helps a lot."
                ),
                "search_query": query,
                "explicit": False,
            }

        if self.config.AGENT_LLM_PLANNING:
            llm_plan = self._llm_plan(query, memory)
            if llm_plan:
                llm_plan.setdefault("explicit", False)
                return llm_plan

        # 5. Everything else is general conversation.
        return {"tool": "answer_directly", "search_query": query, "explicit": False}

    def _llm_plan(self, query: str, memory: List[Dict]) -> Optional[Dict]:
        """Optional small-LLM planner. Returns None to fall back to heuristics."""
        memory_text = "\n".join(
            f"- {item.get('role', 'user')}: {(item.get('content') or '')[:200]}"
            for item in memory[:3]
        ) or "(none)"
        system = (
            "You are a routing agent with three tools:\n"
            "1. retrieve_documents: for questions about companies, finance, SEC filings "
            "or uploaded documents.\n"
            "2. answer_directly: for general knowledge, chit-chat and everything else.\n"
            "3. ask_clarification: when the request is too vague to act on.\n"
            "Reply with ONE line of JSON only, e.g. "
            '{"tool": "retrieve_documents", "search_query": "apple revenue 2024"}'
        )
        messages = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": f"Conversation memory:\n{memory_text}\n\nUser request: {query}",
            },
        ]
        try:
            raw = self._generate(messages, temperature=0.0, max_new_tokens=64)
            match = re.search(r"\{.*?\}", raw, re.DOTALL)
            if not match:
                return None
            data = json.loads(match.group(0))
            tool = data.get("tool")
            if tool not in self.TOOLS:
                return None
            plan = {"tool": tool, "search_query": data.get("search_query") or query}
            if tool == "ask_clarification":
                plan["clarification_question"] = data.get(
                    "clarification_question",
                    "Could you give me a little more detail about what you need?",
                )
            return plan
        except Exception as exc:  # noqa: BLE001 - planning is best effort
            logger.warning("LLM planner failed: %s", exc)
            return None

    # ------------------------------------------------------------------ #
    # Retrieval
    # ------------------------------------------------------------------ #
    def _build_where_clause(self, user_id: str) -> Optional[dict]:
        """ChromaDB metadata filter.

        In the default (shared) mode every user sees every document, which
        satisfies the requirement that the finance corpus *and* user uploads are
        available to all users. Set ``SHARE_KNOWLEDGE_ACROSS_USERS=false`` to
        fall back to per-user isolation.
        """
        if self.config.SHARE_KNOWLEDGE_ACROSS_USERS:
            return None
        return {"$or": [{"user_id": "global"}, {"user_id": str(user_id)}]}

    def _expand_query(self, query: str) -> str:
        extras = []
        for token in _tokens(query):
            expansion = QUERY_EXPANSIONS.get(token)
            if expansion and expansion not in extras:
                extras.append(expansion)
        if not extras:
            return query
        return f"{query} {' '.join(extras)}"

    def _detect_companies(self, query: str) -> List[str]:
        tokens = set(_tokens(query))
        found: List[str] = []
        for term in COMPANY_TERMS:
            if term in tokens:
                canonical = COMPANY_SYNONYMS.get(term, term)
                if canonical not in found:
                    found.append(canonical)
        return found

    def _hybrid_retrieve_ids(
        self, query: str, user_id: str, n: int, company: Optional[str] = None
    ) -> List[str]:
        where = self._build_where_clause(user_id)
        if company:
            company_clause = {"company": company}
            where = (
                company_clause
                if where is None
                else {"$and": [where, company_clause]}
            )

        # --- Dense (vector) leg ---
        vector_ids: List[str] = []
        try:
            query_embedding = self.embedding_model.embed_query(query)
            vector_res = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=max(n, 1),
                where=where,
            )
            if vector_res.get("ids"):
                vector_ids = [i for i in vector_res["ids"][0] if i in self.id_to_doc]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Vector search failed: %s", exc)

        # --- Lexical (BM25) leg ---
        keyword_ids: List[str] = []
        with self._index_lock:
            if self.bm25 is not None:
                scores = self.bm25.get_scores(_tokens(query))
                ranked = sorted(
                    range(len(scores)), key=lambda i: scores[i], reverse=True
                )
                for idx in ranked[: max(n * 3, 1)]:
                    if scores[idx] <= 0:
                        break
                    doc_id = self._bm25_ids[idx]
                    meta = self.id_to_meta.get(doc_id, {})
                    if where and not self._matches_where(meta, where):
                        continue
                    keyword_ids.append(doc_id)
                    if len(keyword_ids) >= n:
                        break

        # --- Reciprocal Rank Fusion ---
        return self._rrf([vector_ids, keyword_ids])

    @staticmethod
    def _matches_where(meta: dict, where: dict) -> bool:
        """Minimal evaluator for the metadata filters we generate."""
        if not where:
            return True
        if "$and" in where:
            return all(RetrivaEngine._matches_where(meta, c) for c in where["$and"])
        if "$or" in where:
            return any(RetrivaEngine._matches_where(meta, c) for c in where["$or"])
        return all(meta.get(k) == v for k, v in where.items())

    def _rrf(self, ranked_lists: List[List[str]]) -> List[str]:
        k = self.config.RRF_K
        scores: Dict[str, float] = {}
        for ranked in ranked_lists:
            for rank, doc_id in enumerate(ranked):
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
        return [doc_id for doc_id, _ in sorted(scores.items(), key=lambda x: -x[1])]

    def _table_lookup(
        self,
        query: str,
        company: Optional[str],
        year: Optional[str],
    ) -> List[Dict]:
        """Targeted lookup of numeric tables for a company + fiscal year.

        Cross-encoders trained on web passages systematically under-rank long
        financial tables, which are exactly where the numbers live. For a
        question like "Amazon revenue 2024" we therefore pull the matching
        table chunks explicitly and put them first in the context.
        """
        if not company or not year:
            return []
        try:
            where = {
                "$and": [
                    {"company": company},
                    {"year": year},
                    {"source_type": "table"},
                ]
            }
            data = self.collection.get(where=where, include=["documents", "metadatas"])
        except Exception as exc:  # noqa: BLE001
            logger.warning("Table lookup failed: %s", exc)
            return []

        ids = data.get("ids") or []
        docs = data.get("documents") or []
        metas = data.get("metadatas") or []

        query_lower = query.lower()
        preferred_type = "10-Q" if QUARTER_RE.search(query_lower) else "10-K"
        metric_terms = {
            token
            for token in _tokens(self._expand_query(query))
            if token not in STOPWORDS and token not in COMPANY_TERMS
        }

        scored = []
        for i, doc_id in enumerate(ids):
            text = docs[i] or ""
            meta = metas[i] or {}
            overlap = sum(1 for term in metric_terms if term in text.lower())
            if overlap == 0:
                continue
            bonus = 2 if meta.get("doc_type") == preferred_type else 0
            scored.append((overlap + bonus, doc_id, text, meta))

        scored.sort(key=lambda item: -item[0])
        return [
            {"id": doc_id, "text": text, "meta": meta, "score": float(rank)}
            for rank, (_, doc_id, text, meta) in enumerate(scored[:3], start=1)
        ]

    def _table_candidates(self, query: str, companies: List[str]) -> List[Dict]:
        year_match = YEAR_RE.search(query)
        year = year_match.group(1) if year_match else None
        hits: List[Dict] = []
        for company in companies[:2]:
            hits.extend(self._table_lookup(query, company, year))
        return hits

    def _upload_lookup(self, query: str, top_n: int = 6) -> List[Dict]:
        """Find the user's most relevant uploaded chunks.

        Uploaded documents are usually summarised rather than keyword-matched,
        so for document-themed questions we make sure the user's own files are
        represented instead of only SEC filings. File-name matches are weighted
        more heavily than body matches (e.g. "the plan document" -> plan.pdf).
        """
        try:
            data = self.collection.get(
                where={"source_type": "pdf_upload"}, include=["documents", "metadatas"]
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Upload lookup failed: %s", exc)
            return []
        ids = data.get("ids") or []
        docs = data.get("documents") or []
        metas = data.get("metadatas") or []
        query_tokens = {
            t
            for t in _tokens(query)
            if len(t) >= 3 and t not in STOPWORDS and not t.isdigit()
        }
        if not query_tokens:
            return []
        scored = []
        for i, doc_id in enumerate(ids):
            text = docs[i] or ""
            meta = metas[i] or {}
            filename_tokens = set(_tokens(str(meta.get("source_file", ""))))
            body = text.lower()
            filename_hits = len(query_tokens & filename_tokens)
            body_hits = sum(1 for token in query_tokens if token in body)
            score = 3 * filename_hits + body_hits
            if score:
                scored.append((score, filename_hits, doc_id, text, meta))
        scored.sort(key=lambda item: -item[0])
        return [
            {
                "id": doc_id,
                "text": text,
                "meta": meta,
                "score": float(score),
                "filename_hits": filename_hits,
            }
            for score, filename_hits, doc_id, text, meta in scored[:top_n]
        ]


    def _filename_boost(self, query: str, meta: dict) -> float:
        """Reward chunks whose file name shares meaningful words with the query."""
        source = str(meta.get("source_file", "") or "")
        if not source:
            return 0.0
        file_tokens = {
            t for t in _tokens(source) if t not in STOPWORDS and not t.isdigit()
        }
        query_tokens = {
            t for t in _tokens(query) if t not in STOPWORDS and not t.isdigit()
        }
        overlap = file_tokens & query_tokens
        return 2.0 * len(overlap)

    def retrieve(
        self, query: str, user_id: str = "global", top_k: Optional[int] = None
    ) -> List[Dict]:
        """Hybrid retrieval + cross-encoder reranking.

        When the query mentions several companies (e.g. "compare Apple and
        Google revenue") we retrieve candidates per company and merge them, so
        every side of the comparison is represented.
        """
        top_k = top_k or self.config.TOP_N_RERANKED
        companies = self._detect_companies(query)
        retrieval_query = self._expand_query(query)

        if len(companies) >= 2:
            candidate_ids: List[str] = []
            for company in companies:
                per_company = self._hybrid_retrieve_ids(
                    retrieval_query, user_id, self.config.TOP_N_RETRIEVAL, company=company
                )
                if not per_company:
                    per_company = self._hybrid_retrieve_ids(
                        retrieval_query, user_id, self.config.TOP_N_RETRIEVAL, company=None
                    )
                candidate_ids.extend(per_company)
        else:
            company = companies[0] if companies else None
            candidate_ids = self._hybrid_retrieve_ids(
                retrieval_query, user_id, self.config.TOP_N_RETRIEVAL, company=company
            )
            if not candidate_ids and company:
                candidate_ids = self._hybrid_retrieve_ids(
                    retrieval_query, user_id, self.config.TOP_N_RETRIEVAL, company=None
                )

        # De-duplicate while preserving order.
        seen = set()
        ordered_ids = []
        for doc_id in candidate_ids:
            if doc_id not in seen and doc_id in self.id_to_doc:
                seen.add(doc_id)
                ordered_ids.append(doc_id)

        if not ordered_ids:
            return []

        candidates = [
            {
                "id": doc_id,
                "text": self.id_to_doc.get(doc_id, ""),
                "meta": self.id_to_meta.get(doc_id, {}),
            }
            for doc_id in ordered_ids
        ]

        # --- Cross-encoder reranking ---
        # The reranker sees the source filename too: names like
        # "amazon 10-k 2024.md" or "SE-Git Workflow.pdf" carry strong signals
        # that the chunk body alone does not.
        try:
            pairs = [
                [query, f"{c['meta'].get('source_file', '')}. {c['text']}"]
                for c in candidates
            ]
            scores = self.reranker.predict(pairs)
            for i, c in enumerate(candidates):
                c["score"] = float(scores[i])
        except Exception as exc:  # noqa: BLE001
            logger.warning("Reranking failed (%s); using hybrid order.", exc)
            for i, c in enumerate(candidates):
                c["score"] = 1.0 - (i / max(len(candidates), 1))

        # Keyword overlap between the query and the file name.
        for candidate in candidates:
            candidate["score"] += self._filename_boost(query, candidate["meta"])

        candidates.sort(key=lambda c: c["score"], reverse=True)

        # Numeric questions: guarantee the relevant table is in the context.
        table_hits = self._table_candidates(query, companies)
        if table_hits:
            best = candidates[0]["score"] if candidates else 1.0
            hit_ids = set()
            boosted = []
            for rank, hit in enumerate(table_hits):
                hit_ids.add(hit["id"])
                hit["score"] = max(best, 1.0) + (len(table_hits) - rank)
                boosted.append(hit)
            candidates = boosted + [c for c in candidates if c["id"] not in hit_ids]

        # Questions with no company/finance signal that are about documents or
        # that name an uploaded file: answer from the user's own uploads only,
        # so unrelated SEC content cannot confuse or derail the answer.
        if not companies and not self._is_finance_query(query.lower()):
            upload_hits = self._upload_lookup(query)
            if upload_hits:
                named_upload = any(hit.get("filename_hits", 0) > 0 for hit in upload_hits)
                if self._is_document_query(query.lower()) or named_upload:
                    for rank, hit in enumerate(upload_hits):
                        hit["score"] = float(len(upload_hits) - rank)
                    return upload_hits[:top_k]

        # For comparison queries make sure every company is represented.
        if len(companies) >= 2:
            by_company: Dict[str, List[Dict]] = {c: [] for c in companies}
            for candidate in candidates:
                comp = candidate["meta"].get("company")
                if comp in by_company:
                    by_company[comp].append(candidate)
            balanced: List[Dict] = []
            index = 0
            while len(balanced) < top_k:
                added = False
                for comp in companies:
                    if index < len(by_company[comp]) and len(balanced) < top_k:
                        balanced.append(by_company[comp][index])
                        added = True
                if not added:
                    break
                index += 1
            if balanced:
                return balanced

        return candidates[:top_k]

    def _format_sources(self, results: List[Dict]) -> List[Dict]:
        sources = []
        for res in results:
            meta = res.get("meta", {})
            sources.append(
                {
                    "company": str(meta.get("company", "Unknown")),
                    "year": str(meta.get("year", "Unknown")),
                    "type": str(meta.get("doc_type", "Unknown")),
                    "source_type": str(meta.get("source_type", "text")),
                    "source_file": str(meta.get("source_file", "Unknown")),
                    "page_number": str(meta.get("page_number", "Unknown")),
                    "relevance_score": float(res.get("score", 0.0)),
                    "text": str(res.get("text", "")),
                }
            )
        return sources

    # ------------------------------------------------------------------ #
    # Memory (cross-chat)
    # ------------------------------------------------------------------ #
    def memory_add(
        self,
        content: str,
        role: str,
        user_id: str = "global",
        chat_id: str = "",
        message_id: Optional[str] = None,
    ) -> None:
        if not self.config.ENABLE_MEMORY:
            return
        content = (content or "").strip()
        if len(content) < 3:
            return
        text = content[:4000]
        try:
            embedding = self.embedding_model.embed_documents([text])[0]
            memory_id = message_id or f"mem_{uuid.uuid4().hex}"
            self.memory_collection.upsert(
                ids=[memory_id],
                embeddings=[embedding],
                documents=[text],
                metadatas=[
                    {
                        "user_id": str(user_id),
                        "chat_id": str(chat_id or ""),
                        "role": str(role),
                        "created_at": int(time.time()),
                    }
                ],
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to store memory: %s", exc)

    @staticmethod
    def _memory_where(user_id: str, chat_id: str) -> dict:
        """Memory is scoped to a single chat so no chat can see another's data."""
        if chat_id:
            return {"chat_id": str(chat_id)}
        return {"user_id": str(user_id)}

    def memory_search(
        self,
        query: str,
        user_id: str = "global",
        chat_id: str = "",
        top_k: Optional[int] = None,
    ) -> List[Dict]:
        if not self.config.ENABLE_MEMORY:
            return []
        try:
            if self.memory_collection.count() == 0:
                return []
            top_k = top_k or self.config.MEMORY_TOP_K
            embedding = self.embedding_model.embed_query(query)
            res = self.memory_collection.query(
                query_embeddings=[embedding],
                n_results=max(top_k, 1),
                where=self._memory_where(user_id, chat_id),
            )
            documents = (res.get("documents") or [[]])[0]
            metadatas = (res.get("metadatas") or [[]])[0]
            distances = (res.get("distances") or [[]])[0]
            hits: List[Dict] = []
            for i, doc in enumerate(documents):
                distance = float(distances[i]) if i < len(distances) else 0.0
                if distance > self.config.MEMORY_MAX_DISTANCE:
                    continue
                meta = metadatas[i] if i < len(metadatas) else {}
                hits.append(
                    {
                        "role": (meta or {}).get("role", "user"),
                        "content": doc,
                        "chat_id": (meta or {}).get("chat_id", ""),
                        "distance": distance,
                    }
                )
            return hits
        except Exception as exc:  # noqa: BLE001
            logger.warning("Memory search failed: %s", exc)
            return []

    def memory_recent(
        self, chat_id: str = "", user_id: str = "global", top_k: Optional[int] = None
    ) -> List[Dict]:
        """Most recent memory items of this chat, regardless of similarity."""
        if not self.config.ENABLE_MEMORY:
            return []
        try:
            if self.memory_collection.count() == 0:
                return []
            data = self.memory_collection.get(
                where=self._memory_where(user_id, chat_id),
                include=["documents", "metadatas"],
            )
            documents = data.get("documents") or []
            metadatas = data.get("metadatas") or []
            items: List[Dict] = []
            for i, doc in enumerate(documents):
                meta = metadatas[i] if i < len(metadatas) else {}
                items.append(
                    {
                        "role": (meta or {}).get("role", "user"),
                        "content": doc,
                        "chat_id": (meta or {}).get("chat_id", ""),
                        "distance": 0.0,
                        "created_at": (meta or {}).get("created_at", 0),
                    }
                )
            items.sort(key=lambda x: x.get("created_at", 0), reverse=True)
            return items[: top_k or self.config.MEMORY_TOP_K]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Recent memory fetch failed: %s", exc)
            return []

    def memory_delete_chat(self, chat_id: str) -> None:
        """Drop all memory belonging to a deleted chat."""
        if not chat_id:
            return
        try:
            data = self.memory_collection.get(where={"chat_id": str(chat_id)})
            ids = data.get("ids") or []
            if ids:
                self.memory_collection.delete(ids=ids)
                logger.info("Deleted %d memory items for chat %s.", len(ids), chat_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to delete chat memory: %s", exc)

    # ------------------------------------------------------------------ #
    # Prompting
    # ------------------------------------------------------------------ #
    def _build_messages(
        self,
        user_query: str,
        results: Optional[List[Dict]],
        memory: List[Dict],
        chat_history: Optional[List[Dict]] = None,
        recall: bool = False,
        profile: str = "",
    ) -> List[Dict]:
        if results:
            context = "\n\n".join(
                f"[{i + 1}] File: {r['meta'].get('source_file', 'Unknown')} "
                f"(company: {r['meta'].get('company', '?')}, "
                f"year: {r['meta'].get('year', '?')}, "
                f"type: {r['meta'].get('doc_type', '?')})\n"
                f"{str(r['text'])[: self.config.MAX_CHUNK_CHARS]}"
                for i, r in enumerate(results[: self.config.MAX_CONTEXT_CHUNKS])
            )
            is_upload = results[0]["meta"].get("source_type") == "pdf_upload"
            if is_upload:
                system = (
                    "You are Retriva, a helpful assistant. The context below comes "
                    "from document(s) the user uploaded. Answer their request using "
                    "ONLY this context; when asked to summarise, produce a clear, "
                    "organised summary of the document. Never claim the document "
                    "does not exist — the context is its content. Add inline "
                    "citations like [1], [2] when useful.\n\n"
                    f"UPLOADED DOCUMENT CONTEXT:\n{context}"
                )
            else:
                system = (
                    "You are Retriva, a meticulous financial research assistant. "
                    "Answer the user's question using the retrieved context below.\n\n"
                    "RULES:\n"
                    "1. Report figures exactly as they appear in the context, with the "
                    "right units (millions/billions) and period.\n"
                    "2. Tables contain the exact numbers — read them carefully and do "
                    "not confuse quarterly with annual figures.\n"
                    "3. Add inline citations like [1], [2] that map to the context blocks.\n"
                    "4. NEVER guess or invent a number. If the figure is not explicitly "
                    "in the context, say you could not find it in the documents.\n"
                    "5. Prefer concise Markdown; use tables when comparing figures.\n\n"
                    f"RETRIEVED CONTEXT:\n{context}"
                )
        else:
            system = (
                "You are Retriva, a helpful, knowledgeable and friendly AI "
                "assistant. Answer general questions naturally and accurately, "
                "like a normal chatbot. If the user shares a fact about "
                "themselves or makes a statement rather than asking a question, "
                "acknowledge it warmly and continue the conversation. Be concise, "
                "use Markdown when helpful, and never fabricate facts. If you "
                "don't know something, say so."
            )

        if memory:
            memory_lines = "\n".join(
                f"- {item.get('role', 'user').capitalize()}: "
                f"{(item.get('content') or '')[:500]}"
                for item in memory[: max(self.config.MEMORY_TOP_K, 6)]
            )
            if recall:
                system = (
                    "You are Retriva. The user is asking about earlier in this "
                    "conversation or about what you remember from it. Answer using "
                    "ONLY the conversation excerpts below. If the user asks whether "
                    "you remember, say yes and briefly mention the topics from the "
                    "excerpts. If the excerpts do not contain the answer, say you "
                    "don't recall. Never invent topics that are not listed.\n\n"
                    f"EARLIER IN THIS CONVERSATION:\n{memory_lines}"
                )
            else:
                system += (
                    "\n\nEarlier in this conversation (use it to resolve follow-ups "
                    "and stay consistent; do not mention it unless relevant):\n"
                    + memory_lines
                )

        if profile and profile.strip():
            system += (
                "\n\nThe following is the USER'S personal profile. It describes the "
                "user, NOT you. Always remember it and use it naturally; when asked "
                "about the user's name/role, answer from this profile.\n"
                + profile.strip()[:2000]
            )

        messages: List[Dict] = [{"role": "system", "content": system}]

        history = list(chat_history or [])
        # Avoid duplicating the current turn if the caller already appended it.
        if history:
            last = history[-1]
            if last.get("role") == "user" and last.get("content", "").strip() == user_query.strip():
                history = history[:-1]
        for turn in history[-self.config.MAX_HISTORY_TURNS * 2:]:
            role = turn.get("role")
            content = (turn.get("content") or "").strip()
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})

        messages.append({"role": "user", "content": user_query})
        return messages

    def _apply_template(self, messages: List[Dict]) -> str:
        if self.tokenizer is None:
            self._load_llm()
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #
    def _generation_kwargs(self, temperature: float, max_new_tokens: int) -> Dict:
        kwargs: Dict = {"max_new_tokens": max_new_tokens}
        pad_id = None
        if self.tokenizer is not None:
            pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        if pad_id is not None:
            kwargs["pad_token_id"] = pad_id
        if temperature and temperature > 0.25:
            kwargs.update(
                do_sample=True,
                temperature=float(temperature),
                top_p=self.config.TOP_P,
                top_k=self.config.TOP_K,
            )
        else:
            # Low temperature (the default) uses greedy decoding: more stable and
            # more accurate for factual/finance answers.
            kwargs.update(do_sample=False)
        return kwargs

    def _generate(
        self,
        messages: List[Dict],
        temperature: float = 0.1,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        if self.llm_pipeline is None:
            self._load_llm()
        prompt = self._apply_template(messages)
        max_new_tokens = max_new_tokens or self.config.MAX_NEW_TOKENS
        try:
            with self._llm_lock:
                output = self.llm_pipeline(
                    prompt,
                    return_full_text=False,
                    **self._generation_kwargs(temperature, max_new_tokens),
                )
            text = output[0]["generated_text"]
        except Exception as exc:  # noqa: BLE001
            logger.error("Generation failed: %s", exc)
            return "I couldn't generate a response right now. Please try again."
        return text.strip()

    def _stream_generate(
        self, messages: List[Dict], temperature: float = 0.1
    ) -> object:
        """Yield generated text chunks as they are produced."""
        if self.llm_pipeline is None:
            self._load_llm()
        prompt = self._apply_template(messages)
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        stop_event = threading.Event()
        kwargs = self._generation_kwargs(temperature, self.config.MAX_NEW_TOKENS)
        kwargs["text_inputs"] = prompt
        kwargs["streamer"] = streamer
        kwargs["stopping_criteria"] = StoppingCriteriaList(
            [_EventStoppingCriteria(stop_event)]
        )

        errors: Dict[str, BaseException] = {}

        def _run() -> None:
            try:
                with self._llm_lock:
                    self.llm_pipeline(**kwargs)
            except BaseException as exc:  # noqa: BLE001
                errors["error"] = exc
                logger.error("Streaming generation failed: %s", exc)
            finally:
                streamer.end()

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        try:
            for piece in streamer:
                if piece:
                    yield piece
        finally:
            # Runs when the client disconnects / presses Stop: signals the model
            # to stop at the next decoding step instead of generating to the end.
            stop_event.set()
            streamer.end()
            thread.join(timeout=5)
        if "error" in errors:
            yield "\n\n*(generation interrupted)*"

    # ------------------------------------------------------------------ #
    # High level query flow
    # ------------------------------------------------------------------ #
    def _no_upload_text(self, query: str) -> str:
        return (
            f"I don't see any uploaded document matching “{query}” in the "
            "knowledge base. You can upload a PDF from the sidebar and I'll "
            "index it right away — or ask me about the filings already available "
            "(Amazon, Apple, Google, Meta)."
        )

    def _clarification_text(
        self, query: str, results: Optional[List[Dict]] = None
    ) -> str:
        closest = ""
        if results:
            source = results[0].get("meta", {}).get("source_file")
            if source:
                closest = (
                    f" The closest match I found was “{source}”, but I'm not "
                    "confident it answers your question."
                )
        return (
            f"I couldn't find a confident answer to “{query}” in the knowledge "
            f"base.{closest} Could you name the company and the period (for "
            "example “Apple revenue FY2024”), or tell me which document to use?"
        )

    def _prepare(
        self,
        user_query: str,
        user_id: str,
        chat_history: Optional[List[Dict]] = None,
        chat_id: str = "",
        profile: str = "",
    ) -> Dict:
        """Run the agent: plan → act (retrieve) → decide on clarification.

        Memory is scoped to ``chat_id``: a conversation can only see its own
        history. The only cross-chat knowledge is the user's profile/bio.
        """
        memory = (
            self.memory_search(user_query, user_id, chat_id)
            if self.config.ENABLE_MEMORY
            else []
        )
        # The current turn may already be persisted; never feed it back as memory.
        query_norm = (user_query or "").strip()
        memory = [
            item
            for item in memory
            if (item.get("content") or "").strip() != query_norm
        ]
        # Also drop anything already present as an explicit turn in this chat.
        history_keys = {
            (turn.get("content") or "").strip()[:80]
            for turn in (chat_history or [])
        }
        memory = [
            item
            for item in memory
            if (item.get("content") or "").strip()[:80] not in history_keys
        ]

        is_recall = self._is_memory_query(user_query)
        if self.config.ENABLE_MEMORY:
            # Long-term memory of *this* chat only.
            recent_k = (
                self.config.MEMORY_TOP_K * 2 if is_recall else max(4, self.config.MEMORY_TOP_K)
            )
            seen = {(m.get("role"), (m.get("content") or "")[:80]) for m in memory}
            for item in self.memory_recent(chat_id, user_id, recent_k):
                content = (item.get("content") or "").strip()
                if not content or content == query_norm:
                    continue
                if content[:80] in history_keys:
                    continue
                key = (item.get("role"), content[:80])
                if key not in seen:
                    memory.append(item)
                    seen.add(key)
        plan = self._plan(user_query, memory)
        tool = plan["tool"]

        has_profile = bool(profile and profile.strip())

        # Questions about the user themselves are answered from the profile with
        # a focused prompt (no history/KB to distract the model).
        if IDENTITY_RE.search(user_query) and has_profile:
            return {
                "tool": "answer_directly",
                "text": None,
                "sources": [],
                "memory": memory,
                "recall": True,
                "identity": True,
                "profile": profile,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are Retriva, a helpful assistant. The text below is "
                            "the USER'S personal profile — it describes the user, NOT "
                            "you. Answer the user's question about themselves using "
                            "ONLY this profile. Never describe yourself or invent "
                            "details.\n\nUSER PROFILE:\n" + profile.strip()[:2000]
                        ),
                    },
                    {"role": "user", "content": user_query},
                ],
                "needs_clarification": False,
            }

        if is_recall and not memory and not chat_history and not has_profile:
            return {
                "tool": "answer_directly",
                "text": (
                    "I don't have anything earlier in this chat to recall yet. "
                    "Ask me a few things and I'll remember them for this conversation."
                ),
                "sources": [],
                "memory": memory,
                "recall": is_recall,
                "messages": None,
                "needs_clarification": False,
            }

        # Not an obvious finance/document question: silently probe the knowledge
        # base so user-uploaded documents are found without keyword hints. If
        # nothing is confident, we fall back to normal chat.
        greeting = self._is_greeting(user_query.lower())
        if tool == "answer_directly" and not is_recall and not greeting:
            probe = self.retrieve(user_query, user_id)
            if probe and probe[0].get("score", 0.0) >= self.config.RAG_CONFIDENCE_THRESHOLD:
                plan = {
                    "tool": "retrieve_documents",
                    "_results": probe,
                    "explicit": False,
                }
                tool = "retrieve_documents"

        if tool == "retrieve_documents":
            explicit = bool(plan.get("explicit"))
            results = plan.get("_results") or self.retrieve(
                plan.get("search_query") or user_query, user_id
            )
            weak = (not results) or (
                results[0].get("score", 0.0) < self.config.CLARIFICATION_THRESHOLD
            )
            # Uploaded documents are frequently summarised rather than
            # keyword-matched, so the cross-encoder scores them low. As long as
            # an uploaded doc is the best match, still answer from it.
            if (
                results
                and results[0].get("meta", {}).get("source_type") == "pdf_upload"
                and results[0].get("score", 0.0)
                >= self.config.UPLOAD_CLARIFICATION_THRESHOLD
            ):
                weak = False
            if weak:
                # Only ask for clarification when the user actually asked a
                # finance/document question; otherwise behave like normal chat.
                if explicit and self._is_upload_query(user_query):
                    return {
                        "tool": "answer_directly",
                        "text": self._no_upload_text(user_query),
                        "sources": [],
                        "memory": memory,
                        "recall": is_recall,
                        "messages": None,
                        "needs_clarification": False,
                    }
                if explicit:
                    logger.info("Weak retrieval for explicit query → clarification.")
                    return {
                        "tool": "ask_clarification",
                        "text": self._clarification_text(user_query, results),
                        "sources": [],
                        "memory": memory,
                        "recall": is_recall,
                        "messages": None,
                        "needs_clarification": True,
                    }
                return {
                    "tool": "answer_directly",
                    "text": None,
                    "sources": [],
                    "memory": memory,
                    "recall": is_recall,
                    "messages": self._build_messages(
                        user_query, None, memory, chat_history,
                        recall=is_recall, profile=profile,
                    ),
                    "needs_clarification": False,
                }
            return {
                "tool": "retrieve_documents",
                "text": None,
                "sources": self._format_sources(results),
                "memory": memory,
                "recall": is_recall,
                "messages": self._build_messages(
                    user_query, results, memory, chat_history,
                    recall=is_recall, profile=profile,
                ),
                "needs_clarification": False,
            }

        if tool == "ask_clarification":
            text = plan.get("clarification_question") or self._clarification_text(user_query)
            return {
                "tool": "ask_clarification",
                "text": text,
                "sources": [],
                "memory": memory,
                "recall": is_recall,
                "messages": None,
                "needs_clarification": True,
            }

        return {
            "tool": "answer_directly",
            "text": None,
            "sources": [],
            "memory": memory,
            "recall": is_recall,
            "messages": self._build_messages(
                user_query, None, memory, chat_history,
                recall=is_recall, profile=profile,
            ),
            "needs_clarification": False,
        }

    def query(
        self,
        user_query: str,
        user_id: str = "default_user",
        chat_history: Optional[List[Dict]] = None,
        temperature: float = 0.1,
        chat_id: str = "",
        profile: str = "",
    ) -> Dict:
        start = time.time()
        prepared = self._prepare(
            user_query, user_id, chat_history, chat_id=chat_id, profile=profile
        )
        if prepared["text"] is not None:
            answer = prepared["text"]
        else:
            answer = self._generate(prepared["messages"], temperature=temperature)
            if prepared.get("identity") and not self._answer_uses_profile(
                answer, prepared.get("profile", "")
            ):
                answer = self._profile_fallback(prepared.get("profile", ""))
            elif prepared.get("recall") and prepared.get("memory") and self._is_refusal(answer):
                answer = self._recall_summary(prepared["memory"])
        return {
            "answer": answer,
            "sources": prepared["sources"],
            "tool": prepared["tool"],
            "used_rag": prepared["tool"] == "retrieve_documents",
            "needs_clarification": prepared["needs_clarification"],
            "time_taken": round(time.time() - start, 2),
        }

    def stream_query(
        self,
        user_query: str,
        user_id: str = "default_user",
        chat_history: Optional[List[Dict]] = None,
        temperature: float = 0.1,
        chat_id: str = "",
        profile: str = "",
    ):
        """Yield SSE-ready event dicts: status / sources / token / done."""
        start = time.time()
        yield {"type": "status", "data": "Understanding your question…"}
        try:
            prepared = self._prepare(
                user_query, user_id, chat_history, chat_id=chat_id, profile=profile
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Agent failed")
            yield {"type": "error", "data": f"Agent error: {exc}"}
            return

        yield {"type": "sources", "data": prepared["sources"]}

        if prepared["text"] is not None:
            # Clarification / direct statement: stream it word by word.
            for word in prepared["text"].split(" "):
                yield {"type": "token", "data": word + " "}
            yield {
                "type": "done",
                "data": {
                    "tool": prepared["tool"],
                    "used_rag": False,
                    "needs_clarification": prepared["needs_clarification"],
                    "time_taken": round(time.time() - start, 2),
                },
            }
            return

        # Identity questions: generate in full so we can guarantee the answer is
        # grounded in the user's profile.
        if prepared.get("identity"):
            answer = self._generate(prepared["messages"], temperature=temperature)
            if not self._answer_uses_profile(answer, prepared.get("profile", "")):
                answer = self._profile_fallback(prepared.get("profile", ""))
            for word in answer.split(" "):
                yield {"type": "token", "data": word + " "}
            yield {
                "type": "done",
                "data": {
                    "tool": "answer_directly",
                    "used_rag": False,
                    "needs_clarification": False,
                    "time_taken": round(time.time() - start, 2),
                },
            }
            return

        # Recall questions are short: generate in full so we can fall back to a
        # deterministic summary if the small model refuses to answer.
        if prepared.get("recall"):
            answer = self._generate(prepared["messages"], temperature=temperature)
            if prepared.get("memory") and self._is_refusal(answer):
                answer = self._recall_summary(prepared["memory"])
            for word in answer.split(" "):
                yield {"type": "token", "data": word + " "}
            yield {
                "type": "done",
                "data": {
                    "tool": "answer_directly",
                    "used_rag": False,
                    "needs_clarification": False,
                    "time_taken": round(time.time() - start, 2),
                },
            }
            return

        generator = self._stream_generate(prepared["messages"], temperature)
        try:
            for chunk in generator:
                yield {"type": "token", "data": chunk}
        except Exception as exc:  # noqa: BLE001
            logger.exception("Streaming failed")
            yield {"type": "error", "data": f"Generation error: {exc}"}
            return
        finally:
            # Ensure the underlying generation stops if the client goes away.
            generator.close()

        yield {
            "type": "done",
            "data": {
                "tool": prepared["tool"],
                "used_rag": prepared["tool"] == "retrieve_documents",
                "needs_clarification": False,
                "time_taken": round(time.time() - start, 2),
            },
        }

    # ------------------------------------------------------------------ #
    # Ingestion
    # ------------------------------------------------------------------ #
    def _guess_metadata(self, filename: str) -> Dict[str, str]:
        name_lower = filename.lower().replace(".pdf", "").replace(".md", "")
        parts = name_lower.split()
        company = next(
            (c for c in COMPANY_TERMS if c in name_lower), parts[0] if parts else "unknown"
        )
        if company == "alphabet":
            company = "google"
        year_match = re.search(r"\b(20\d{2})\b", name_lower)
        year = year_match.group(1) if year_match else "Unknown"
        if "10-k" in name_lower:
            doc_type = "10-K"
        elif "10-q" in name_lower:
            doc_type = "10-Q"
        elif "8-k" in name_lower:
            doc_type = "8-K"
        elif "annual" in name_lower or "report" in name_lower:
            doc_type = "Annual Report"
        else:
            doc_type = "General Document"
        return {"company": company, "year": year, "doc_type": doc_type}

    def ingest_pdf(
        self,
        filename: str,
        file_content: bytes,
        user_id: str = "global",
        storage_key: Optional[str] = None,
    ) -> Dict[str, object]:
        """Parse, chunk and embed a user-uploaded PDF into the knowledge base."""
        try:
            logger.info("Ingesting '%s' for user '%s'", filename, user_id)

            # Replace any previous version of the same file (shared knowledge).
            try:
                existing = self.collection.get(where={"source_file": filename})
                if existing.get("ids"):
                    self.collection.delete(ids=existing["ids"])
                    self._index_remove(existing["ids"])
                    logger.info("Replaced %d old chunks.", len(existing["ids"]))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not remove previous version: %s", exc)

            reader = PdfReader(io.BytesIO(file_content))
            if reader.is_encrypted:
                return {
                    "status": "error",
                    "message": "PDF is password protected and cannot be processed.",
                }

            meta_base = self._guess_metadata(filename)
            document_id = str(uuid.uuid4())
            timestamp = datetime.utcnow().isoformat()

            docs_to_add: List[str] = []
            metas_to_add: List[Dict] = []
            ids_to_add: List[str] = []

            for page_number, page in enumerate(reader.pages, start=1):
                page_text = (page.extract_text() or "").strip()
                if len(page_text) < 30:
                    continue
                for chunk in self.text_splitter.split_text(page_text):
                    chunk = chunk.strip()
                    if len(chunk) < 50:
                        continue
                    docs_to_add.append(chunk)
                    chunk_meta = {
                        **meta_base,
                        "source_file": filename,
                        "source_type": "pdf_upload",
                        "user_id": str(user_id),
                        "document_id": document_id,
                        "page_number": page_number,
                        "ingested_at": timestamp,
                        "status": "processed",
                    }
                    if storage_key:
                        chunk_meta["storage_key"] = storage_key
                    metas_to_add.append(chunk_meta)
                    ids_to_add.append(f"upload_{uuid.uuid4().hex}")

            if not docs_to_add:
                return {
                    "status": "error",
                    "message": (
                        "PDF contains too little extractable text. It might be a "
                        "scanned image."
                    ),
                }

            # Embed locally so retrieval always uses the same embedding model.
            embeddings = self.embedding_model.embed_documents(docs_to_add)
            self.collection.add(
                ids=ids_to_add,
                documents=docs_to_add,
                metadatas=metas_to_add,
                embeddings=embeddings,
            )
            self._index_add(ids_to_add, docs_to_add, metas_to_add)
            logger.info("Ingested %d chunks from '%s'.", len(docs_to_add), filename)
            return {
                "status": "success",
                "chunks_added": len(docs_to_add),
                "document_id": document_id,
            }
        except Exception as exc:  # noqa: BLE001
            logger.exception("Ingestion failed")
            return {"status": "error", "message": f"Failed to parse PDF: {exc}"}
