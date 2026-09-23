"""
Retriva knowledge-base ingestion.

Indexes the financial corpus under ``data/finance`` into ChromaDB:

* ``markdown/<company>/*.md``  -> ``source_type="text"``
* ``tables/<company>/<report>/*.md`` -> ``source_type="table"`` (page aware)

The script is idempotent: chunk ids are deterministic, so re-running it updates
existing chunks instead of duplicating them. Set ``CLEAR_DB=1`` to wipe the
collection first and rebuild from scratch.

Usage:
    python ingest.py                 # incremental
    $env:CLEAR_DB=1; python ingest.py  # full rebuild
"""

import os
import re
import sys

import app  # noqa: F401  (sets HF offline mode before any model import)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import Settings
from app.vector_store import build_collection

_settings = Settings()
DATA_DIR = os.getenv("DATA_DIR", "data/finance")
CHROMA_DB_PATH = _settings.CHROMA_DB_PATH
COLLECTION_NAME = _settings.COLLECTION_NAME

COMPANIES = ["amazon", "apple", "google", "alphabet", "meta", "microsoft", "nvidia", "tesla"]


def _company_from_name(name: str) -> str:
    lowered = name.lower()
    match = next((c for c in COMPANIES if c in lowered), None)
    if match == "alphabet":
        return "google"
    return match or lowered


def _period_from_name(name: str) -> tuple:
    lowered = name.lower()
    year_match = re.search(r"\b(20\d{2})\b", lowered)
    year = year_match.group(1) if year_match else "Unknown"
    if "10-k" in lowered:
        doc_type = "10-K"
    elif "10-q" in lowered:
        doc_type = "10-Q"
    elif "8-k" in lowered:
        doc_type = "8-K"
    else:
        doc_type = "Unknown"
    return year, doc_type


def _base_meta(company: str, year: str, doc_type: str, source_file: str,
               source_type: str, page_number=None) -> dict:
    meta = {
        "company": company,
        "year": year,
        "doc_type": doc_type,
        "source_file": source_file,
        "source_type": source_type,
        "user_id": "global",
        "status": "processed",
    }
    if page_number is not None:
        meta["page_number"] = page_number
    return meta


def collect_chunks(splitter):
    """Return parallel lists of (documents, metadatas, ids)."""
    documents, metadatas, ids = [], [], []
    counter = 0

    def add(text, meta, id_prefix):
        nonlocal counter
        text = (text or "").strip()
        if len(text) < 50:
            return
        documents.append(text)
        metadatas.append(meta)
        ids.append(f"{id_prefix}_{counter}")
        counter += 1

    # --- Markdown prose -------------------------------------------------
    md_base = os.path.join(DATA_DIR, "markdown")
    if os.path.isdir(md_base):
        for company_dir in sorted(os.listdir(md_base)):
            company_path = os.path.join(md_base, company_dir)
            if not os.path.isdir(company_path):
                continue
            print(f"  text  / {company_dir}")
            for filename in sorted(os.listdir(company_path)):
                if not filename.endswith(".md"):
                    continue
                with open(os.path.join(company_path, filename), "r", encoding="utf-8", errors="ignore") as fh:
                    text = fh.read()
                company = _company_from_name(company_dir)
                year, doc_type = _period_from_name(filename)
                meta = _base_meta(company, year, doc_type, filename, "text")
                prefix = f"md_{company}_{os.path.splitext(filename)[0].replace(' ', '_')}"
                for chunk in splitter.split_text(text):
                    add(chunk, dict(meta), prefix)

    # --- Tables (page aware, keep the report context) -------------------
    tbl_base = os.path.join(DATA_DIR, "tables")
    if os.path.isdir(tbl_base):
        for company_dir in sorted(os.listdir(tbl_base)):
            company_path = os.path.join(tbl_base, company_dir)
            if not os.path.isdir(company_path):
                continue
            company = _company_from_name(company_dir)
            for report in sorted(os.listdir(company_path)):
                report_path = os.path.join(company_path, report)
                if not os.path.isdir(report_path):
                    continue
                print(f"  table / {company_dir} / {report}")
                year, doc_type = _period_from_name(report)
                for filename in sorted(os.listdir(report_path)):
                    if not filename.endswith(".md"):
                        continue
                    with open(os.path.join(report_path, filename), "r", encoding="utf-8", errors="ignore") as fh:
                        content = fh.read()
                    page_match = re.search(r"\*\*Page:\*\*\s*(\d+)", content)
                    page = int(page_match.group(1)) if page_match else None
                    if report.lower().startswith(company):
                        source_file = f"{report} - {filename}"
                    else:
                        source_file = f"{company} {report} - {filename}"
                    meta = _base_meta(company, year, doc_type, source_file, "table", page)
                    prefix = (
                        f"tbl_{company}_{report.replace(' ', '_')}_"
                        f"{os.path.splitext(filename)[0]}"
                    )
                    add(content, meta, prefix)

    return documents, metadatas, ids


def main() -> int:
    import torch

    device = "cpu"
    requested = (_settings.DEVICE or "auto").strip().lower()
    if torch.cuda.is_available() and (requested in ("", "auto") or requested.startswith("cuda")):
        device = "cuda"
    print(f"Loading embedding model (local, device={device})…")
    embeddings = HuggingFaceEmbeddings(
        model_name=_settings.EMBEDDING_MODEL_NAME,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=150, length_function=len
    )

    collection = build_collection(_settings, COLLECTION_NAME, _settings.EMBEDDING_DIM)
    if os.environ.get("CLEAR_DB"):
        try:
            collection.clear()
            print("Cleared existing collection.")
        except Exception as exc:  # noqa: BLE001
            print(f"Nothing to clear: {exc}")

    print("Collecting chunks…")
    documents, metadatas, ids = collect_chunks(splitter)
    if not documents:
        print("No chunks found. Check the data/finance directory.")
        return 1

    print(f"Embedding {len(documents)} chunks…")
    vectors = embeddings.embed_documents(documents)

    batch = 500
    for start in range(0, len(documents), batch):
        end = start + batch
        collection.upsert(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
            embeddings=vectors[start:end],
        )
        print(f"  upserted {min(end, len(documents))}/{len(documents)}")

    print(
        f"Done. Collection now holds {collection.count()} chunks "
        f"(backend={_settings.VECTOR_BACKEND})."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
