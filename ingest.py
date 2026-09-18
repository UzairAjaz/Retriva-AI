import os
import re
import chromadb
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings

# 1. Configuration
DATA_DIR = "data/finance"
CHROMA_DB_PATH = "./retriva_chroma_db"

print("Loading Embedding Model (This might take a minute on first run)...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
semantic_splitter = SemanticChunker(embeddings, breakpoint_threshold_type="percentile", breakpoint_threshold_amount=90)

print("Connecting to Persistent Vector Database...")
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = client.get_or_create_collection(name="retriva_financial_docs")

# 2. Helper Function: Metadata Extractor
def get_metadata(filename, source_type, page_num=None, user_id="default_user"):
    name_lower = filename.lower().replace(".md", "").replace(".pdf", "")
    company = name_lower.split()[0]
    year_match = re.search(r'\b(20\d{2})\b', name_lower)
    year = year_match.group(1) if year_match else "Unknown"
    
    doc_type = "Unknown"
    if "10-k" in name_lower: doc_type = "10-K"
    elif "10-q" in name_lower: doc_type = "10-Q"
    elif "8-k" in name_lower: doc_type = "8-K"
        
    meta = {
        "company": company, 
        "year": year, 
        "doc_type": doc_type, 
        "source_file": filename,
        "source_type": source_type,
        "user_id": user_id          # <--- CRITICAL: Added for multi-tenancy
    }
    if page_num:
        meta["page_number"] = page_num
    return meta

# 3. Main Ingestion Logic
all_chunks, all_metadatas, all_ids = [], [], []
chunk_id_counter = 0

print("\nStarting Actual Data Ingestion...\n")

# --- PART A: Process Markdown Files (Text) ---
md_base = os.path.join(DATA_DIR, "markdown")
if os.path.exists(md_base):
    for company in os.listdir(md_base):
        company_path = os.path.join(md_base, company)
        if os.path.isdir(company_path):
            print(f"📂 Processing Text for: {company}")
            for file in os.listdir(company_path):
                if file.endswith(".md"):
                    with open(os.path.join(company_path, file), 'r', encoding='utf-8') as f:
                        text = f.read()
                    chunks = semantic_splitter.split_text(text)
                    for chunk in chunks:
                        if len(chunk.strip()) > 50:
                            all_chunks.append(chunk)
                            all_metadatas.append(get_metadata(file, "text", user_id="default_user"))
                            all_ids.append(f"chunk_{chunk_id_counter}")
                            chunk_id_counter += 1

# --- PART B: Process Table Files ---
tbl_base = os.path.join(DATA_DIR, "tables")
if os.path.exists(tbl_base):
    for company in os.listdir(tbl_base):
        company_path = os.path.join(tbl_base, company)
        if os.path.isdir(company_path):
            print(f"📂 Processing Tables for: {company}")
            for folder in os.listdir(company_path):
                folder_path = os.path.join(company_path, folder)
                if os.path.isdir(folder_path):
                    for file in os.listdir(folder_path):
                        if file.endswith(".md"):
                            with open(os.path.join(folder_path, file), 'r', encoding='utf-8') as f:
                                table_content = f.read()
                            page_match = re.search(r'\*\*Page:\*\*\s*(\d+)', table_content)
                            page = int(page_match.group(1)) if page_match else 0
                            
                            all_chunks.append(table_content)
                            all_metadatas.append(get_metadata(file, "table", page, user_id="default_user"))
                            all_ids.append(f"chunk_{chunk_id_counter}")
                            chunk_id_counter += 1

# 4. Save to Vector Database
print(f"\n⚙️ Generating embeddings for {len(all_chunks)} chunks... (Please wait)")
collection.add(documents=all_chunks, metadatas=all_metadatas, ids=all_ids)
print(f"\nSUCCESS! Retriva has successfully ingested {len(all_chunks)} chunks.")
print(f"Database saved at: {CHROMA_DB_PATH}")