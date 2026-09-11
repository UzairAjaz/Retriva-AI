class Settings:
    # Database Settings
    CHROMA_DB_PATH: str = "./retriva_chroma_db"
    COLLECTION_NAME: str = "retriva_financial_docs"
    
    # Model Settings
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    RERANKER_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    LLM_MODEL_NAME: str = "Qwen/Qwen2.5-1.5B-Instruct"
    
    # Inference Settings
    MAX_NEW_TOKENS: int = 150
    TEMPERATURE: float = 0.1
    TOP_N_RETRIEVAL: int = 10      # Vector search se kitne chunks layein
    TOP_N_RERANKED: int = 3        # Rerank karne ke baad LLM ko kitne dein
    
    # Device Settings
    DEVICE_MAP: str = "auto"       # "auto" for GPU if available, else CPU
    TORCH_DTYPE: str = "float32"   # "float16" or "bfloat16" if using GPU