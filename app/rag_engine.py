import logging
import time
import chromadb
import torch
from typing import Dict, List, Optional
from sentence_transformers import CrossEncoder
from transformers import pipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from app.config import Settings

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RetrivaEngine:
    """
    Production-ready RAG Engine for Financial Documents.
    Handles Vector Search, Reranking, and Local LLM Generation.
    """
    
    def __init__(self):
        logger.info("Initializing Retriva Engine...")
        self.config = Settings()
        
        # Load components
        self.embedding_model = self._load_embedding_model()
        self.reranker = self._load_reranker()
        self.llm_pipeline = self._load_llm()
        self.chroma_client = self._init_chroma_db()
        
        logger.info("Retriva Engine initialized successfully.")

    def _load_embedding_model(self):
        logger.info(f"Loading Embedding Model: {self.config.EMBEDDING_MODEL_NAME}")
        return HuggingFaceEmbeddings(model_name=self.config.EMBEDDING_MODEL_NAME)

    def _load_reranker(self):
        logger.info(f"Loading Reranker Model: {self.config.RERANKER_MODEL_NAME}")
        return CrossEncoder(self.config.RERANKER_MODEL_NAME)

    def _load_llm(self):
        logger.info(f"Loading Local LLM: {self.config.LLM_MODEL_NAME}")
        dtype = getattr(torch, self.config.TORCH_DTYPE)
        return pipeline(
            "text-generation",
            model=self.config.LLM_MODEL_NAME,
            dtype=dtype,
            device_map=self.config.DEVICE_MAP,
            max_new_tokens=self.config.MAX_NEW_TOKENS
        )

    def _init_chroma_db(self):
        logger.info(f"Connecting to ChromaDB at: {self.config.CHROMA_DB_PATH}")
        client = chromadb.PersistentClient(path=self.config.CHROMA_DB_PATH)
        return client.get_collection(name=self.config.COLLECTION_NAME)

    def _build_metadata_filter(self, query: str) -> Optional[Dict]:
        """Dynamically builds ChromaDB $and filters based on query keywords."""
        conditions = []
        query_lower = query.lower()
        
        company_map = {"amazon": "amazon", "apple": "apple", "google": "google", "meta": "meta"}
        for key, value in company_map.items():
            if key in query_lower:
                conditions.append({"company": value})
                break # Only one company at a time
                
        year_map = {"2023": "2023", "2024": "2024", "2025": "2025"}
        for key, value in year_map.items():
            if key in query_lower:
                conditions.append({"year": value})
                break

        if len(conditions) == 0:
            return None
        elif len(conditions) == 1:
            return conditions[0]
        else:
            return {"$and": conditions}

    def _format_prompt(self, context: str, query: str) -> str:
        """Formats the prompt specifically for Qwen 2.5 Instruct."""
        return f"""<|im_start|>system
You are Retriva, an expert financial AI assistant. 
Answer the user's question based ONLY on the provided context. 
If the exact answer is not in the context, say exactly: "I cannot find this specific information in the provided documents."
Be concise, direct, and mention the company and year.
<|im_end|>
<|im_start|>user
Context:
{context}

Question: {query}
<|im_end|>
<|im_start|>assistant
"""

    def query(self, user_query: str) -> Dict[str, any]:
        """Main execution method."""
        start_time = time.time()
        logger.info(f"Processing query: '{user_query}'")

        # --- NATURAL BEHAVIOR: GREETING ROUTER ---
        query_lower = user_query.lower().strip()
        greetings = ["hi", "hello", "hey", "how are you"]
        if any(greet in query_lower for greet in greetings):
            return {
                "answer": "Hello! I am Retriva, your financial AI assistant. How can I help you with Amazon, Apple, Google, or Meta documents today?",
                "sources": [],
                "time_taken": 0.1
            }
        # ------------------------------------------

        # 1. Metadata Filtering
        where_clause = self._build_metadata_filter(user_query)
        
        # 2. Vector Search
        results = self.chroma_client.query(
            query_embeddings=[self.embedding_model.embed_query(user_query)],
            n_results=self.config.TOP_N_RETRIEVAL,
            where=where_clause
        )

        if not results['ids'][0]:
            return {"answer": "No documents found matching the criteria.", "sources": [], "time_taken": 0.0}

        # 3. Reranking
        pairs = [[user_query, doc] for doc in results['documents'][0]]
        scores = self.reranker.predict(pairs)
        
        ranked_results = sorted([
            {
                'score': float(scores[i]), 
                'text': results['documents'][0][i], 
                'meta': results['metadatas'][0][i]
            }
            for i in range(len(results['ids'][0]))
        ], key=lambda x: x['score'], reverse=True)

        # 4. Context Preparation (Top N Reranked)
        context_text = "\n\n".join([
            f"[Source {i+1}: {res['meta'].get('company', 'Unknown').upper()} {res['meta'].get('year', 'Unknown')} ({res['meta'].get('source_type', 'Unknown')})]\n{res['text'][:600]}"
            for i, res in enumerate(ranked_results[:self.config.TOP_N_RERANKED])
        ])

        # 5. LLM Generation
        prompt = self._format_prompt(context_text, user_query)
        
        try:
            response = self.llm_pipeline(
                prompt, 
                max_new_tokens=self.config.MAX_NEW_TOKENS, 
                temperature=self.config.TEMPERATURE, 
                do_sample=True, 
                pad_token_id=151643
            )
            full_text = response[0]['generated_text']
            answer = full_text.split("<|im_start|>assistant")[-1].replace("<|im_end|>", "").strip()
        except Exception as e:
            logger.error(f"LLM Generation failed: {e}")
            answer = "Error generating response from the local model."

        end_time = time.time()
        
        # 6. Return structured response (Perfect for future UI/API)
        return {
            "answer": answer,
            "sources": [
                {
                    "company": res['meta'].get('company', 'Unknown'),
                    "year": res['meta'].get('year', 'Unknown'),
                    "type": res['meta'].get('source_type', 'Unknown'),
                    "relevance_score": res['score']
                }
                for res in ranked_results[:self.config.TOP_N_RERANKED]
            ],
            "time_taken": round(end_time - start_time, 2)
        }