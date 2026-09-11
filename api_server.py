from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.rag_engine import RetrivaEngine
import uvicorn
import traceback

app = FastAPI(title="Retriva API")

# CORS Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Engine
print(" Loading Retriva Engine...")
try:
    engine = RetrivaEngine()
    print("Engine Loaded Successfully!")
except Exception as e:
    print(f"Failed to load engine: {e}")
    raise

class QueryRequest(BaseModel):
    query: str

@app.post("/api/query")
async def query_documents(request: QueryRequest):
    try:
        result = engine.query(request.query)
        
        # CRITICAL FIX: Force convert everything to native Python types for JSON
        safe_sources = []
        for src in result.get("sources", []):
            safe_sources.append({
                "company": str(src.get("company", "Unknown")),
                "year": str(src.get("year", "Unknown")),
                "type": str(src.get("type", "Unknown")),
                "relevance_score": float(src.get("relevance_score", 0.0)) # Force native float
            })
            
        return {
            "answer": str(result.get("answer", "")),
            "sources": safe_sources,
            "time_taken": float(result.get("time_taken", 0.0))
        }
        
    except Exception as e:
        # Print exact error in backend terminal
        print("❌ BACKEND ERROR:")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"message": "Retriva API is running"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)