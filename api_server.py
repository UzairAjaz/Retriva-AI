import json
import os
import re
import traceback
import logging
from threading import Thread, Event, BoundedSemaphore
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, Depends, status, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Literal
from sqlalchemy.orm import Session, selectinload
from fastapi.responses import StreamingResponse
from app.rag_engine import RetrivaEngine
from app.database import get_db, SessionLocal, User, Chat, Message, UserMemory
from app.memory_manager import MemoryManager
from app.auth import (
    get_password_hash, create_access_token, authenticate_user, get_current_user, ACCESS_TOKEN_EXPIRE_MINUTES
)
import uvicorn

app = FastAPI(title="Retriva API - Production Ready")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://localhost:3001"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

print("🔄 Loading Retriva Engine...")
try:
    engine = RetrivaEngine()
    print("✅ Engine Loaded Successfully!")
except Exception as e:
    print(f"❌ Failed to load engine: {e}")
    raise

logger = logging.getLogger(__name__)
_memory_slots = BoundedSemaphore(4)


def start_memory_extraction(user_id: int, user_query: str):
    """Start bounded best-effort work; caller releases it after answer generation."""
    ready = Event()
    if not user_query.strip() or not _memory_slots.acquire(blocking=False):
        return ready

    def extract():
        try:
            # Never share the request's SQLAlchemy session with another thread.
            if ready.wait(timeout=300):
                with SessionLocal() as db:
                    MemoryManager(db, user_id).extract_and_save_facts(user_query)
        except Exception:
            logger.warning("Background memory extraction failed")
        finally:
            _memory_slots.release()

    try:
        Thread(target=extract, name="retriva-memory", daemon=True).start()
    except Exception:
        _memory_slots.release()
        logger.warning("Could not start memory worker")
    return ready


# --- Security Constants ---
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB Limit
ALLOWED_MIME_TYPES = {"application/pdf"}

def sanitize_filename(filename: str) -> str:
    filename = os.path.basename(filename)
    filename = re.sub(r'[^\w\s\.\-]', '', filename)
    return filename.lower().strip()

class UserCreate(BaseModel):
    email: str
    password: str

class UserResponse(BaseModel):
    id: int
    email: str
    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str

class QueryRequest(BaseModel):
    query: str
    chat_history: list = []
    temperature: float = 0.1

@app.post("/api/auth/signup", response_model=UserResponse)
def signup(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    new_user = User(email=user.email, hashed_password=get_password_hash(user.password))
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user

@app.post("/api/auth/login", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=401, detail="Incorrect email or password", headers={"WWW-Authenticate": "Bearer"})
    return {"access_token": create_access_token(data={"sub": str(user.id)}, expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)), "token_type": "bearer"}

@app.get("/api/users/me", response_model=UserResponse)
def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user

class MessagePayload(BaseModel):
    role: Literal["user", "assistant", "system"]
    text: str
    sources: str = "[]"
    created_at: Optional[datetime] = None

    @field_validator("sources")
    @classmethod
    def validate_sources(cls, value):
        if not isinstance(json.loads(value), list):
            raise ValueError("sources must be a JSON array encoded as a string")
        return value


class ChatPayload(BaseModel):
    id: Optional[int] = Field(default=None, gt=0)
    title: str
    messages: List[MessagePayload]


def serialize_chat(chat):
    return {
        "id": chat.id, "title": chat.title, "created_at": chat.created_at,
        "messages": [
            {"id": message.id, "role": message.role, "text": message.text,
             "sources": message.sources, "created_at": message.created_at}
            for message in chat.messages
        ],
    }


@app.get("/api/chats")
def list_chats(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    chats = db.query(Chat).options(selectinload(Chat.messages)).filter(
        Chat.user_id == current_user.id
    ).order_by(Chat.created_at.desc(), Chat.id.desc()).all()
    return [serialize_chat(chat) for chat in chats]


@app.post("/api/chats")
def save_chat(payload: ChatPayload, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if payload.id is None:
        chat = Chat(user_id=current_user.id, title=payload.title)
        db.add(chat)
    else:
        chat = db.query(Chat).filter(
            Chat.id == payload.id, Chat.user_id == current_user.id
        ).first()
        if chat is None:
            raise HTTPException(status_code=404, detail="Chat not found")
    try:
        chat.title = payload.title
        # Replace the full snapshot atomically; never trust incoming message IDs.
        chat.messages = [Message(
            role=message.role, text=message.text, sources=message.sources,
            created_at=message.created_at or datetime.utcnow()
        ) for message in payload.messages]
        db.commit()
        db.refresh(chat)
        return serialize_chat(chat)
    except Exception:
        db.rollback()
        raise


@app.delete("/api/chats/{chat_id}", status_code=204)
def delete_chat(chat_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    chat = db.query(Chat).filter(
        Chat.id == chat_id, Chat.user_id == current_user.id
    ).first()
    if chat is None:
        raise HTTPException(status_code=404, detail="Chat not found")
    try:
        db.delete(chat)
        db.commit()
    except Exception:
        db.rollback()
        raise


@app.get("/api/memories")
def list_memories(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    memories = db.query(UserMemory).filter(
        UserMemory.user_id == current_user.id
    ).order_by(UserMemory.created_at.desc(), UserMemory.id.desc()).all()
    return [{"id": memory.id, "content": memory.content, "created_at": memory.created_at}
            for memory in memories]


@app.delete("/api/memories/{memory_id}", status_code=204)
def delete_memory(memory_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    memory = db.query(UserMemory).filter(
        UserMemory.id == memory_id, UserMemory.user_id == current_user.id
    ).first()
    if memory is None:
        raise HTTPException(status_code=404, detail="Memory not found")
    try:
        db.delete(memory)
        db.commit()
    except Exception:
        db.rollback()
        raise


@app.post("/api/query")
async def query_documents(request: QueryRequest, current_user: User = Depends(get_current_user)):
    memory_ready = start_memory_extraction(current_user.id, request.query)
    try:
        result = engine.query(request.query, user_id=str(current_user.id), chat_history=request.chat_history, temperature=request.temperature)
        
        safe_sources = []
        for src in result.get("sources", []):
            safe_sources.append({
                "company": str(src.get("company", "Unknown")),
                "year": str(src.get("year", "Unknown")),
                "type": str(src.get("type", "Unknown")),
                "source_type": str(src.get("source_type", "text")),
                "source_file": str(src.get("source_file", "Unknown")),
                "page_number": str(src.get("page_number", "Unknown")),
                "relevance_score": float(src.get("relevance_score", 0.0)),
                "text": str(src.get("text", ""))
            })
            
        return {
            "answer": str(result.get("answer", "")),
            "sources": safe_sources,
            "time_taken": float(result.get("time_taken", 0.0)),
            "used_rag": bool(result.get("used_rag", False))
        }
    except Exception as e:
        print("❌ BACKEND ERROR:\n", traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        memory_ready.set()

@app.post("/api/ingest")
async def ingest_document(file: UploadFile = File(...), current_user: User = Depends(get_current_user)):
    try:
        if file.content_type not in ALLOWED_MIME_TYPES:
            raise HTTPException(status_code=400, detail="Only PDF files (application/pdf) are allowed.")
        
        file_content = await file.read()
        if len(file_content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="File too large. Maximum size is 10MB.")
        if len(file_content) == 0:
            raise HTTPException(status_code=400, detail="File is empty.")
            
        clean_name = sanitize_filename(file.filename)
        if not clean_name.endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Invalid file extension.")

        result = engine.ingest_pdf(clean_name, file_content, user_id=str(current_user.id))
        
        if result["status"] == "success":
            return {
                "message": f"Successfully processed '{clean_name}'", 
                "chunks": result["chunks_added"],
                "document_id": result.get("document_id")
            }
        else:
            raise HTTPException(status_code=422, detail=result.get("message", "Ingestion failed"))
            
    except HTTPException:
        raise
    except Exception as e:
        print("❌ INGESTION ERROR:\n", traceback.format_exc())
        raise HTTPException(status_code=500, detail="Internal server error during ingestion.")

@app.post("/api/stream")
async def stream_query(request: QueryRequest, current_user: User = Depends(get_current_user)):
    memory_ready = start_memory_extraction(current_user.id, request.query)
    try:
        # First, get the full result (with sources)
        result = engine.query(
            request.query, 
            user_id=str(current_user.id), 
            chat_history=request.chat_history,
            temperature=request.temperature
        )
        
        answer = result.get("answer", "")
        sources = result.get("sources", [])
        
        # Create a generator that streams the answer word by word
        def generate():
            import json
            import time
            
            # First, send sources as JSON
            sources_json = json.dumps(sources)
            yield f"data: {sources_json}\n\n"
            time.sleep(0.1)  # Small delay to ensure sources arrive first
            
            # Then stream the answer word by word
            words = answer.split(' ')
            for i, word in enumerate(words):
                yield f"data: {word}\n\n"
                if i < len(words) - 1:
                    yield "data:  \n\n"  # Space between words
                    time.sleep(0.02)  # Typing effect delay
        
        return StreamingResponse(generate(), media_type="text/event-stream")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        memory_ready.set()
    
@app.get("/")
def root():
    return {"message": "Retriva API is running"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)