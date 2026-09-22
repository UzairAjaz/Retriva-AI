import json
import os
import re
import traceback
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import List, Optional

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from app.auth import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    authenticate_user,
    create_access_token,
    get_current_user,
    get_password_hash,
)
from app.database import Chat, Message, SessionLocal, User, get_db
from app.rag_engine import RetrivaEngine

# --- Engine singleton (loaded once on startup) ---
rag_engine: Optional[RetrivaEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_engine
    print("Loading Retriva Engine (this can take a minute)…")
    try:
        rag_engine = RetrivaEngine()
        print("Retriva Engine loaded successfully.")
        _reindex_memory(rag_engine)
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to load Retriva Engine: {exc}")
        traceback.print_exc()
        raise
    yield
    rag_engine = None
    print("Retriva Engine shut down.")


def _reindex_memory(engine: RetrivaEngine) -> None:
    """Backfill per-chat long-term memory from stored messages.

    Memory is keyed by message id, so this only embeds messages that are not
    already indexed. That keeps restarts fast (no re-embedding the whole
    history) and avoids blocking startup on large databases.
    """
    try:
        existing_ids = set()
        try:
            existing_ids = set(engine.memory_collection.get().get("ids") or [])
        except Exception as exc:  # noqa: BLE001
            print(f"Memory index lookup skipped: {exc}")

        with SessionLocal() as db:
            rows = db.query(Message, Chat.user_id).join(Chat, Message.chat_id == Chat.id).all()

        added = 0
        for message, owner_id in rows:
            if message.id in existing_ids:
                continue
            if not (message.content or "").strip():
                continue
            engine.memory_add(
                message.content, message.role, str(owner_id), message.chat_id, message.id
            )
            added += 1
        print(f"Memory re-index: {added} new message(s) indexed ({len(rows)} total).")
    except Exception as exc:  # noqa: BLE001
        print(f"Memory re-index skipped: {exc}")


def get_engine() -> RetrivaEngine:
    if rag_engine is None:
        raise HTTPException(status_code=503, detail="Engine is still starting up.")
    return rag_engine


app = FastAPI(title="Retriva API", version="2.0.0", lifespan=lifespan)

allowed_origins = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://localhost:5173,http://localhost:3001,"
    "http://127.0.0.1:5173,http://127.0.0.1:3000",
).split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in allowed_origins if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB


# ===================== Pydantic schemas =====================
class UserCreate(BaseModel):
    email: str
    password: str


class UserResponse(BaseModel):
    id: int
    email: str
    bio: str = ""
    model_config = ConfigDict(from_attributes=True)


class ProfileUpdate(BaseModel):
    bio: str = ""


class Token(BaseModel):
    access_token: str
    token_type: str


class QueryRequest(BaseModel):
    query: str
    chat_id: Optional[str] = None
    temperature: float = 0.1


class ChatCreate(BaseModel):
    title: str = "New Conversation"


class ChatRename(BaseModel):
    title: str


# ===================== AUTH =====================
@app.post("/api/auth/signup", response_model=UserResponse)
def signup(user: UserCreate, db: Session = Depends(get_db)):
    email = user.email.strip().lower()
    if not email or len(user.password) < 4:
        raise HTTPException(
            status_code=400, detail="A valid email and a password of 4+ chars are required."
        )
    if db.query(User).filter(User.email == email).first():
        raise HTTPException(status_code=400, detail="Email already registered")
    new_user = User(email=email, hashed_password=get_password_hash(user.password))
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user


@app.post("/api/auth/login", response_model=Token)
def login(
    form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)
):
    user = authenticate_user(db, form_data.username.strip().lower(), form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return {
        "access_token": create_access_token(
            data={"sub": str(user.id)},
            expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
        ),
        "token_type": "bearer",
    }


@app.get("/api/users/me", response_model=UserResponse)
def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user


# ===================== PROFILE / BIO =====================
@app.get("/api/profile")
def get_profile(current_user: User = Depends(get_current_user)):
    return {"bio": current_user.bio or ""}


@app.put("/api/profile")
def update_profile(
    body: ProfileUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    user = db.query(User).filter(User.id == current_user.id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    user.bio = (body.bio or "").strip()[:4000]
    db.commit()
    return {"bio": user.bio}


# ===================== CHAT CRUD =====================
@app.get("/api/chats")
def get_chats(
    db: Session = Depends(get_db), current_user: User = Depends(get_current_user)
):
    chats = (
        db.query(Chat)
        .filter(Chat.user_id == current_user.id)
        .order_by(Chat.updated_at.desc())
        .all()
    )
    return [
        {"id": c.id, "title": c.title, "created_at": str(c.created_at)}
        for c in chats
    ]


@app.post("/api/chats")
def create_chat(
    body: ChatCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    new_chat = Chat(user_id=current_user.id, title=body.title or "New Conversation")
    db.add(new_chat)
    db.commit()
    db.refresh(new_chat)
    return {
        "id": new_chat.id,
        "title": new_chat.title,
        "created_at": str(new_chat.created_at),
    }


@app.get("/api/chats/{chat_id}/messages")
def get_messages(
    chat_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    chat = (
        db.query(Chat)
        .filter(Chat.id == chat_id, Chat.user_id == current_user.id)
        .first()
    )
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    messages = (
        db.query(Message).filter(Message.chat_id == chat_id).order_by(Message.created_at).all()
    )
    return [
        {
            "id": m.id,
            "role": m.role,
            "content": m.content,
            "sources": json.loads(m.sources) if m.sources else [],
            "created_at": str(m.created_at),
        }
        for m in messages
    ]


@app.put("/api/chats/{chat_id}")
def rename_chat(
    chat_id: str,
    body: ChatRename,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    chat = (
        db.query(Chat)
        .filter(Chat.id == chat_id, Chat.user_id == current_user.id)
        .first()
    )
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    chat.title = body.title or chat.title
    db.commit()
    return {"id": chat.id, "title": chat.title}


@app.delete("/api/chats/{chat_id}")
def delete_chat(
    chat_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    chat = (
        db.query(Chat)
        .filter(Chat.id == chat_id, Chat.user_id == current_user.id)
        .first()
    )
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    db.delete(chat)
    db.commit()
    # A chat's long-term memory is isolated, so drop it with the chat.
    try:
        get_engine().memory_delete_chat(chat_id)
    except HTTPException:
        pass
    return {"message": "Chat deleted"}


# ===================== STREAM & QUERY =====================
def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


@app.post("/api/stream")
async def stream_query(
    request: QueryRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    engine = get_engine()
    user_id = str(current_user.id)
    profile = (current_user.bio or "").strip()

    # 1. Resolve or create the chat.
    chat_id = request.chat_id
    if not chat_id:
        new_chat = Chat(user_id=current_user.id, title=(request.query[:40] or "New Conversation"))
        db.add(new_chat)
        db.commit()
        db.refresh(new_chat)
        chat_id = new_chat.id
    else:
        chat = (
            db.query(Chat)
            .filter(Chat.id == chat_id, Chat.user_id == current_user.id)
            .first()
        )
        if not chat:
            raise HTTPException(status_code=404, detail="Chat not found")
        if chat.title == "New Conversation":
            chat.title = request.query[:40] or chat.title
            db.commit()

    # 2. Recent history of *this* chat (before the new turn) as grounded context.
    recent = (
        db.query(Message)
        .filter(Message.chat_id == chat_id)
        .order_by(Message.created_at.desc())
        .limit(8)
        .all()
    )
    chat_history: List[dict] = [
        {"role": m.role, "content": m.content} for m in reversed(recent)
    ]

    # 3. Persist the user message and index it for this chat's long-term memory.
    user_message_id = str(uuid.uuid4())
    db.add(Message(id=user_message_id, chat_id=chat_id, role="user", content=request.query))
    chat_row = db.query(Chat).filter(Chat.id == chat_id).first()
    if chat_row:
        chat_row.updated_at = datetime.utcnow()
    db.commit()
    engine.memory_add(request.query, "user", user_id, chat_id, user_message_id)

    def generate():
        collected: List[str] = []
        sources: List[dict] = []
        try:
            yield _sse({"type": "meta", "data": {"chat_id": chat_id}})
            for event in engine.stream_query(
                request.query,
                user_id=user_id,
                chat_history=chat_history,
                temperature=request.temperature,
                chat_id=chat_id,
                profile=profile,
            ):
                if event["type"] == "sources":
                    sources = event.get("data") or []
                elif event["type"] == "token":
                    collected.append(event.get("data", ""))
                yield _sse(event)
        except Exception as exc:  # noqa: BLE001
            print("STREAM ERROR:\n", traceback.format_exc())
            yield _sse({"type": "error", "data": str(exc)})
        finally:
            answer = "".join(collected).strip()
            if answer:
                try:
                    assistant_id = str(uuid.uuid4())
                    with SessionLocal() as save_db:
                        save_db.add(
                            Message(
                                id=assistant_id,
                                chat_id=chat_id,
                                role="assistant",
                                content=answer,
                                sources=json.dumps(sources),
                            )
                        )
                        save_db.commit()
                    # Long-term memory is scoped to this chat.
                    engine.memory_add(
                        answer, "assistant", user_id, chat_id, assistant_id
                    )
                except Exception as save_err:  # noqa: BLE001
                    print(f"Failed to save assistant message: {save_err}")

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/query")
async def query_once(
    request: QueryRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Non-streaming convenience endpoint (used by the CLI / integrations)."""
    engine = get_engine()
    return engine.query(
        request.query,
        user_id=str(current_user.id),
        chat_id=request.chat_id or "",
        profile=(current_user.bio or "").strip(),
        temperature=request.temperature,
    )


# ===================== INGESTION =====================
def sanitize_filename(filename: str) -> str:
    filename = os.path.basename(filename or "document.pdf")
    return re.sub(r"[^\w\s.-]", "", filename).strip() or "document.pdf"


@app.post("/api/ingest")
async def ingest_document(
    file: UploadFile = File(...), current_user: User = Depends(get_current_user)
):
    engine = get_engine()
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext != ".pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    file_content = await file.read()
    if len(file_content) == 0:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(file_content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large. Max 20MB.")

    clean_name = sanitize_filename(file.filename)
    try:
        result = await run_in_threadpool(
            engine.ingest_pdf, clean_name, file_content, str(current_user.id)
        )
    except Exception as exc:  # noqa: BLE001
        print("INGEST ERROR:\n", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}")

    if result.get("status") == "success":
        return {
            "message": f"Processed '{clean_name}'",
            "chunks": result.get("chunks_added", 0),
            "document_id": result.get("document_id"),
        }
    raise HTTPException(status_code=422, detail=result.get("message", "Ingestion failed"))


@app.get("/health")
@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "engine_loaded": rag_engine is not None,
        "knowledge_chunks": len(rag_engine.id_to_doc) if rag_engine else 0,
    }


@app.get("/")
def root():
    return {"message": "Retriva API is running"}


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
