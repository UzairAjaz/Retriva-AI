"""
Database layer.

The connection is driven entirely by ``DATABASE_URL`` so the same code runs on
SQLite (local development) or PostgreSQL (production / AWS RDS / Azure Database
for PostgreSQL):

    sqlite:///./retriva.db
    postgresql+psycopg2://user:password@host:5432/retriva

No host, port or file path is hard-coded.
"""

import uuid
from datetime import datetime
from urllib.parse import urlsplit

from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    inspect,
    text,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

from app.config import Settings

settings = Settings()


def normalize_database_url(url: str) -> str:
    """Accept the common managed-database URL spellings.

    ``postgres://``      (Heroku/Render style) -> keep the psycopg2 driver
    ``postgresql://``    (SQLAlchemy default)  -> keep the psycopg2 driver
    """
    if not url:
        return "sqlite:///./retriva.db"
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+psycopg2://", 1)
    elif url.startswith("postgresql://") and "+psycopg2" not in url:
        url = url.replace("postgresql://", "postgresql+psycopg2://", 1)
    return url


DATABASE_URL = normalize_database_url(settings.DATABASE_URL)
IS_SQLITE = DATABASE_URL.startswith("sqlite")


def _engine_kwargs() -> dict:
    if IS_SQLITE:
        # SQLite is single-file / single-process: only needs thread sharing.
        return {"connect_args": {"check_same_thread": False}}
    # PostgreSQL (RDS/Aurora/Cloud SQL/Azure PG): pooled + resilient.
    return {
        "pool_pre_ping": True,
        "pool_size": settings.DB_POOL_SIZE,
        "max_overflow": settings.DB_MAX_OVERFLOW,
        "pool_recycle": settings.DB_POOL_RECYCLE,
    }


engine = create_engine(DATABASE_URL, echo=settings.DB_ECHO, **_engine_kwargs())
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


# --- Models -----------------------------------------------------------------
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    # Free-form profile the user writes once and which is visible in every chat.
    bio = Column(Text, default="", nullable=False, server_default="")
    created_at = Column(DateTime, default=datetime.utcnow)

    chats = relationship("Chat", back_populates="user", cascade="all, delete-orphan")


class Chat(Base):
    __tablename__ = "chats"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    title = Column(String, default="New Conversation")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", back_populates="chats")
    messages = relationship("Message", back_populates="chat", cascade="all, delete-orphan")


class Message(Base):
    __tablename__ = "messages"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    chat_id = Column(String, ForeignKey("chats.id"), index=True)
    role = Column(String)  # 'user' or 'assistant'
    content = Column(Text)
    sources = Column(Text, nullable=True)  # sources as a JSON string
    created_at = Column(DateTime, default=datetime.utcnow)

    chat = relationship("Chat", back_populates="messages")


Base.metadata.create_all(bind=engine)


def _migrate_schema() -> None:
    """Add columns introduced after a database was first created.

    Works on both SQLite and PostgreSQL by inspecting the existing columns
    before issuing a plain ``ALTER TABLE``.
    """
    inspector = inspect(engine)
    if "users" not in inspector.get_table_names():
        return
    existing = {column["name"] for column in inspector.get_columns("users")}
    if "bio" not in existing:
        with engine.begin() as conn:
            conn.execute(
                text("ALTER TABLE users ADD COLUMN bio TEXT NOT NULL DEFAULT ''")
            )


_migrate_schema()


def describe_database() -> str:
    """A safe (credential-free) description for logs."""
    if IS_SQLITE:
        return DATABASE_URL
    parts = urlsplit(DATABASE_URL)
    return f"{parts.scheme}://{parts.hostname}:{parts.port or ''}{parts.path}"


# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
