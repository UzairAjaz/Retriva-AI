import uuid
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import sessionmaker, relationship, declarative_base

# SQLite database file (automatically created)
SQLALCHEMY_DATABASE_URL = "sqlite:///./retriva.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Use the modern SQLAlchemy 2.0 import
Base = declarative_base()

# --- User Model ---
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    # Free-form profile the user writes once and which is visible in every chat.
    bio = Column(Text, default="", nullable=False, server_default="")
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationship to chats
    chats = relationship("Chat", back_populates="user", cascade="all, delete-orphan")

# --- Chat Model ---
class Chat(Base):
    __tablename__ = "chats"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # 🌟 FIX: Changed from String to Integer to match User.id
    user_id = Column(Integer, ForeignKey("users.id"), index=True) 
    
    title = Column(String, default="New Conversation")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    user = relationship("User", back_populates="chats")
    messages = relationship("Message", back_populates="chat", cascade="all, delete-orphan")

# --- Message Model ---
class Message(Base):
    __tablename__ = "messages"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    chat_id = Column(String, ForeignKey("chats.id"))
    role = Column(String)  # 'user' or 'assistant'
    content = Column(Text)
    sources = Column(Text, nullable=True)  # Store sources as JSON string
    created_at = Column(DateTime, default=datetime.utcnow)
    
    chat = relationship("Chat", back_populates="messages")

# Create database tables
Base.metadata.create_all(bind=engine)


def _migrate_schema() -> None:
    """Add columns that were introduced after a database was first created."""
    with engine.begin() as conn:
        columns = {
            row[1]
            for row in conn.exec_driver_sql("PRAGMA table_info(users)").fetchall()
        }
        if "bio" not in columns:
            conn.exec_driver_sql(
                "ALTER TABLE users ADD COLUMN bio TEXT NOT NULL DEFAULT ''"
            )


_migrate_schema()


# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()