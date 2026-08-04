from sqlalchemy import ForeignKey, DateTime, Index, UniqueConstraint, func
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped, mapped_column, relationship
from typing import List, Optional
from pgvector.sqlalchemy import HALFVEC, VECTOR
from datetime import datetime


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "document"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str]
    path: Mapped[str] = mapped_column(unique=True)
    text: Mapped[Optional[str]]
    text_failed: Mapped[bool] = mapped_column(
        default=False
    )  # whether doc failed to convert to text
    embedding: Mapped[Optional[VECTOR]] = mapped_column(
        VECTOR(768)
    )  # sparse single vector embeddings, either bm25 or tf-idf
    embedding_failed: Mapped[bool] = mapped_column(
        default=False
    )  # whether embedding failed to generate for doc
    pages: Mapped[List["Page"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class Page(Base):
    __tablename__ = "page"
    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(
        ForeignKey("document.id", ondelete="CASCADE")
    )
    document: Mapped["Document"] = relationship(back_populates="pages")
    image_path: Mapped[str] = mapped_column(unique=True)
    number: Mapped[int]
    text: Mapped[Optional[str]]
    text_failed: Mapped[bool] = mapped_column(default=False)
    qwen3_2b_embedding: Mapped[Optional[HALFVEC]] = mapped_column(HALFVEC(2048))
    qwen3_8b_embedding: Mapped[Optional[HALFVEC]] = mapped_column(HALFVEC(4096))
    gemini_embedding: Mapped[Optional[VECTOR]] = mapped_column(VECTOR(1536))
    __table_args__ = (
        Index(
            "ix_page_qwen3_2b_embedding_hnsw",
            "qwen3_2b_embedding",
            postgresql_using="hnsw",
            postgresql_ops={"qwen3_2b_embedding": "halfvec_cosine_ops"},
        ),
        Index(
            "ix_page_gemini_embedding_hnsw",
            "gemini_embedding",
            postgresql_using="hnsw",
            postgresql_ops={"gemini_embedding": "vector_cosine_ops"},
        ),
    )



class EvaluationAnnotation(Base):
    __tablename__ = "evaluation_annotation"
    id: Mapped[int] = mapped_column(primary_key=True)
    annotator: Mapped[str]
    query_id: Mapped[int]
    query: Mapped[str]
    page_id: Mapped[int]
    score: Mapped[int]
    submitted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    __table_args__ = (
        UniqueConstraint("annotator", "query_id", "page_id", name="uq_annotation"),
    )


class EvaluationNote(Base):
    __tablename__ = "evaluation_note"
    id: Mapped[int] = mapped_column(primary_key=True)
    annotator: Mapped[str]
    query_id: Mapped[int]
    note: Mapped[str] = mapped_column(default="")
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    __table_args__ = (
        UniqueConstraint("annotator", "query_id", name="uq_note"),
    )
