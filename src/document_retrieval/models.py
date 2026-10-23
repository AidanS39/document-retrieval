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
    annotations: Mapped[List["EvaluationAnnotation"]] = relationship(back_populates="page")
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

class Annotator(Base):
    __tablename__ = "annotator"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(unique=True, index=True)
    role: Mapped[str]
    annotations: Mapped[List["EvaluationAnnotation"]] = relationship(back_populates="annotator")
    notes: Mapped[List["EvaluationNote"]] = relationship(back_populates="annotator")

class Query(Base):
    __tablename__ = "query"
    id: Mapped[int] = mapped_column(primary_key=True)
    text: Mapped[str]
    active: Mapped[bool]
    annotations: Mapped[List["EvaluationAnnotation"]] = relationship(back_populates="query")
    notes: Mapped[List["EvaluationNote"]] = relationship(back_populates="query")

class EvaluationAnnotation(Base):
    __tablename__ = "evaluation_annotation"
    id: Mapped[int] = mapped_column(primary_key=True)
    annotator_id: Mapped[int] = mapped_column(ForeignKey("annotator.id"))
    annotator: Mapped[Annotator] = relationship(back_populates="annotations")
    query_id: Mapped[int] = mapped_column(ForeignKey("query.id"))
    query: Mapped[Query] = relationship(back_populates="annotations")
    page_id: Mapped[int] = mapped_column(ForeignKey("page.id"))
    page: Mapped[Page] = relationship(back_populates="annotations")
    score: Mapped[int]
    explanation: Mapped[str] = mapped_column(server_default="")
    submitted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    __table_args__ = (
        UniqueConstraint("annotator_id", "query_id", "page_id", name="uq_annotation"),
    )

class EvaluationNote(Base):
    __tablename__ = "evaluation_note"
    id: Mapped[int] = mapped_column(primary_key=True)
    annotator_id: Mapped[int] = mapped_column(ForeignKey("annotator.id"))
    annotator: Mapped[Annotator] = relationship(back_populates="notes")
    query_id: Mapped[int] = mapped_column(ForeignKey("query.id"))
    query: Mapped[Query] = relationship(back_populates="notes")
    note: Mapped[str] = mapped_column(default="")
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    __table_args__ = (
        UniqueConstraint("annotator_id", "query_id", name="uq_note"),
    )
