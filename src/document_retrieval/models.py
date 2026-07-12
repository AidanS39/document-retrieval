from sqlalchemy import ForeignKey, DateTime, func
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped, mapped_column, relationship
from typing import List, Optional
from pgvector.sqlalchemy import VECTOR
from datetime import datetime


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "document"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str]
    path: Mapped[str]
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
    image_path: Mapped[str]
    number: Mapped[int]
    embedding: Mapped[Optional[VECTOR]] = mapped_column(VECTOR(2048))
    gemini_embedding: Mapped[Optional[VECTOR]] = mapped_column(VECTOR(1536))


class NSTXPaper(Base):
    __tablename__ = "nstx_papers"
    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    drive_file_id: Mapped[Optional[str]] = mapped_column(index=True, unique=True)
    original_filename: Mapped[Optional[str]]
    subfolder: Mapped[Optional[str]]
    title: Mapped[Optional[str]]
    authors: Mapped[Optional[str]]
    journal: Mapped[Optional[str]]
    publication_date: Mapped[Optional[str]]
    doi: Mapped[Optional[str]] = mapped_column(index=True)
    abstract: Mapped[Optional[str]]
    key_findings: Mapped[Optional[str]]
    experiment_type: Mapped[Optional[str]]
    document_type: Mapped[Optional[str]]
    page_count: Mapped[Optional[int]]
    nstx_embeddings: Mapped[List["NSTXEmbedding"]] = relationship(
        back_populates="paper", cascade="all, delete-orphan"
    )


class NSTXEmbedding(Base):
    __tablename__ = "nstx_embeddings"
    id: Mapped[int] = mapped_column(primary_key=True)
    paper_id: Mapped[int] = mapped_column(ForeignKey("nstx_papers.id", ondelete="CASCADE"))
    paper: Mapped["NSTXPaper"] = relationship(
        back_populates="nstx_embeddings",   
    )
    content_type: Mapped[str]
    chunk_index: Mapped[Optional[int]]
    page_number: Mapped[Optional[int]]
    figure_id: Mapped[Optional[str]]
    content: Mapped[Optional[str]]
    image_uri: Mapped[Optional[str]]
    section: Mapped[Optional[str]]
    page_start: Mapped[Optional[int]]
    page_end: Mapped[Optional[int]]
    embedding: Mapped[Optional[VECTOR]] = mapped_column(VECTOR(1536))
    created_at: Mapped[Optional[datetime]] = mapped_column(DateTime, server_default=func.now())
