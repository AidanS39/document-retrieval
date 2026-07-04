from sqlalchemy import ForeignKey
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped, mapped_column, relationship
from typing import List, Optional
from pgvector.sqlalchemy import VECTOR


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
    image_failed: Mapped[bool] = mapped_column(
        default=False
    )  # whether page failed to convert to image
    embedding: Mapped[Optional[VECTOR]] = mapped_column(
        VECTOR(2048)
    )  # dense single vector embeddings for bi-encoder
    embedding_failed: Mapped[bool] = mapped_column(
        default=False
    )  # whether embedding failed to generate for page
    col_embeddings_failed: Mapped[bool] = mapped_column(
        default=False
    )  # whether col embeddings failed to generate for page
    gemini_page_embedding: Mapped[Optional["GeminiPageEmbedding"]] = relationship(
        back_populates="page"
    )


class GeminiPageEmbedding(Base):
    __tablename__ = "gemini_page_embedding"
    id: Mapped[int] = mapped_column(primary_key=True)
    page_id: Mapped[int] = mapped_column(
        ForeignKey("page.id", ondelete="CASCADE"), unique=True
    )
    page: Mapped["Page"] = relationship(back_populates="gemini_page_embedding")
    embedding: Mapped[Optional[VECTOR]] = mapped_column(VECTOR(1536))
