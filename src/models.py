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
    embedding: Mapped[Optional[VECTOR]] = mapped_column(VECTOR(768)) # sparse embeddings, either bm25 or tf-idf
    pages: Mapped[List["Page"]] = relationship(back_populates="document", cascade="all, delete-orphan")

class Page(Base):
    __tablename__ = "page"
    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("document.id", ondelete="CASCADE"))
    document: Mapped["Document"] = relationship(back_populates="pages")
    image_path: Mapped[str]
    number: Mapped[int]
    embedding: Mapped[Optional[VECTOR]] = mapped_column(VECTOR(768)) # dense embeddings for bi-encoder
