"""baseline

Revision ID: 301e6edf2c5b
Revises:
Create Date: 2026-07-03 19:17:42.581805

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

import pgvector


# revision identifiers, used by Alembic.
revision: str = "301e6edf2c5b"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    existing_tables = inspector.get_table_names()

    if "document" not in existing_tables:
        op.create_table(
            "document",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("name", sa.String(), nullable=False),
            sa.Column("path", sa.String(), nullable=False),
            sa.Column("text", sa.String(), nullable=True),
            sa.Column(
                "embedding", pgvector.sqlalchemy.vector.VECTOR(dim=768), nullable=True
            ),
            sa.PrimaryKeyConstraint("id"),
        )

    doc_columns = (
        {col["name"] for col in inspector.get_columns("document")}
        if "document" in existing_tables
        else set()
    )
    if "text_failed" not in doc_columns:
        op.add_column(
            "document",
            sa.Column(
                "text_failed",
                sa.Boolean(),
                nullable=False,
                server_default=sa.text("false"),
            ),
        )
    if "embedding_failed" not in doc_columns:
        op.add_column(
            "document",
            sa.Column(
                "embedding_failed",
                sa.Boolean(),
                nullable=False,
                server_default=sa.text("false"),
            ),
        )

    if "page" not in existing_tables:
        op.create_table(
            "page",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("document_id", sa.Integer(), nullable=False),
            sa.Column("image_path", sa.String(), nullable=False),
            sa.Column("number", sa.Integer(), nullable=False),
            sa.Column(
                "embedding", pgvector.sqlalchemy.vector.VECTOR(dim=2048), nullable=True
            ),
            sa.ForeignKeyConstraint(
                ["document_id"], ["document.id"], ondelete="CASCADE"
            ),
            sa.PrimaryKeyConstraint("id"),
        )

    page_columns = (
        {col["name"] for col in inspector.get_columns("page")}
        if "page" in existing_tables
        else set()
    )
    if "image_failed" not in page_columns:
        op.add_column(
            "page",
            sa.Column(
                "image_failed",
                sa.Boolean(),
                nullable=False,
                server_default=sa.text("false"),
            ),
        )
    if "embedding_failed" not in page_columns:
        op.add_column(
            "page",
            sa.Column(
                "embedding_failed",
                sa.Boolean(),
                nullable=False,
                server_default=sa.text("false"),
            ),
        )
    if "col_embeddings_failed" not in page_columns:
        op.add_column(
            "page",
            sa.Column(
                "col_embeddings_failed",
                sa.Boolean(),
                nullable=False,
                server_default=sa.text("false"),
            ),
        )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("page")
    op.drop_table("document")
