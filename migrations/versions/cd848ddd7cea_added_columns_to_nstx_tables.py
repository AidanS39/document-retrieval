"""added columns to nstx tables

Revision ID: cd848ddd7cea
Revises: f8597f27dbab
Create Date: 2026-07-08 15:53:16.133551

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


# revision identifiers, used by Alembic.
revision: str = 'cd848ddd7cea'
down_revision: Union[str, Sequence[str], None] = 'f8597f27dbab'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    inspector = inspect(op.get_bind())

    embedding_cols = {col['name'] for col in inspector.get_columns('nstx_embeddings')}
    if 'figure_id' not in embedding_cols:
        op.add_column('nstx_embeddings', sa.Column('figure_id', sa.String(), nullable=True))

    paper_cols = {col['name'] for col in inspector.get_columns('nstx_papers')}
    if 'document_type' not in paper_cols:
        op.add_column('nstx_papers', sa.Column('document_type', sa.String(), nullable=True))
    if 'page_count' not in paper_cols:
        op.add_column('nstx_papers', sa.Column('page_count', sa.Integer(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    inspector = inspect(op.get_bind())

    embedding_cols = {col['name'] for col in inspector.get_columns('nstx_embeddings')}
    if 'figure_id' in embedding_cols:
        op.drop_column('nstx_embeddings', 'figure_id')

    paper_cols = {col['name'] for col in inspector.get_columns('nstx_papers')}
    if 'document_type' in paper_cols:
        op.drop_column('nstx_papers', 'document_type')
    if 'page_count' in paper_cols:
        op.drop_column('nstx_papers', 'page_count')
