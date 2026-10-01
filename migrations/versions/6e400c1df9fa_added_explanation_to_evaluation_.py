"""added explanation to evaluation annotation

Revision ID: 6e400c1df9fa
Revises: f9be3bca9dc4
Create Date: 2026-09-27 17:23:56.135225

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '6e400c1df9fa'
down_revision: Union[str, Sequence[str], None] = 'f9be3bca9dc4'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('evaluation_annotation', sa.Column('explanation', sa.String(), server_default='', nullable=False))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('evaluation_annotation', 'explanation')
