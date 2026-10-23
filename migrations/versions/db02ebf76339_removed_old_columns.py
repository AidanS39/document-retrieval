"""removed old columns

Revision ID: db02ebf76339
Revises: 635aa5bf2813
Create Date: 2026-10-05 02:12:54.387745

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'db02ebf76339'
down_revision: Union[str, Sequence[str], None] = '635aa5bf2813'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # drop the old columns, then rename query_tid to query_id (the fk follows the rename)
    op.drop_constraint(op.f('uq_annotation'), 'evaluation_annotation', type_='unique')
    op.drop_column('evaluation_annotation', 'query_id')
    op.drop_column('evaluation_annotation', 'query')
    op.alter_column('evaluation_annotation', 'query_tid', new_column_name='query_id')
    # the fk keeps its postgres default name from the old column, so rename it to match
    op.execute("ALTER TABLE evaluation_annotation RENAME CONSTRAINT evaluation_annotation_query_tid_fkey TO evaluation_annotation_query_id_fkey")
    op.create_unique_constraint('uq_annotation', 'evaluation_annotation', ['annotator_id', 'query_id', 'page_id'])

    op.drop_constraint(op.f('uq_note'), 'evaluation_note', type_='unique')
    op.drop_column('evaluation_note', 'query_id')
    op.drop_column('evaluation_note', 'annotator')
    op.alter_column('evaluation_note', 'query_tid', new_column_name='query_id')
    op.execute("ALTER TABLE evaluation_note RENAME CONSTRAINT evaluation_note_query_tid_fkey TO evaluation_note_query_id_fkey")
    op.create_unique_constraint('uq_note', 'evaluation_note', ['annotator_id', 'query_id'])

    op.drop_constraint(op.f('uq_query_fid'), 'query', type_='unique')
    op.drop_column('query', 'fid')


def downgrade() -> None:
    """Downgrade schema."""
    # old columns are added back as nullable, filled from the fk columns, then set to NOT NULL.
    # query.fid is filled first since the annotation and note fills read it
    op.add_column('query', sa.Column('fid', sa.INTEGER(), autoincrement=False, nullable=True))
    op.execute("UPDATE query SET fid = id")
    op.alter_column('query', 'fid', nullable=False)
    op.create_unique_constraint(op.f('uq_query_fid'), 'query', ['fid'], postgresql_nulls_not_distinct=False)

    op.drop_constraint('uq_note', 'evaluation_note', type_='unique')
    op.alter_column('evaluation_note', 'query_id', new_column_name='query_tid')
    op.execute("ALTER TABLE evaluation_note RENAME CONSTRAINT evaluation_note_query_id_fkey TO evaluation_note_query_tid_fkey")
    op.add_column('evaluation_note', sa.Column('annotator', sa.VARCHAR(), autoincrement=False, nullable=True))
    op.add_column('evaluation_note', sa.Column('query_id', sa.INTEGER(), autoincrement=False, nullable=True))
    op.execute("""
        UPDATE evaluation_note
        SET annotator = a.name
        FROM annotator AS a
        WHERE evaluation_note.annotator_id = a.id
    """)
    op.execute("""
        UPDATE evaluation_note
        SET query_id = q.fid
        FROM query AS q
        WHERE evaluation_note.query_tid = q.id
    """)
    op.alter_column('evaluation_note', 'annotator', nullable=False)
    op.alter_column('evaluation_note', 'query_id', nullable=False)
    op.create_unique_constraint(op.f('uq_note'), 'evaluation_note', ['annotator', 'query_id'], postgresql_nulls_not_distinct=False)

    op.drop_constraint('uq_annotation', 'evaluation_annotation', type_='unique')
    op.alter_column('evaluation_annotation', 'query_id', new_column_name='query_tid')
    op.execute("ALTER TABLE evaluation_annotation RENAME CONSTRAINT evaluation_annotation_query_id_fkey TO evaluation_annotation_query_tid_fkey")
    op.add_column('evaluation_annotation', sa.Column('query', sa.VARCHAR(), autoincrement=False, nullable=True))
    op.add_column('evaluation_annotation', sa.Column('query_id', sa.INTEGER(), autoincrement=False, nullable=True))
    op.execute("""
        UPDATE evaluation_annotation
        SET query = q.text, query_id = q.fid
        FROM query AS q
        WHERE evaluation_annotation.query_tid = q.id
    """)
    op.alter_column('evaluation_annotation', 'query', nullable=False)
    op.alter_column('evaluation_annotation', 'query_id', nullable=False)
    op.create_unique_constraint(op.f('uq_annotation'), 'evaluation_annotation', ['annotator_id', 'query_id', 'page_id'], postgresql_nulls_not_distinct=False)
