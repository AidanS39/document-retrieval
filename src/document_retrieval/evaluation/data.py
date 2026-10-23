from sqlalchemy import select, func, Engine
from sqlalchemy.engine import URL, create_engine
from sqlalchemy.orm import Session
from collections import Counter, defaultdict

from document_retrieval.evaluation import QueryPool, RetrievalSystem
from document_retrieval.models import EvaluationAnnotation, Annotator, Query

def load_annotation_data_2(query_pool: QueryPool, engine, exclude_names: list[str] = []) -> tuple:
    """Fetch and prepare annotation data shared by all report functions.

    Returns: (annotators, query_pages, annotations)
        annotators  : lists of Annotator (excluding excluded_names, restricted to those with annotations)
        query_pages : set of all active (query_id, page_id) pairs in query pool
        annotations : (query_id, page_id, annotator_id) -> score
    Only pooled pairs from active queries are included.
    """
    with Session(engine) as session:
        active_query_ids = set(session.scalars(select(Query.id).where(Query.active)))

    # set of all (query_id, page_id) pairs from query pool
    query_pages: set[tuple[int, int]] = {
        (query["id"], page_id)
        for query in query_pool.queries
        if query["id"] in active_query_ids
        for page_id in query["pooled_page_ids"]
    }

    with Session(engine) as session:
        annotators = session.execute(select(Annotator)).scalars().all()

        ann_rows = session.execute(
            select(
                EvaluationAnnotation.annotator_id,
                EvaluationAnnotation.query_id,
                EvaluationAnnotation.page_id,
                EvaluationAnnotation.score,
                Annotator.role,
            )
            .join(Annotator, EvaluationAnnotation.annotator_id == Annotator.id)
            .join(Query, EvaluationAnnotation.query_id == Query.id)
            .where(Query.active)
        ).all()

    # filter excluded annotators and any that do not have any annotations
    excluded_ids = {a.id for a in annotators if a.name in exclude_names}
    annotator_ids_with_annotations = {row.annotator_id for row in ann_rows}
    annotators = [a for a in annotators if a.id not in excluded_ids and a.id in annotator_ids_with_annotations]
    ann_rows = [row for row in ann_rows if row.annotator_id not in excluded_ids and row.annotator_id in annotator_ids_with_annotations]

    annotations: dict[tuple[int, int, int], float] = dict()

    for row in ann_rows:
        query_page = (row.query_id, row.page_id)
        if query_page not in query_pages:
            continue
        annotations[(row.query_id, row.page_id, row.annotator_id)] = row.score

    return annotators, query_pages, annotations

def human_ai_split(annotators):
    humans = [annotator for annotator in annotators if annotator.role == "human"]
    ais = [annotator for annotator in annotators if annotator.role == "ai"]

    return humans, ais

