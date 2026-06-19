from sqlalchemy import insert
from sqlalchemy.orm import Session

from models import Document, Page

def insert_test_docs(engine):
    docs = [
        {"name": "test_doc_1", "path": "/tmp/test_doc_1.pdf", "text": "This is test document one."},
        {"name": "test_doc_2", "path": "/tmp/test_doc_2.pdf", "text": "This is test document two."},
        {"name": "test_doc_3", "path": "/tmp/test_doc_3.pdf", "text": "This is test document three."},
    ]
    with Session(engine) as session:
        result = session.execute(insert(Document).returning(Document.id, Document.name), docs)
        inserted = list(result)
        session.commit()
    print(f"inserted {len(inserted)} test documents: {[(row.id, row.name) for row in inserted]}")

