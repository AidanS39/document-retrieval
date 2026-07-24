from document_retrieval.utils import sanitize_string
from pathlib import Path
from collections import deque
import concurrent.futures
import os
import pymupdf
import docx
from pptx import Presentation
from pptx.shapes.autoshape import Shape
from PIL import Image
from sqlalchemy.orm import Session
from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert
from .models import Document, Page


def find_docs(docs_dir: Path) -> list[Path]:
    doc_paths = list()
    path_queue = deque()
    for path in docs_dir.iterdir():
        path_queue.append(path)

    while len(path_queue) > 0:
        cur_path = path_queue.popleft()
        if cur_path.is_file():
            if cur_path.suffix.lower() == ".pdf":
                doc_paths.append(cur_path)
            else:
                print(f"Skipping {cur_path}: not a PDF")
        else:
            for path in cur_path.iterdir():
                path_queue.append(path)

    return doc_paths


def _pdf_to_text(doc_path: Path) -> str:
    doc = pymupdf.open(doc_path)
    text = ""
    for page in doc:
        text += str(page.get_text())
    return text


def _docx_to_text(doc_path: Path) -> str:
    document = docx.Document(str(doc_path))
    return "\n".join(p.text for p in document.paragraphs)


def _pptx_to_text(doc_path: Path) -> str:
    prs = Presentation(str(doc_path))
    parts = []
    for slide in prs.slides:
        for shape in slide.shapes:
            if isinstance(shape, Shape):
                for para in shape.text_frame.paragraphs:
                    parts.append(para.text)
    return "\n".join(parts)


def _txt_to_text(doc_path: Path) -> str:
    return doc_path.read_text(errors="replace")


def convert_doc_to_text(doc_path: Path) -> str:
    match doc_path.suffix.lower():
        case ".pdf":
            return _pdf_to_text(doc_path)
        # case ".docx":
        #     return _docx_to_text(doc_path)
        # case ".pptx":
        #     return _pptx_to_text(doc_path)
        # case ".txt":
        #     return _txt_to_text(doc_path)
        case suffix:
            raise ValueError(f"Unsupported file format: {suffix!r} ({doc_path.name})")


def convert_docs_to_texts(doc_paths: list[Path]):
    texts = list()
    successful = list()
    failed = list()

    for doc_path in doc_paths:
        try:
            text = convert_doc_to_text(doc_path)
            texts.append(text)
            successful.append(doc_path)
        except Exception as e:
            print(f"Skipping {doc_path}: {e}")
            failed.append(doc_path)
    if failed:
        print(f"Failed to convert {len(failed)} file(s): {failed}")

    return texts, successful, failed


def _verify_image(path: Path) -> bool:
    try:
        with Image.open(path) as img:
            img.load()
        return True
    except Exception:
        return False


def _convert_doc_to_images(
    doc_id: int, doc_path_str: str, images_dir_str: str, dpi: int
) -> list[dict]:
    doc_path = Path(doc_path_str)
    images_dir = Path(images_dir_str)

    if not doc_path.is_file():
        raise FileNotFoundError(f"Not a file: {doc_path}")
    if doc_path.suffix.lower() != ".pdf":
        raise ValueError(f"Not a PDF: {doc_path}")

    image_dir = images_dir / doc_path.stem
    image_dir.mkdir(parents=True, exist_ok=True)

    pages = []
    with pymupdf.open(doc_path) as doc:
        doc.xref_set_key(doc.pdf_catalog(), "StructTreeRoot", "null")
        for page_num in range(len(doc)):
            image_path = image_dir / f"{doc_path.stem}_page_{page_num + 1}.jpg"

            if image_path.exists():
                pages.append(
                    {
                        "document_id": doc_id,
                        "image_path": str(image_path),
                        "number": page_num + 1,
                    }
                )
                continue

            is_corrupt = False

            try:
                doc[page_num].get_pixmap(dpi=dpi).save(str(image_path))
            except Exception as e:
                print(
                    f"WARNING: failed to convert page {page_num + 1} of {doc_path.name}: {e}"
                )
                is_corrupt = True

            if not is_corrupt and not _verify_image(image_path):
                print(
                    f"WARNING: page {page_num + 1} of {doc_path.name} failed integrity check."
                )
                is_corrupt = True

            pages.append(
                {
                    "document_id": doc_id,
                    "image_path": str(image_path),
                    "number": page_num + 1,
                }
            )

    return pages


def convert_docs_to_images(
    engine, data_dir: Path, dpi: int = 150, max_workers: int = 4
):
    max_workers = min(os.cpu_count() or 4, max_workers)
    images_dir = data_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    with Session(engine) as session:
        docs = session.execute(select(Document.id, Document.path)).all()

    all_pages = []
    failed_docs = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_doc = {
            executor.submit(
                _convert_doc_to_images, doc.id, doc.path, str(images_dir), dpi
            ): doc
            for doc in docs
        }
        for future in concurrent.futures.as_completed(future_to_doc):
            doc = future_to_doc[future]
            try:
                all_pages.extend(future.result())
            except Exception as e:
                print(f"ERROR: failed to convert {doc.path}: {e}")
                failed_docs.append(doc.path)

    if all_pages:
        with Session(engine) as session:
            session.execute(insert(Page).on_conflict_do_nothing(), all_pages)
            session.commit()

    print(
        f"Done: {len(all_pages)} pages from {len(docs) - len(failed_docs)} documents."
    )
    if failed_docs:
        print(f"Failed documents ({len(failed_docs)}): {failed_docs}")

    return len(all_pages)


_OCR_TIMEOUT = 60  # seconds before falling back to plain text extraction


def _extract_page_text(pdf_page: pymupdf.Page) -> str:
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        future = executor.submit(pdf_page.get_textpage_ocr, flags=0, dpi=150, full=False)
        textpage = future.result(timeout=_OCR_TIMEOUT)
        return pdf_page.get_text(textpage=textpage).strip()
    except concurrent.futures.TimeoutError:
        print("WARNING: OCR timed out, falling back to plain text extraction")
        return pdf_page.get_text().strip()
    except Exception:
        return pdf_page.get_text().strip()
    finally:
        executor.shutdown(wait=False)


def _extract_texts_for_doc(
    doc_id: int, doc_path_str: str, page_numbers: list[int]
) -> list[dict]:
    doc_path = Path(doc_path_str)

    try:
        pdf = pymupdf.open(doc_path)
    except Exception as e:
        print(f"ERROR: could not open {doc_path}: {e}")
        return [{"id": page_id, "text": None, "text_failed": True} for page_id, _ in page_numbers]

    results = []
    with pdf:
        for page_id, page_number in page_numbers:
            try:
                text = sanitize_string(_extract_page_text(pdf[page_number - 1]))
                results.append({"id": page_id, "text": text, "text_failed": False})
            except Exception as e:
                print(f"WARNING: failed to extract text from page {page_number} of {doc_path.name}: {e}")
                results.append({"id": page_id, "text": None, "text_failed": True})

    print("results append")
    return results


def _flush_page_text_updates(engine, batch: list[dict]) -> tuple[int, int]:
    with Session(engine) as session:
        session.execute(update(Page), batch)
        session.commit()
    succeeded = sum(1 for u in batch if not u["text_failed"])
    return succeeded, len(batch) - succeeded


def extract_page_texts(engine, max_workers: int = 4, batch_size: int = 50):
    """Extract text for all pages that do not yet have text, writing results to the DB.

    Results are flushed to the DB every `batch_size` pages so progress is saved
    incrementally rather than all at once at the end.
    """
    max_workers = min(os.cpu_count() or 4, max_workers)

    with Session(engine) as session:
        rows = session.execute(
            select(Page.id, Page.number, Document.id, Document.path)
            .join(Document, Page.document_id == Document.id)
            .where(Page.text.is_(None), Page.text_failed.is_(False))
        ).all()

    if not rows:
        print("No pages require text extraction.")
        return 0

    # Group pages by document
    doc_pages: dict[tuple[int, str], list[tuple[int, int]]] = {}
    for page_id, page_number, doc_id, doc_path in rows:
        key = (doc_id, doc_path)
        doc_pages.setdefault(key, []).append((page_id, page_number))

    total_pages = len(rows)
    print(f"Extracting text for {total_pages} pages across {len(doc_pages)} documents.")

    pending: list[dict] = []
    total_succeeded = 0
    total_failed = 0
    total_written = 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        print("before process submit")
        future_to_doc = {
            executor.submit(_extract_texts_for_doc, doc_id, doc_path, pages): (doc_id, doc_path)
            for (doc_id, doc_path), pages in doc_pages.items()
        }
        print("after process submit")
        for future in concurrent.futures.as_completed(future_to_doc):
            print("COMPLETE")
            doc_id, doc_path = future_to_doc[future]
            print(f"doc completed: {doc_path}")
            try:
                pending.extend(future.result())
            except Exception as e:
                print(f"ERROR: unexpected failure for document {doc_path}: {e}")

            try:
                succeeded, failed = _flush_page_text_updates(engine, pending)
                total_succeeded += succeeded
                total_failed += failed
                total_written += len(pending)
                print(f"  [{total_written}/{total_pages}] flushed {len(pending)} pages to DB.")
            except Exception as e:
                print(f"ERROR: flush failed for batch of {len(pending)} pages: {e}")
                failed_batch = [{"id": u["id"], "text": None, "text_failed": True} for u in pending]
                try:
                    _flush_page_text_updates(engine, failed_batch)
                    print(f"  Marked {len(failed_batch)} pages as text_failed=True.")
                except Exception as e2:
                    print(f"ERROR: could not mark pages as failed: {e2}")
            finally:
                pending.clear()

    if pending:
        try:
            succeeded, failed = _flush_page_text_updates(engine, pending)
            total_succeeded += succeeded
            total_failed += failed
            total_written += len(pending)
        except Exception as e:
            print(f"ERROR: final flush failed for batch of {len(pending)} pages: {e}")
            failed_batch = [{"id": u["id"], "text": None, "text_failed": True} for u in pending]
            try:
                _flush_page_text_updates(engine, failed_batch)
                print(f"  Marked {len(failed_batch)} pages as text_failed=True.")
            except Exception as e2:
                print(f"ERROR: could not mark pages as failed: {e2}")

    print(f"Done: {total_succeeded} pages extracted, {total_failed} failed.")
    return total_succeeded
