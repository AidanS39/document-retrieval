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
from sqlalchemy import select, insert
from .models import Document, Page


def find_docs(docs_dir: Path) -> list[Path]:
    doc_paths = list()
    path_queue = deque()
    for path in docs_dir.iterdir():
        path_queue.append(path)

    while len(path_queue) > 0:
        cur_path = path_queue.popleft()
        if cur_path.is_file():
            doc_paths.append(cur_path)
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
        case ".docx":
            return _docx_to_text(doc_path)
        case ".pptx":
            return _pptx_to_text(doc_path)
        case ".txt":
            return _txt_to_text(doc_path)
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
                    "is_corrupt": is_corrupt,
                }
            )

    return pages


def convert_docs_to_images(
    engine, data_dir: Path, dpi: int = 150, max_workers: int = None
):
    if max_workers is None:
        max_workers = min(os.cpu_count() or 4, 4)
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
            session.execute(insert(Page), all_pages)
            session.commit()

    corrupt_count = sum(1 for p in all_pages if p["is_corrupt"])
    print(
        f"Done: {len(all_pages)} pages from {len(docs) - len(failed_docs)} documents. {corrupt_count} corrupt page(s)."
    )
    if failed_docs:
        print(f"Failed documents ({len(failed_docs)}): {failed_docs}")

    return len(all_pages), corrupt_count
