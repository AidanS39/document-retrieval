from pathlib import Path
from collections import deque
import concurrent.futures
import re
import pandas as pd
import pymupdf
import docx
from pptx import Presentation
from pptx.shapes.autoshape import Shape

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
    failed = list()
    for doc_path in doc_paths:
        try:
            text = convert_doc_to_text(doc_path)
            texts.append(text)
        except Exception as e:
            print(f"Skipping {doc_path}: {e}")
            failed.append(doc_path)
            texts.append("")
    if failed:
        print(f"Failed to convert {len(failed)} file(s): {failed}")

    return texts

def convert_doc_to_images(doc_path: Path, images_dir: Path, dpi: int):
    if not doc_path.is_file():
        raise FileNotFoundError(f"Not a file: {doc_path}")
    if doc_path.suffix.lower() != ".pdf":
        raise ValueError(f"Not a PDF: {doc_path}")

    image_dir = images_dir / doc_path.stem
    image_dir.mkdir(parents=True, exist_ok=True)
    
    skipped = 0
    converted = 0
    
    image_paths = list()
    doc_paths = list()
    page_nums = list()

    with pymupdf.open(doc_path) as doc:
        # clears strutural tags of document (not needed for image conversion)
        doc.xref_set_key(doc.pdf_catalog(), "StructTreeRoot", "null")
        for page_num in range(0, len(doc)):
            image_path = image_dir / f"{doc_path.stem}_page_{page_num + 1}.jpg"
            if image_path.is_file():
                skipped += 1
            else:
                doc[page_num].get_pixmap(dpi=dpi).save(str(image_path))
                converted += 1
            image_paths.append(image_path)
            doc_paths.append(doc_path)
            page_nums.append(page_num)
    return image_paths, doc_paths, page_nums, skipped, converted

# saves each document as a folder of images, processed in parallel
def convert_docs_to_images(doc_paths: list, images_dir: Path, dpi: int = 150, max_workers: int = 16):
    images_dir.mkdir(parents=True, exist_ok=True)
    
    image_paths = list()
    page_doc_paths = list()
    page_nums = list()
    skipped = 0
    converted = 0
    failed = list()

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_path = {
            executor.submit(convert_doc_to_images, doc_path, images_dir, dpi): doc_path
            for doc_path in doc_paths
        }
        for future in concurrent.futures.as_completed(future_to_path):
            doc_path = future_to_path[future]
            try:
                cur_image_paths, cur_doc_paths, cur_page_nums, cur_skipped, cur_converted = future.result()
                image_paths += cur_image_paths
                page_doc_paths += cur_doc_paths
                page_nums += cur_page_nums
                skipped += cur_skipped
                converted += cur_converted
            except Exception as e:
                print(f"Error converting {doc_path}: {e}")
                failed.append(doc_path)

    print(f"Done: {converted} pages converted, {skipped} already existed.")
    if failed:
        print(f"Failed ({len(failed)}): {failed}")

    return image_paths, page_doc_paths, page_nums

# gets image paths and corresponding document name and page number from images folder
def get_images_metadata(images_dir: Path, metadata_path: Path):
    # if metadata_path.is_file():
    #     metadata = pd.read_csv(metadata_path)
    #     return metadata
    # else:
    image_paths = list()
    page_doc_paths = list()
    page_nums = list()

    for doc_path in images_dir.iterdir():
        doc_name = doc_path.stem
        if doc_path.is_dir():
            for image_path in doc_path.iterdir():
                image_paths.append(str(image_path))
                page_doc_paths.append(doc_name)
                page_nums.append(image_path.stem[-1])

    metadata = pd.DataFrame(
        {
            "image_path": image_paths, 
            "doc_name": page_doc_paths, 
            "page_num": page_nums
        }
    )
    metadata.to_csv(metadata_path, index=False)
    return image_paths, page_doc_paths, page_nums
