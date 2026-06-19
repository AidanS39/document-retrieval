import time
from pathlib import Path
import torch
from sentence_transformers import SentenceTransformer
from sqlalchemy.orm import Session
from sqlalchemy import select, delete, or_, and_
from models import Document, Page
from transformers.image_utils import load_image
from utils import time_function, get_device

def get_embedding_model(models_dir: Path, model_name: str, device: torch.device):
    model_dir = models_dir / model_name
    if model_dir.exists():
        print(f"{model_name} found locally. loading from {model_dir}...")
        model = SentenceTransformer(str(model_dir), device=str(device))
    else:
        print(f"{model_name} not found locally. loading from remote...")
        model = SentenceTransformer(model_name, device=str(device))

        print(f"saving model to {model_dir}")
        model.save(str(model_dir))
    
    return model

def embed_pages(model, page_embeddings_path: Path, image_paths: list[Path]):
    if page_embeddings_path.exists():
        print(f"page embeddings found locally. loading from {page_embeddings_path}...")
        page_embeddings = torch.load(str(page_embeddings_path))
    else:
        print("page embeddings not found locally. calculating page embeddings...")
        page_embeddings = model.encode([str(path) for path in image_paths], convert_to_tensor=True)
        print(f"saving page embeddings to {page_embeddings_path}")
        torch.save(page_embeddings, str(page_embeddings_path))

    return page_embeddings

@time_function
def col_embed_docs(engine, embed_model, index, doc_ids: list[int]):

    device = get_device()
    print(device)

    with Session(engine) as session:
        for doc_id in doc_ids:
            stmt = select(Page.id, Page.image_path).where(Page.document_id == doc_id)
            pages = session.execute(stmt).all()
            
            images = [load_image(page[1]) for page in pages]
            start_time = time.time() 
            
            image_embeddings = embed_model.forward_images(images, batch_size=8)
            print(f"image embeddings shape: {image_embeddings.shape}")
            
            end_time = time.time()
            print(f"image embeddings creation took {end_time - start_time} seconds.")
            start_time = time.time() 
            
            image_embeddings = torch.flatten(image_embeddings, 0, 1)

            peak_allocated = torch.cuda.max_memory_allocated()
            print(f"peak VRAM allocation: {peak_allocated / 1024**3:.2f} GB")
            print(f"doc embedding shape: {image_embeddings.shape}")
            index.update(
                documents_embeddings=[image_embeddings],
                metadata=[{"doc_id": doc_id}]
            )

            end_time = time.time()
            print(f"image embeddings indexing took {end_time - start_time} seconds.")


