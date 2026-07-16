import os
from dotenv import load_dotenv
from pathlib import Path
from sqlalchemy.engine import URL, create_engine
from document_retrieval.utils import get_device
from document_retrieval.embedding import BiEncoderPageEmbedder, _last_token_pool_embed, Qwen3_5ColPageEmbedder
from document_retrieval.setup import EmbeddingSetup

load_dotenv()

DB_DRIVER = os.getenv("DB_DRIVER", "postgresql")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_DATABASE = os.getenv("DB_DATABASE")



def main():
    data_dir = Path(os.getenv("DATA_DIR", "/app/data"))

    model_name = "vultr/VultronRetrieverCore-Qwen3.5-4.5B"
    device = get_device()

    print(f"Starting embedding process for {model_name}")

    conn_url = URL.create(
        drivername=DB_DRIVER,
        username=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT,
        database=DB_DATABASE,
    )
    engine = create_engine(conn_url)

    setup = EmbeddingSetup(engine, data_dir)

    # setup embeddings for bi encoder
    # embedder = BiEncoderPageEmbedder(model_name, engine, device, data_dir, _last_token_pool_embed)
 
    # setup embeddings for col embedder
    embedder = Qwen3_5ColPageEmbedder(model_name, engine, device, data_dir)

    setup.setup_page_embeddings(embedder)


if __name__ == "__main__":
    main()
