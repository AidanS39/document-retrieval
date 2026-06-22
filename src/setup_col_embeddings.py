import os
from dotenv import load_dotenv
from pathlib import Path
from sqlalchemy.engine import URL, create_engine
from setup import Setup

load_dotenv()

DB_DRIVER = os.getenv("DB_DRIVER", "postgresql")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_DATABASE = os.getenv("DB_DATABASE")

def main():
    data_dir = Path("../data")

    conn_url = URL.create(
        drivername=DB_DRIVER,
        username=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT,
        database=DB_DATABASE
    )
    engine = create_engine(conn_url)
    
    setup = Setup(engine, data_dir)
    
    # setup embeddings for col embedder
    index_name = "llama-nemotron-colembed-vl-3b-v2_index"
    col_model_name = "nvidia/llama-nemotron-colembed-vl-3b-v2"
    setup.setup_col_embeddings(col_model_name, index_name)
    
if __name__ == "__main__":
    main()
