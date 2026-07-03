import os
from dotenv import load_dotenv
from pathlib import Path
from sqlalchemy import create_engine, select
from sqlalchemy.engine import URL
from sqlalchemy.orm import Session
from document_retrieval.models import Document
from document_retrieval.utils import db_initialized, db_seeded, get_device
from document_retrieval.setup import DatabaseSetup, EmbeddingSetup
from document_retrieval.embed import (
    _last_token_pool_embed,
    BiEncoderPageEmbedder,
    NemotronColPageEmbedder,
    WebAIColPageEmbedder,
)
from document_retrieval.indexing import Indexer
from document_retrieval.ranking import ColPageRanker, BiEncoderPageRanker

load_dotenv()


def main_menu():
    print("(1) Retrieve documents from query")
    print("(2) Change model")
    print("(q) Quit program")

    selection = input("Please choose an option: ").strip()
    return selection


def choose_ranker(device, data_dir, engine):
    options = {
        "1": {
            "id": "1",
            "name": "nvidia/llama-nemotron-colembed-vl-3b-v2",
            "type": "col",
            "subtype": "nvidia",
        },
        "2": {
            "id": "2",
            "name": "webAI-Official/webAI-ColVec1-9b",
            "type": "col",
            "subtype": "webAI",
        },
        "3": {
            "id": "3",
            "name": "webAI-Official/webAI-ColVec1-4b",
            "type": "col",
            "subtype": "webAI",
        },
        "4": {
            "id": "4",
            "name": "Qwen/Qwen3-VL-Embedding-2B",
            "type": "biencoder",
            "subtype": "Qwen",
        },
    }
    for option_key in options.keys():
        option = options[option_key]
        print(f"({option['id']}) {option['name']} - {option['type']}")
    model_id = input("Please choose a model: ").strip()
    model = options[model_id]

    if model["type"] == "col":
        indexer = Indexer(model["name"], device, data_dir, low_memory=True)
        match model["subtype"]:
            case "webAI":
                embedder = WebAIColPageEmbedder(model["name"], device, data_dir)
            case "nvidia":
                embedder = NemotronColPageEmbedder(model["name"], device, data_dir)

        ranker = ColPageRanker(embedder, indexer, engine)
    elif model["type"] == "biencoder":
        match model["subtype"]:
            case "Qwen":
                embedder = BiEncoderPageEmbedder(
                    model["name"], device, data_dir, _last_token_pool_embed
                )
        ranker = BiEncoderPageRanker(engine, embedder, data_dir)

    return ranker


def main():
    data_dir = Path(os.getenv("DATA_DIR"))

    device = get_device()

    DB_DRIVER = os.getenv("DB_DRIVER", "postgresql")
    DB_USER = os.getenv("POSTGRES_USER")
    DB_PASSWORD = os.getenv("POSTGRES_PASSWORD")
    DB_HOST = os.getenv("DB_HOST")
    DB_PORT = int(os.getenv("DB_PORT", "5432"))
    DB_DATABASE = os.getenv("POSTGRES_DATABASE")

    print(DB_DRIVER)
    print(DB_USER)

    conn_url = URL.create(
        drivername=DB_DRIVER,
        username=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT,
        database=DB_DATABASE,
    )
    # setup database if not already set up
    if db_initialized(data_dir) is False:
        print(f"Database {DB_DATABASE} is not yet initialized.")
        db_setup = DatabaseSetup(conn_url, data_dir)

        print(f"Setting up database {DB_DATABASE}...")
        db_setup.setup_db()

        print(f"Seeding database {DB_DATABASE}...")
        db_setup.seed_db()

    elif db_seeded(data_dir) is False:
        print(f"Database {DB_DATABASE} is initialized but is not yet seeded.")
        db_setup = DatabaseSetup(conn_url, data_dir)

        print(f"Seeding database {DB_DATABASE}...")
        db_setup.seed_db()
    print(f"Database {DB_DATABASE} is ready.")

    engine = create_engine(conn_url)

    # embed_setup = EmbeddingSetup(engine, data_dir)

    # embed_setup.setup_col_index(indexer, device)

    ranker = None
    end_program = False

    while not end_program:
        if ranker is None:
            ranker = choose_ranker(device, data_dir, engine)

        selection = main_menu()
        if selection == "1":
            query = input("Please enter a query:\n").strip()
            rankings = ranker.rank([query])
            for ranking in rankings:
                print(ranking)
        elif selection == "2":
            ranker = choose_ranker(device, data_dir, engine)


if __name__ == "__main__":
    main()
