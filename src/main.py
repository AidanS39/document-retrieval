import os
from dotenv import load_dotenv
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from document_retrieval.utils import db_initialized, db_seeded, get_device
from document_retrieval.setup import DatabaseSetup
from document_retrieval.evaluation import RetrievalSystem
from scripts.config import RETRIEVAL_SYSTEMS, build_ranker

load_dotenv()


def choose_ranker(device, data_dir, engine):
    for id, system in RETRIEVAL_SYSTEMS.items():
        print(f"({id}) {system.name}")
    valid_ranker_chosen = False
    while not valid_ranker_chosen:
        try:
            system_id = int(input("Please choose a retrieval system: ").strip())
            if system_id in RETRIEVAL_SYSTEMS:
                valid_ranker_chosen = True
            else:
                print(f"Invalid selection. Please choose from {list(RETRIEVAL_SYSTEMS.keys())}.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    return RETRIEVAL_SYSTEMS[system_id], build_ranker(system_id, engine, device, data_dir)


def prompt_top_k(default: int = 100) -> int:
    while True:
        raw = input(f"Enter top_k (default: {default}): ").strip()
        if not raw:
            return default
        try:
            val = int(raw)
            if val > 0:
                return val
            print("top_k must be a positive integer.")
        except ValueError:
            print("Invalid input. Please enter a positive integer.")


def main_menu():
    print("(1) Retrieve documents from query")
    print("(q) Quit program")
    return input("Please choose an option: ").strip()


def main():
    data_dir = Path(os.getenv("DATA_DIR", "/app/data"))

    device = get_device("cuda:1")

    DB_DRIVER = os.getenv("DB_DRIVER", "postgresql")
    DB_USER = os.getenv("POSTGRES_USER")
    DB_PASSWORD = os.getenv("POSTGRES_PASSWORD")
    DB_HOST = os.getenv("DB_HOST")
    DB_PORT = int(os.getenv("DB_PORT", "5432"))
    DB_DATABASE = os.getenv("POSTGRES_DATABASE")

    conn_url = URL.create(
        drivername=DB_DRIVER,
        username=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT,
        database=DB_DATABASE,
    )

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

    system, ranker = None, None
    end_program = False

    while not end_program:
        if system is None or ranker is None:
            system, ranker = choose_ranker(device, data_dir, engine)

        selection = main_menu()
        if selection == "1":
            query = input("Please enter a query:\n").strip()
            top_k = prompt_top_k()
            rankings = ranker.rank([query], top_k=top_k)
            for ranking in rankings:
                print(ranking)
        elif selection == "q":
            end_program = True


if __name__ == "__main__":
    main()
