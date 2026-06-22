import os
from dotenv import load_dotenv
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.engine import URL

load_dotenv()


def main():
    data_dir = Path("../data")

    DB_DRIVER = os.getenv("DB_DRIVER", "postgresql")
    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    DB_HOST = os.getenv("DB_HOST")
    DB_PORT = int(os.getenv("DB_PORT", "5432"))
    DB_DATABASE = os.getenv("DB_DATABASE")

    conn_url = URL.create(
        drivername=DB_DRIVER,
        username=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT,
        database=DB_DATABASE,
    )

    engine = create_engine(conn_url)


if __name__ == "__main__":
    main()
