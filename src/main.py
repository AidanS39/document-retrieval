import os
from dotenv import load_dotenv
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from document_retrieval.utils import db_initialized, db_seeded, get_device
from document_retrieval.setup import DatabaseSetup
from document_retrieval.embedding import (
    _last_token_pool_embed,
    BiEncoderPageEmbedder,
    NemotronColPageEmbedder,
    WebAIColPageEmbedder,
    TomoroAIColPageEmbedder,
    Qwen3_5ColPageEmbedder,
)
from document_retrieval.indexing import FastPlaidIndexer, PGVectorIndexer
from document_retrieval.ranking import ColPageRanker, BiEncoderPageRanker
from document_retrieval.benchmarking import RetrievalSystem, RelevanceScores

load_dotenv()

EMBEDDER_REGISTRY: dict[str, type] = {
    "nvidia/llama-nemotron-colembed-vl-3b-v2":   NemotronColPageEmbedder,
    "webAI-Official/webAI-ColVec1-9b":           WebAIColPageEmbedder,
    "webAI-Official/webAI-ColVec1-4b":           WebAIColPageEmbedder,
    "Qwen/Qwen3-VL-Embedding-2B":                BiEncoderPageEmbedder,
    "vultr/VultronRetrieverCore-Qwen3.5-4.5B":   Qwen3_5ColPageEmbedder,
    "athrael-soju/colqwen3.5-4.5B-v3":           Qwen3_5ColPageEmbedder,
    "TomoroAI/tomoro-colqwen3-embed-4b":         TomoroAIColPageEmbedder,
    "TomoroAI/tomoro-colqwen3-embed-8b":         TomoroAIColPageEmbedder,
}

INDEXER_REGISTRY: dict[str, type] = {
    "fast_plaid": FastPlaidIndexer,
    "pgvector":   PGVectorIndexer,
}

RANKER_REGISTRY: dict[str, type] = {
    "col":        ColPageRanker,
    "bi_encoder": BiEncoderPageRanker,
}

RETRIEVAL_SYSTEMS: dict[int, RetrievalSystem] = {
    1: RetrievalSystem(
        1,
        "Nemotron ColEmbed 3B + FastPlaid",
        "nvidia/llama-nemotron-colembed-vl-3b-v2",
        "col",
        "fast_plaid"
    ),
    2: RetrievalSystem(
        2,
        "WebAI ColVec1 9B + FastPlaid",
        "webAI-Official/webAI-ColVec1-9b",
        "col",
        "fast_plaid"
    ),
    3: RetrievalSystem(
        3,
        "WebAI ColVec1 4B + FastPlaid",
        "webAI-Official/webAI-ColVec1-4b",
        "col",
        "fast_plaid"
    ),
    4: RetrievalSystem(
        4,
        "Qwen3 VL Embedding 2B + pgvector",
        "Qwen/Qwen3-VL-Embedding-2B",
        "bi_encoder",
        "pgvector"
    ),
    5: RetrievalSystem(
        5,
        "Vultron Qwen3.5 4.5B + FastPlaid",
        "vultr/VultronRetrieverCore-Qwen3.5-4.5B",
        "col",
        "fast_plaid"
    ),
    6: RetrievalSystem(
        6,
        "ColQwen3.5 4.5B v3 + FastPlaid",
        "athrael-soju/colqwen3.5-4.5B-v3",
        "col",
        "fast_plaid"
    ),
    7: RetrievalSystem(
        7,
        "TomoroAI ColQwen3 4B + FastPlaid",
        "TomoroAI/tomoro-colqwen3-embed-4b",
        "col",
        "fast_plaid"
    ),
    8: RetrievalSystem(
        8,
        "TomoroAI ColQwen3 8B + FastPlaid",
        "TomoroAI/tomoro-colqwen3-embed-8b",
        "col",
        "fast_plaid"
    ),
}

def build_ranker(system_id: int, engine, device, data_dir: Path):
    system = RETRIEVAL_SYSTEMS[system_id]
    embedder_cls = EMBEDDER_REGISTRY[system.embed_model]
    indexer_cls = INDEXER_REGISTRY[system.search_method]
    ranker_cls = RANKER_REGISTRY[system.embed_paradigm]

    if ranker_cls is ColPageRanker:
        embedder = embedder_cls(system.embed_model, engine, device, data_dir)
        indexer = indexer_cls(system.embed_model, device, data_dir, low_memory=True)
        return ColPageRanker(embedder, indexer, engine)
    else:
        embedder = embedder_cls(system.embed_model, engine, device, data_dir, _last_token_pool_embed)
        return BiEncoderPageRanker(embedder, engine, data_dir)

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

    json_path = data_dir / "evaluation" / "retrieval_systems" / f"retrieval_system_{system_id}.json"
    if json_path.exists():
        print(f"Loading retrieval system from {json_path}")
        system = RetrievalSystem.import_from_json(json_path)
    else:
        print(f"No saved retrieval system found for {RETRIEVAL_SYSTEMS[system_id].name}. Creating new one.")
        system = RETRIEVAL_SYSTEMS[system_id]

    return system, build_ranker(system_id, engine, device, data_dir)

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


def load_queries(data_dir: Path) -> dict[str, RelevanceScores]:
    evaluation_dir = data_dir / "evaluation"
    relevance_scores_dir = evaluation_dir / "relevance_scores"
    filename = input("Enter query file name (default: queries.txt): ").strip() or "queries.txt"
    query_file = evaluation_dir / filename

    with open(query_file) as f:
        queries = [line.strip() for line in f if line.strip()]

    relevance_scores = {}
    for i, query in enumerate(queries, start=1):
        json_path = relevance_scores_dir / f"relevance_scores_{i}.json"
        if json_path.exists():
            print(f"Loading relevance scores {i} from {json_path}")
            rs = RelevanceScores.import_from_json(json_path)
        else:
            rs = RelevanceScores(id=i, query=query, top_k=10)
            rs.export_to_json(relevance_scores_dir)
        relevance_scores[query] = rs

    print(f"Loaded {len(relevance_scores)} queries.")
    return relevance_scores


def main_menu():
    print("(1) Retrieve documents from query")
    print("(2) Evaluate current Retrieval System")
    print("(3) Change retrieval system")
    print("(q) Quit program")

    selection = input("Please choose an option: ").strip()
    return selection

def main():
    data_dir = Path(os.getenv("DATA_DIR", "/app/data"))
    evaluation_dir = data_dir / "evaluation"
    relevance_scores_dir = evaluation_dir / "relevance_scores"
    retrieval_systems_dir = evaluation_dir / "retrieval_systems"


    device = get_device("cuda:1")

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

    system, ranker = None, None
    relevance_scores: dict[str, RelevanceScores] = dict()
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
        elif selection == "2":
            relevance_scores = load_queries(data_dir)
            queries = list(relevance_scores.keys())
            top_k = 10
            rankings = ranker.rank(queries, top_k=top_k)
            for ranking in rankings:
                print(ranking)
                query, ranks = ranking.to_ranking()
                system.add_ranking(query, ranks)
                rs = relevance_scores[ranking.query]
                rs.add_page_ids_to_pool([rank.page.id for rank in ranking.ranks], system.id)
                rs.export_to_json(relevance_scores_dir)
            out_path = system.export_to_json(retrieval_systems_dir)
            print(f"Retrieval system saved to {out_path}")
        elif selection == "3":
            system, ranker = choose_ranker(device, data_dir, engine)
        elif selection == "q":
            end_program = True


if __name__ == "__main__":
    main()
