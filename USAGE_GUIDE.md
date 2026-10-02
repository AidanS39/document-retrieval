# Document Retrieval — Usage Guide

## Overview

This framework provides a unified interface for document retrieval across many different model backends. You drop documents into a folder, run a sequence of setup scripts to process and index them, then query across any of the 12 pre-configured retrieval systems from a single interactive menu. The same pipeline structure also supports systematic evaluation: you can benchmark multiple systems head-to-head using NDCG scoring and a browser-based annotation tool.

The four pipeline stages are: **Preprocessing** (extract text and render page images), **Embedding** (encode pages into vectors), **Indexing** (build a search index), and **Ranking** (query the index and return results).

---

## Prerequisites

- Python 3.11 or later
- [uv](https://docs.astral.sh/uv/) (package manager)
- Docker and Docker Compose
- A CUDA-capable GPU (required for all neural embedding models; TF-IDF and BM25 run on CPU)
- A Hugging Face account with an access token (for downloading gated models)
- A Gemini API key (only if using System 12 — `gemini-embedding-2`)

---

## Installation

1. Clone the repository and enter the project directory:

   ```bash
   git clone <repo-url>
   cd document_retrieval
   ```

2. Install Python dependencies:

   ```bash
   uv sync
   ```

3. Copy the example environment file:

   ```bash
   cp .env.example .env
   ```

4. Open `.env` and fill in all required values (see [Environment Configuration](#environment-configuration) below).

5. Start the PostgreSQL database (pgvector-enabled) and the application container:

   ```bash
   docker compose up -d
   ```

6. Run database migrations to create the schema:

   ```bash
   uv run alembic upgrade head
   ```

---

## Environment Configuration

Edit `.env` with your specific values. The key variables are:

**Database connection**

| Variable | Description |
|----------|-------------|
| `DB_DRIVER` | Set to `postgresql+psycopg` when running scripts locally |
| `DB_USER` | Your Postgres username |
| `DB_PASSWORD` | Your Postgres password |
| `DB_HOST` | `localhost` when accessing from host; `db` when inside Docker |
| `DB_PORT` | `5433` (host-side port; Docker maps this to `5432` inside the container) |
| `DB_DATABASE` | Your database name |
| `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DATABASE` | Mirror the `DB_` values; used by the Docker container itself |

**Paths**

| Variable | Description |
|----------|-------------|
| `DATA_DIR` | Where documents, embeddings, and evaluation data live (default: `/app/data`) |
| `TEST_DATA_DIR` | Data directory used during test runs |
| `HF_HOME` | Where Hugging Face model weights are cached (default: `/app/data/.cache/huggingface`) |

**API keys**

| Variable | Description |
|----------|-------------|
| `HF_TOKEN` | Your Hugging Face access token (get one at huggingface.co/settings/tokens) |
| `GEMINI_API_KEY` | Only required for System 12 |

**Optional**

| Variable | Description |
|----------|-------------|
| `HF_HUB_DISABLE_XET` | Set to `1` to disable XET protocol during model downloads if you see errors |
| `LOCAL_SYNC_PATH` / `REMOTE_SYNC_URL` | Used by a remote sync utility; not required for core functionality |
| `HOST` | Hostname and port for the annotator backend, used in production deployments |

---

## Pipeline Setup

### Step 1 — Prepare Your Documents

Create the documents directory and place your files there:

```bash
mkdir -p data/documents
```

Supported formats: PDF, DOCX, PPTX, TXT.

If you have DOC, PPT, or other legacy Office formats, the `conversion/` directory contains helper scripts (`find_doc.sh`, `find_docx.sh`, `convert_to_pdf.sh`, etc.) and a Dockerfile for running LibreOffice-based batch conversion to PDF before ingestion.

### Step 2 — Set Up the Database Schema

This creates all necessary tables (Documents, Pages, Embeddings, Evaluation annotations, etc.):

```bash
uv run python src/scripts/setup_db.py
```

### Step 3 — Seed the Database

Seeding reads every document from `data/documents/`, records metadata in the database, extracts text content, and renders each page as a JPEG image stored under `data/pages/`:

```bash
uv run python src/scripts/seed_db.py
```

This step can take significant time on large document sets since it renders every page of every document as an image. Progress is printed to the console. If seeding is interrupted and restarted, already-processed documents are skipped.

### Step 4 — Generate Embeddings

This script loads a model and embeds every page, saving the embeddings to disk under `data/embeddings/<model-name>/`:

```bash
uv run python src/scripts/setup_embeddings.py --model <model-name> --device cuda:0
```

Example:

```bash
uv run python src/scripts/setup_embeddings.py \
    --model TomoroAI/tomoro-colqwen3-embed-4b \
    --device cuda:0
```

Available `--model` values (see [Retrieval Systems Reference](#retrieval-systems-reference) for the full list):

```
nvidia/llama-nemotron-colembed-vl-3b-v2
webAI-Official/webAI-ColVec1-9b
webAI-Official/webAI-ColVec1-4b
Qwen/Qwen3-VL-Embedding-2B
Qwen/Qwen3-VL-Embedding-8B
vultr/VultronRetrieverCore-Qwen3.5-4.5B
athrael-soju/colqwen3.5-4.5B-v3
TomoroAI/tomoro-colqwen3-embed-4b
TomoroAI/tomoro-colqwen3-embed-8b
jinaai/jina-embeddings-v4
gemini-embedding-2
```

Available `--device` values: `cuda:0`, `cuda:1`

You must run this step separately for each model you want to use. TF-IDF and BM25 (Systems 10 and 11) do not require this step — they build their index on the fly at query time from stored page text.

Models are downloaded automatically from Hugging Face on first use and cached in `HF_HOME`.

### Step 5 — Build the Search Index

This step reads the saved embeddings from disk and constructs the search index used during retrieval:

```bash
uv run python src/scripts/setup_index.py --model <model-name> --device cuda:0
```

The `--model` argument must match the model used in the embedding step. For late-interaction models (ColQwen-style), this builds a FastPlaid k-means quantized index. For bi-encoder models, this writes vectors into pgvector columns in PostgreSQL.

Run this step once per model after embedding is complete.

### Step 6 — Run an Interactive Retrieval Session

```bash
uv run python src/main.py
```

The program checks whether the database has been initialized and seeded. If either step was not done, it runs it automatically before proceeding.

You are then presented with a numbered menu of all configured retrieval systems:

```
(1) Nemotron ColEmbed 3B + FastPlaid
(2) WebAI ColVec1 9B + FastPlaid
...
(12) Gemini Embedding 2 + pgvector
Please choose a retrieval system:
```

After choosing a system, you reach the main menu:

```
(1) Retrieve documents from query
(q) Quit program
```

Select option `1`, enter your query as a free-text string, and optionally set a `top_k` value (the number of results to return; default is 100). Results are printed as a ranked list of pages with document names and page numbers.

To switch retrieval systems, quit (`q`) and restart, or the program will prompt you to choose again if the current session's ranker is released.

---

## Retrieval Systems Reference

| ID | Name | Model | Type | Index |
|----|------|-------|------|-------|
| 1 | Nemotron ColEmbed 3B + FastPlaid | `nvidia/llama-nemotron-colembed-vl-3b-v2` | Late-interaction | FastPlaid |
| 2 | WebAI ColVec1 9B + FastPlaid | `webAI-Official/webAI-ColVec1-9b` | Late-interaction | FastPlaid |
| 3 | WebAI ColVec1 4B + FastPlaid | `webAI-Official/webAI-ColVec1-4b` | Late-interaction | FastPlaid |
| 4 | Qwen3 VL Embedding 2B + pgvector | `Qwen/Qwen3-VL-Embedding-2B` | Bi-encoder | pgvector |
| 5 | Vultron Qwen3.5 4.5B + FastPlaid | `vultr/VultronRetrieverCore-Qwen3.5-4.5B` | Late-interaction | FastPlaid |
| 6 | ColQwen3.5 4.5B v3 + FastPlaid | `athrael-soju/colqwen3.5-4.5B-v3` | Late-interaction | FastPlaid |
| 7 | TomoroAI ColQwen3 4B + FastPlaid | `TomoroAI/tomoro-colqwen3-embed-4b` | Late-interaction | FastPlaid |
| 8 | TomoroAI ColQwen3 8B + FastPlaid | `TomoroAI/tomoro-colqwen3-embed-8b` | Late-interaction | FastPlaid |
| 9 | Jina V4 Embedding + pgvector | `jinaai/jina-embeddings-v4` | Bi-encoder | pgvector |
| 10 | TF-IDF | `tf_idf` | Bi-encoder | In-memory |
| 11 | BM25 | `bm_25` | Bi-encoder | In-memory |
| 12 | Gemini Embedding 2 + pgvector | `gemini-embedding-2` | Bi-encoder | pgvector |

- **Late-interaction** models use FastPlaid's k-means quantized index and require a GPU.
- **Bi-encoder** models store and query vectors in PostgreSQL via pgvector.
- **TF-IDF and BM25** are bi-encoder methods that operate entirely in memory from stored page text; no embedding or index setup is needed.

**NOTE: as of now, systems 1, 7, 8, and 9 do not work due to version mismatch with remote model code.**

---

## Evaluation Workflow

The evaluation pipeline scores retrieval quality using NDCG (Normalized Discounted Cumulative Gain) over a pooled annotation set.

### Step A — Prepare Queries

Create a JSON file containing an array of query strings at `data/evaluation/queries.json`:

```json
[
  "What is the confinement time in NSTX experiments?",
  "Explain the role of neutral beam injection in plasma heating"
]
```

### Step B — Run Each System Over the Query Set

For each system you want to evaluate:

```bash
uv run python src/scripts/run_system_ranking.py \
    --system-id 7 \
    --queries-file data/evaluation/queries.json \
    --top-k 10
```

This produces `data/evaluation/systems/system_7.json`. Repeat for every system you want to compare. The `--top-k` flag controls how many pages are retrieved per query (default: 10).

### Step C — Build the Annotation Pool

Merge the per-system rankings into a unified pool of pages to annotate:

```bash
uv run python src/scripts/setup_evaluation.py
```

| Flag | Description |
|------|-------------|
| `--queries-file` | Path to your queries JSON (default: `$DATA_DIR/evaluation/queries.json`) |
| `--systems-dir` | Directory containing `system_*.json` files (default: `$DATA_DIR/evaluation/systems`) |
| `--pool-out` | Output path for `query_pool.json` (default: `$DATA_DIR/evaluation/query_pool.json`) |
| `--top-k` | How many results per system per query to include in the pool (default: 10) |
| `--systems` | Restrict to specific system IDs, e.g. `--systems 1 4 12` |

This writes `query_pool.json`, which the annotator tool reads to present pages for labeling.

### Step D — Annotate Using the Web Tool

Start the annotator (see [Annotator Tool](#annotator-tool) below), open `http://localhost:5173` in a browser, and label each page's relevance for each query on a 0–3 scale (0 = not relevant, 3 = highly relevant).

### Step E — Compute NDCG

```bash
uv run python src/scripts/compute_ndcg.py --k 10
```

| Flag | Description |
|------|-------------|
| `--k` | Cutoff for NDCG@k (default: 10) |
| `--aggregate` | How to merge multiple annotators' scores: `mean` or `majority` (default: `mean`) |
| `--pool` | Path to `query_pool.json`; if omitted, reads annotations directly from the database |
| `--systems-dir` | Directory with `system_*.json` files |
| `--out` | Output directory for results (default: `$DATA_DIR/evaluation/results`) |
| `--include-zero-idcg` | Include queries with no relevant pages in the mean (default: excluded) |
| `--after` | Only use annotations submitted after this ISO 8601 timestamp |
| `--before` | Only use annotations submitted before this ISO 8601 timestamp |

Results are printed to the console ranked by mean NDCG and also written to `data/evaluation/results/ndcg_results_<timestamp>.json`.

---

## Annotator Tool

The annotator is a browser-based tool for labeling page relevance. It consists of two Docker services.

Start both services:

```bash
docker compose up -d annotator-backend annotator-frontend
```

Then open `http://localhost:5173` in a browser.

The backend (FastAPI) runs on port `8001`. The frontend (Vite/Node) runs on port `5173`. The backend reads `query_pool.json` at startup — if the file does not exist yet, it will print a warning and the UI will show an error until you run `setup_evaluation.py`.

**In the UI:**

- Select a query from the list on the left.
- For each page shown, assign a relevance score (0 = not relevant, 1 = marginally relevant, 2 = relevant, 3 = highly relevant).
- Scores are saved immediately on submission.
- You can add a text note per query.
- A progress indicator shows how many pages you have annotated for each query.
- Multiple annotators can work simultaneously by entering different annotator names. NDCG computation can aggregate across annotators.

---

## Benchmarking

### Pipeline Throughput

The `setup_embeddings.py` and `setup_index.py` scripts automatically collect per-batch timing and GPU memory telemetry via `PipelineMetadata`. At the end of a run, summary statistics are printed:

```
Total time spent embedding: X.XXXX seconds
Total pages embedded: N
Time per page (embedding): X.XXXX seconds
Total time spent indexing: X.XXXX seconds
...
```

You can also export detailed per-batch telemetry to a timestamped JSON file from Python:

```python
metadata.export_to_json()
```

The metadata object is saved as `data/embeddings/<model-name>/metadata.pt` after each batch.

### Per-Query Latency

To benchmark how fast a system responds to individual queries:

```bash
uv run python src/scripts/benchmark_query_latency.py \
    --system-id 7 \
    --queries-file data/evaluation/queries.json \
    --top-k 10
```

This runs each query one at a time (not batched) and prints a per-query breakdown plus aggregate statistics:

```
Queries : N
Avg     : X.X ms
Min     : X.X ms
Max     : X.X ms
```

---

## Docker Services Reference

| Service | Description | Host Port |
|---------|-------------|-----------|
| `db` | PostgreSQL 17 with pgvector | 5433 |
| `app` | Interactive retrieval CLI | — |
| `annotator-backend` | FastAPI annotation server | 8001 |
| `annotator-frontend` | Vite/Node annotation UI | 5173 |
| `app-test` | Test runner (pytest); profile: `test` | — |

**Common commands:**

```bash
# Start the database only
docker compose up -d db

# Start everything
docker compose up -d

# Start annotator services only
docker compose up -d annotator-backend annotator-frontend

# Run the test suite inside Docker
docker compose --profile test up app-test

# Stop all services
docker compose down

# Stop all services and remove volumes (deletes all database data)
docker compose down -v
```

---

## Running Tests Locally

```bash
uv run pytest -s
```

Tests are organized per module: `test_preprocessing.py`, `test_embedding.py`, `test_indexing.py`, `test_ranking.py`, `test_setup.py`, `test_benchmarking.py`, `test_utils.py`, and `test_cross_module.py`. The test suite uses a separate database (name: `test`) and a separate data directory (`TEST_DATA_DIR`).

---

## Typical Workflow Summary

```bash
# 1. Configure environment
cp .env.example .env   # fill in credentials

# 2. Start services and migrate
docker compose up -d
uv run alembic upgrade head

# 3. Place documents
mkdir -p data/documents
# copy PDFs/DOCX/PPTX/TXT into data/documents/

# 4. Set up and seed the database
uv run python src/scripts/setup_db.py
uv run python src/scripts/seed_db.py

# 5. Embed and index (repeat for each model you want to use)
uv run python src/scripts/setup_embeddings.py --model <model> --device cuda:0
uv run python src/scripts/setup_index.py --model <model> --device cuda:0

# 6. Query interactively
uv run python src/main.py
```

To evaluate multiple systems:

```bash
# 7. Create query set
# Write data/evaluation/queries.json as a JSON array of strings

# 8. Run each system
uv run python src/scripts/run_system_ranking.py --system-id <id> --queries-file data/evaluation/queries.json

# 9. Build annotation pool
uv run python src/scripts/setup_evaluation.py

# 10. Annotate — open http://localhost:5173 after starting the annotator

# 11. Score
uv run python src/scripts/compute_ndcg.py --k 10
```

---

## Using the Framework as a Library

The `document_retrieval` package is designed to be imported directly into your own Python code, not only used through the CLI scripts. This section covers the programmatic API for querying, composing custom pipelines, and extending the framework with new models.

### Package Layout

The package lives under `src/document_retrieval/`. The public modules are:

| Module | Contents |
|--------|----------|
| `document_retrieval.ranking` | Ranker classes and result types (`PageRanking`, `DocRanking`) |
| `document_retrieval.embedding` | Embedder abstract base classes and all built-in implementations |
| `document_retrieval.indexing` | Indexer abstract base classes and all built-in implementations |
| `document_retrieval.preprocessing` | Document ingestion utilities |
| `document_retrieval.setup` | `DatabaseSetup`, `EmbeddingSetup`, `IndexingSetup` orchestration classes |
| `document_retrieval.models` | SQLAlchemy ORM models (`Document`, `Page`, `EvaluationAnnotation`) |
| `document_retrieval.benchmarking` | `PipelineMetadata`, `Timer`, and telemetry classes |
| `document_retrieval.evaluation` | `RetrievalSystem`, `QueryPool`, `AnnotationStore`, `NDCGComputer` |

The `scripts/` directory under `src/` is a collection of standalone CLI entry points that wire these modules together; it is not part of the installable package API.

### Quick Start: Query with a Pre-Configured System

The fastest way to query programmatically is to use `build_ranker` from `scripts/config.py`, which constructs the correct embedder/indexer pair for any of the 12 pre-configured system IDs.

```python
import os
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from document_retrieval.utils import get_device
from scripts.config import build_ranker

load_dotenv()

conn_url = URL.create(
    drivername=os.getenv("DB_DRIVER", "postgresql"),
    username=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    host=os.getenv("DB_HOST"),
    port=int(os.getenv("DB_PORT", "5432")),
    database=os.getenv("DB_DATABASE"),
)
engine = create_engine(conn_url)
device = get_device("cuda:0")
data_dir = Path(os.getenv("DATA_DIR", "/app/data"))

# Build a ranker for System 7 (TomoroAI ColQwen3 4B + FastPlaid)
ranker = build_ranker(system_id=7, engine=engine, device=device, data_dir=data_dir)

# Run one or more queries
rankings = ranker.rank(["What is the plasma confinement time?"], top_k=10)

for ranking in rankings:
    print(ranking)   # prints a formatted table of results
```

`build_ranker` returns a `PageRanker` (or `TfIdfPageRanker` / `BM25PageRanker` for Systems 10 and 11). The system must already have been embedded and indexed — run `setup_embeddings.py` and `setup_index.py` for that system ID first.

### Result Objects

`ranker.rank()` returns a `list[PageRanking]`, one per query.

**`PageRanking`**

| Attribute | Description |
|-----------|-------------|
| `ranking.query` | The original query string |
| `ranking.ranks` | List of `PageRank` objects, ordered by score descending |

**`PageRank`**

| Attribute | Description |
|-----------|-------------|
| `rank.position` | 1-based rank position |
| `rank.score` | Retrieval score (float) |
| `rank.page` | The `Page` ORM object |
| `rank.page.id` | Database page ID |
| `rank.page.number` | Page number within the document (1-based) |
| `rank.page.image_path` | Path to the rendered JPEG on disk |
| `rank.page.text` | Extracted page text (may be `None`) |
| `rank.page.document` | The parent `Document` ORM object |
| `rank.page.document.name` | Document filename stem |
| `rank.page.document.path` | Full path to the source file |

`PageRanking` also exposes `to_ranking()`, which serializes the result to a plain Python structure:

```python
query_str, ranks = ranking.to_ranking()
# ranks is a list of {"page_id": int, "position": int}
```

For document-level retrieval (TF-IDF and BM25 doc rankers), the equivalent types are `DocRanking` and `DocRank`, with `rank.doc` instead of `rank.page`.

### Composing a Ranker Manually

You can construct any embedder/indexer pair directly without going through `build_ranker`, which is useful when you want a combination not covered by the pre-configured systems.

**Late-interaction (ColQwen-style) + FastPlaid:**

```python
from document_retrieval.embedding import TomoroAIColPageEmbedder
from document_retrieval.indexing import FastPlaidIndexer
from document_retrieval.ranking import PageRanker

model_name = "TomoroAI/tomoro-colqwen3-embed-4b"

embedder = TomoroAIColPageEmbedder(model_name, engine, device, data_dir)
indexer = FastPlaidIndexer(model_name, device, data_dir, low_memory=True)
ranker = PageRanker(embedder, indexer, engine)

rankings = ranker.rank(["your query here"], top_k=25)
```

**Bi-encoder + pgvector:**

```python
from document_retrieval.embedding import Qwen3VLBiEncoderPageEmbedder
from document_retrieval.indexing import Qwen3VL2BIndexer
from document_retrieval.ranking import PageRanker

model_name = "Qwen/Qwen3-VL-Embedding-2B"

embedder = Qwen3VLBiEncoderPageEmbedder(model_name, engine, device, data_dir)
indexer = Qwen3VL2BIndexer(model_name, device, data_dir, engine)
ranker = PageRanker(embedder, indexer, engine)
```

**TF-IDF or BM25 (no pre-embedded index required):**

```python
from sqlalchemy import select
from sqlalchemy.orm import Session
from document_retrieval.models import Page
from document_retrieval.embedding import TfIdfPageEmbedder, BM25PageEmbedder
from document_retrieval.indexing import TfIdfIndexer, BM25Indexer
from document_retrieval.ranking import TfIdfPageRanker, BM25PageRanker

with Session(engine) as session:
    page_ids = list(session.scalars(select(Page.id)).all())

# TF-IDF
embedder = TfIdfPageEmbedder(engine)
indexer = TfIdfIndexer()
embeddings, valid_ids = embedder.embed_pages(page_ids)
indexer.index_pages(embeddings, valid_ids)
ranker = TfIdfPageRanker(embedder, indexer, engine)

# BM25 (same pattern, swap the classes)
embedder = BM25PageEmbedder(engine)
indexer = BM25Indexer()
tokenized, valid_ids = embedder.embed_pages(page_ids)
indexer.index_pages(tokenized, valid_ids)
ranker = BM25PageRanker(embedder, indexer, engine)

rankings = ranker.rank(["your query here"], top_k=25)
```

### Running the Database Setup Programmatically

If you are integrating the framework into a larger application and want to drive setup from code rather than the CLI scripts:

```python
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from document_retrieval.setup import DatabaseSetup, EmbeddingSetup, IndexingSetup

conn_url = URL.create(...)

# Create the database, enable pgvector, and create all tables
db_setup = DatabaseSetup(conn_url, data_dir)
db_setup.setup_db()

# Ingest documents: record metadata, extract text, render page images
db_setup.seed_db()

# Generate embeddings for a given embedder (skips already-embedded pages)
engine = create_engine(conn_url)
embedding_setup = EmbeddingSetup(engine, data_dir)
embedding_setup.setup_page_embeddings(embedder, batch_size=32)

# Build the search index
indexing_setup = IndexingSetup(engine, data_dir)
indexing_setup.setup_page_index(indexer)
```

`setup_page_embeddings` is idempotent: it checks which pages already have embeddings and skips them, so it is safe to call again if a previous run was interrupted.

### Implementing a Custom Embedder

Subclass `PageEmbedder` to plug in a new vision-language model. You must implement `embed_pages` and `embed_queries`.

```python
import torch
from pathlib import Path
from sqlalchemy.engine import Engine
from document_retrieval.embedding import PageEmbedder

class MyCustomPageEmbedder(PageEmbedder):
    def __init__(self, model_name: str, engine: Engine, device: torch.device, data_dir: Path):
        super().__init__(engine)
        self.model_name = model_name
        self.device = device
        self.data_dir = data_dir
        # load your model here

    def embed_pages(self, page_ids: list[int], batch_size: int = 32):
        # Load page images from the DB (image_path column on Page),
        # run them through your model, and save embeddings to disk as .pt files
        # under data_dir / "embeddings" / model_name / batch_<N>.pt
        # using the same format as the built-in embedders:
        #   torch.save({"embeddings": tensor, "page_ids": page_ids}, path)
        pass

    def embed_queries(self, queries: list[str]):
        # Encode queries and return embeddings in the format your indexer expects
        # (a list of vectors for pgvector, or a tensor for FastPlaid)
        pass
```

If your model follows the standard Hugging Face `AutoModel` + `AutoProcessor` pattern, you can subclass `TransformersBasedPageEmbedder` instead and only implement `_embedding_pipeline` (for pages) and `embed_queries`. The base class handles model loading, batch iteration, telemetry, and saving embeddings to disk automatically.

```python
from document_retrieval.embedding import TransformersBasedPageEmbedder

class MyHFPageEmbedder(TransformersBasedPageEmbedder):
    def _embedding_pipeline(self, images):
        # images is a list of PIL.Image objects for the current batch
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :]   # or whichever pooling you use

    def embed_queries(self, queries):
        inputs = self.processor(text=queries, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :].tolist()
```

### Implementing a Custom Indexer

Subclass `Indexer` to plug in a new retrieval backend.

```python
import torch
from pathlib import Path
from document_retrieval.indexing import Indexer

class MyCustomIndexer(Indexer):
    def __init__(self, index_name: str, device: torch.device, data_dir: Path):
        super().__init__(index_name, device, data_dir)
        # initialize your index structure here

    def index_pages(self, total_pages: int):
        # Load saved embeddings from disk and populate your index.
        # Use PipelineMetadata.load() to iterate over saved batch files,
        # same as FastPlaidIndexer and PGVectorIndexer do.
        pass

    def retrieve(self, query_embeddings, top_k: int = 25) -> list[list[tuple[int, float]]]:
        # Return a list (one element per query) of lists of (page_id, score) tuples,
        # ordered by score descending.
        pass
```

Once implemented, pass your indexer to `PageRanker` alongside any `PageEmbedder` to form a complete retrieval pipeline.

### Adding a Custom System to the Registry

To make your custom embedder and indexer available through `build_ranker` and the interactive menu, add entries to `MODEL_REGISTRY`, `INDEX_REGISTRY`, and `RETRIEVAL_SYSTEMS` in `src/scripts/config.py`:

```python
from document_retrieval.embedding import MyCustomPageEmbedder
from document_retrieval.indexing import MyCustomIndexer
from document_retrieval.evaluation import RetrievalSystem

MODEL_REGISTRY["my-org/my-model"] = MyCustomPageEmbedder
INDEX_REGISTRY["my-org/my-model"] = MyCustomIndexer

RETRIEVAL_SYSTEMS[13] = RetrievalSystem(
    id=13,
    name="My Custom Model + My Index",
    embed_model="my-org/my-model",
    embed_paradigm="col",      # "col" for late-interaction, "bi_encoder" for dense
    index_type="custom",
    top_k=10,
)
```

You may also need to update the `build_indexer` function in `config.py` if your indexer requires non-standard constructor arguments.

### Working with the ORM Models Directly

The SQLAlchemy ORM models can be imported and queried independently of the retrieval pipeline, for example to inspect document metadata or page content:

```python
from sqlalchemy.orm import Session
from sqlalchemy import select
from document_retrieval.models import Document, Page

with Session(engine) as session:
    # List all documents
    docs = session.scalars(select(Document)).all()
    for doc in docs:
        print(doc.id, doc.name, doc.path)

    # Get pages for a specific document
    pages = session.scalars(
        select(Page).where(Page.document_id == docs[0].id)
    ).all()
    for page in pages:
        print(page.number, page.image_path, page.text[:100] if page.text else "(no text)")
```

The `Document` model holds full-document text and document-level embedding columns. The `Page` model holds per-page image paths, per-page text, and all model-specific embedding columns (`qwen3_2b_embedding`, `qwen3_8b_embedding`, `gemini_embedding`). Late-interaction embeddings (FastPlaid) are stored on disk rather than in the database.
