# document-retrieval

A Python framework that provides a common interface for the document retrieval process. Working across many different retrieval systems, each implementation required bespoke tooling despite following the same underlying pattern. This framework abstracts those implementations into four components, each responsible for a distinct stage of the pipeline.

**Benefits:**
- Modular design — swap any component without touching the rest
- Consistent evaluation — compare implementations on a level playing field
- Isolated testing — unit-test each stage independently
- Ease of use — a single coherent API regardless of the backend

---

## Install

**Prerequisites:** Python 3.11+, [uv](https://docs.astral.sh/uv/), Docker

1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd document_retrieval
   ```

2. Install dependencies:
   ```bash
   uv sync
   ```

3. Copy the example environment file and fill in your values:
   ```bash
   cp .env.example .env
   ```

4. Start the PostgreSQL database with pgvector and the application:
   ```bash
   docker-compose up -d
   ```

5. Run database migrations:
   ```bash
   uv run alembic upgrade head
   ```

---

## Getting Started

Place your documents (PDF, DOCX, PPTX, or TXT) in `data/documents/`.

The setup utilities in `src/document_retrieval/setup.py` walk through each stage of the pipeline:

```python
from sqlalchemy import create_engine
from pathlib import Path
from document_retrieval.setup import DatabaseSetup, EmbeddingSetup
from document_retrieval.indexing import FastPlaidIndexer
from scripts.config import build_ranker
import torch

engine = create_engine("postgresql+psycopg://...")
data_dir = Path("data")
device = torch.device("cuda")

# 1. Seed the database and preprocess documents
setup = DatabaseSetup(engine, data_dir)
setup.setup_db()
setup.seed_db()

# 2. Embed all pages and build the search index (system 7: TomoroAI ColQwen3 4B + FastPlaid)
embedding_setup = EmbeddingSetup(engine, data_dir)
embedding_setup.setup_page_embeddings(build_ranker(7, engine, device, data_dir).embedder)

# 3. Build the search index
indexer = FastPlaidIndexer("TomoroAI/tomoro-colqwen3-embed-4b", device, data_dir)
indexer.index_pages(total_pages=1000)

# 4. Rank documents against a query
ranker = build_ranker(7, engine, device, data_dir)
rankings = ranker.rank(["what is the plasma current?"], top_k=10)
print(rankings[0])
```

For an interactive retrieval session, run:
```bash
uv run python src/main.py
```

This presents a numbered menu of the configured retrieval systems and prompts for a query.

---

## Retrieval Systems

`src/scripts/config.py` defines 12 pre-configured retrieval systems. Each pairs an embedder with an indexer:

| ID | Name | Model | Paradigm | Index |
|----|------|-------|----------|-------|
| 1 | Nemotron ColEmbed 3B + FastPlaid | `nvidia/llama-nemotron-colembed-vl-3b-v2` | col | fast_plaid |
| 2 | WebAI ColVec1 9B + FastPlaid | `webAI-Official/webAI-ColVec1-9b` | col | fast_plaid |
| 3 | WebAI ColVec1 4B + FastPlaid | `webAI-Official/webAI-ColVec1-4b` | col | fast_plaid |
| 4 | Qwen3 VL Embedding 2B + pgvector | `Qwen/Qwen3-VL-Embedding-2B` | bi_encoder | pgvector |
| 5 | Vultron Qwen3.5 4.5B + FastPlaid | `vultr/VultronRetrieverCore-Qwen3.5-4.5B` | col | fast_plaid |
| 6 | ColQwen3.5 4.5B v3 + FastPlaid | `athrael-soju/colqwen3.5-4.5B-v3` | col | fast_plaid |
| 7 | TomoroAI ColQwen3 4B + FastPlaid | `TomoroAI/tomoro-colqwen3-embed-4b` | col | fast_plaid |
| 8 | TomoroAI ColQwen3 8B + FastPlaid | `TomoroAI/tomoro-colqwen3-embed-8b` | col | fast_plaid |
| 9 | Jina V4 Embedding + pgvector | `jinaai/jina-embeddings-v4` | bi_encoder | pgvector |
| 10 | TF-IDF | `tf_idf` | bi_encoder | local |
| 11 | BM25 | `bm_25` | bi_encoder | local |
| 12 | Gemini Embedding 2 + pgvector | `gemini-embedding-2` | bi_encoder | pgvector |

`build_ranker(system_id, engine, device, data_dir)` constructs the appropriate embedder/indexer pair for any system ID.

---

## Concepts

The framework is built around four components. Each component is defined as an abstract base class; you implement the interface to plug in a new backend.

### Preprocessor

Transforms raw documents into a normalized form the rest of the pipeline can consume. It extracts text content and renders each page as an image, capturing document metadata (filename, path, page count) in the database along the way.

Supported formats: PDF, DOCX, PPTX, TXT.

```
Document (PDF / DOCX / PPTX / TXT)
        │
        ├─ extract text ──► Document.text (PostgreSQL)
        │
        └─ render pages ──► Page images (JPEG on disk, path in DB)
```

### Embedder

Converts documents and queries into vector representations. Different models have meaningfully different processing pipelines — tokenizers, image processors, pooling strategies, batching requirements — so each model gets its own `Embedder` subclass that encapsulates those details behind a uniform interface.

Two abstract variants cover the two granularities:

- **`PageEmbedder`** — embeds page images; used with vision-language models
- **`DocEmbedder`** — embeds document text; used with sparse retrieval methods

Built-in implementations:

| Class | Model(s) | Type |
|-------|----------|------|
| `TfIdfPageEmbedder` / `TfIdfDocEmbedder` | TF-IDF | Sparse |
| `BM25PageEmbedder` / `BM25DocEmbedder` | BM25 | Sparse |
| `Qwen3VLBiEncoderPageEmbedder` | `Qwen3-VL-Embedding-2B/8B` | Bi-encoder |
| `JinaV4BiEncoderPageEmbedder` | `jinaai/jina-embeddings-v4` | Bi-encoder |
| `GeminiBiEncoderPageEmbedder` | `gemini-embedding-2` | Bi-encoder |
| `NemotronColPageEmbedder` | `nvidia/llama-nemotron-colembed-vl-3b-v2` | Late-interaction |
| `WebAIColPageEmbedder` | `webAI-Official/webAI-ColVec1-4b/9b` | Late-interaction |
| `TomoroAIColPageEmbedder` | `TomoroAI/tomoro-colqwen3-embed-4b/8b` | Late-interaction |
| `Qwen3_5ColPageEmbedder` | `vultr/VultronRetrieverCore-Qwen3.5-4.5B`, `athrael-soju/colqwen3.5-4.5B-v3` | Late-interaction |

### Indexer

Organizes embeddings for fast retrieval and owns the retrieval implementation. It is deliberately decoupled from the `Embedder` so that different search paradigms (dense ANN, sparse inverted index, late-interaction, etc.) can be swapped without changing the rest of the pipeline.

The abstract interface is:

```python
class Indexer(ABC):
    def index_pages(self, total_pages: int): ...
    def retrieve(self, query_embeddings, top_k: int = 25) -> list[tuple[int, float]]: ...
```

Built-in implementations:

| Class | Backend | Best used with |
|-------|---------|----------------|
| `FastPlaidIndexer` | [FastPlaid](https://github.com/stanford-futuredata/plaid) k-means quantized index | Late-interaction (ColQwen-style) embedders |
| `PGVectorIndexer` | PostgreSQL pgvector ANN | Bi-encoder embedders |
| `Qwen3VL2BIndexer` / `Qwen3VL8BIndexer` | pgvector (model-specific columns) | `Qwen3-VL-Embedding-2B/8B` |
| `GeminiEmbedding2Indexer` | pgvector (model-specific column) | `gemini-embedding-2` |
| `TfIdfIndexer` | In-memory sklearn | `TfIdfPageEmbedder` |
| `BM25Indexer` | In-memory rank-bm25 | `BM25PageEmbedder` |

### Ranker

Orchestrates the end-to-end retrieval process: it uses an `Embedder` to encode the query, passes the query embedding to an `Indexer` to retrieve candidate pages, and returns the results as a ranked list of typed result objects (`PageRanking`, `DocRanking`).

The abstract interface is:

```python
class PageRanker(ABC):
    def rank(self, queries: list[str], top_k: int = 100) -> list[PageRanking]: ...
```

Concrete rankers (`PageRanker`, `TfIdfPageRanker`, `BM25PageRanker`, `TfIdfDocRanker`, `BM25DocRanker`) pair specific embedders with specific retrieval backends. Keeping this composition explicit makes it straightforward to benchmark one configuration against another.

---

## Evaluation

The evaluation pipeline measures retrieval quality (NDCG) across systems using pooled annotations.

### Workflow

1. **Run systems** — generate per-system rankings for a query set:
   ```bash
   uv run python src/scripts/run_system_ranking.py --system-id 7 --queries-file data/evaluation/queries.json
   ```

2. **Build query pool** — merge rankings across systems into a pool for annotation:
   ```bash
   uv run python src/scripts/setup_evaluation.py
   ```

3. **Annotate** — use the annotator tool (see below) to label page relevance.

4. **Compute NDCG** — score all systems against the collected annotations:
   ```bash
   uv run python src/scripts/compute_ndcg.py --k 10
   ```

Results are written to `data/evaluation/results/ndcg_results_{timestamp}.json`.

---

## Annotator

The annotator is a web tool for labeling page relevance. It runs as two Docker services defined in `docker-compose.yml`:

- **`annotator-backend`** — FastAPI server on port 8001
- **`annotator-frontend`** — Vite/Node frontend on port 5173

Start both with:
```bash
docker-compose up -d annotator-backend annotator-frontend
```

Then open `http://localhost:5173` in a browser.

---

## Benchmarking

`src/document_retrieval/benchmarking.py` provides telemetry classes for profiling the embedding and indexing stages of the pipeline. `PipelineMetadata` accumulates per-batch timing and GPU memory stats and can export a JSON summary:

```python
metadata.print_telemetry_summary()   # prints timing per page for embed + index
metadata.export_to_json()            # writes a timestamped JSON file to the embeddings path
```

Per-query latency can be benchmarked separately:
```bash
uv run python src/scripts/benchmark_query_latency.py --system-id 7
```

---

## Docker

The full stack is defined in `docker-compose.yml`. Key services:

| Service | Description | Port |
|---------|-------------|------|
| `db` | PostgreSQL 17 + pgvector | 5433 (host) |
| `app` | Interactive retrieval CLI | — |
| `annotator-backend` | Annotation API | 8001 |
| `annotator-frontend` | Annotation UI | 5173 |

To run the test suite inside Docker:
```bash
docker-compose --profile test up app-test
```
