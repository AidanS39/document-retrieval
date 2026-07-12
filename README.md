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

3. Start the PostgreSQL database with pgvector:
   ```bash
   docker-compose up -d
   ```

4. Copy the example environment file and fill in your values:
   ```bash
   cp .env.example .env
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
from document_retrieval.embedding import TomoroAIColPageEmbedder
from document_retrieval.indexing import FastPlaidIndexer
from document_retrieval.ranking import ColPageRanker
import torch

engine = create_engine("postgresql+psycopg://...")
data_dir = Path("data")
device = torch.device("cuda")

# 1. Seed the database and preprocess documents
setup = DatabaseSetup(engine, data_dir)
setup.setup_db()
setup.seed_db()

# 2. Embed all pages
embedder = TomoroAIColPageEmbedder(engine, device, data_dir)
embedding_setup = EmbeddingSetup(engine, data_dir)
embedding_setup.setup_page_embeddings(embedder)

# 3. Build the search index
indexer = FastPlaidIndexer("my-index", device, data_dir)
indexer.index_pages(total_pages=1000)

# 4. Rank documents against a query
ranker = ColPageRanker(embedder, indexer, engine)
rankings = ranker.rank(["what is the plasma current?"], top_k=10)
print(rankings[0])
```

For an interactive retrieval session, run:
```bash
uv run python src/main.py
```

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

Built-in implementations include TF-IDF, BM25, a bi-encoder (`Qwen3-VL-Embedding-2B`), and several ColQwen-style late-interaction models (Nemotron, WebAI, TomoroAI, Qwen3).

### Indexer

Organizes embeddings for fast retrieval and owns the retrieval implementation. It is deliberately decoupled from the `Embedder` so that different search paradigms (dense ANN, sparse inverted index, late-interaction, etc.) can be swapped without changing the rest of the pipeline.

The abstract interface is:

```python
class Indexer(ABC):
    def index_pages(self, total_pages: int): ...
    def retrieve(self, query_embeddings, top_k: int = 25) -> list[tuple[int, float]]: ...
```

The built-in implementation, `FastPlaidIndexer`, uses the [FastPlaid](https://github.com/stanford-futuredata/plaid) library for late-interaction retrieval. It loads embeddings from disk in batches, builds a k-means quantized index, and returns `(page_id, score)` tuples.

### Ranker

Orchestrates the end-to-end retrieval process: it uses an `Embedder` to encode the query, passes the query embedding to an `Indexer` to retrieve candidate pages, and returns the results as a ranked list of typed result objects (`PageRanking`, `DocRanking`).

The abstract interface is:

```python
class PageRanker(ABC):
    def rank(self, queries: list[str], top_k: int = 100) -> list[PageRanking]: ...
```

Concrete rankers (`ColPageRanker`, `BiEncoderPageRanker`, `TfIdfDocRanker`, `BM25DocRanker`) pair specific embedders with specific retrieval backends. Keeping this composition explicit makes it straightforward to benchmark one configuration against another.
