import argparse
import os
from dotenv import load_dotenv
from pathlib import Path

from document_retrieval.benchmarking import PipelineMetadata
from scripts.config import MODEL_REGISTRY, add_embedder_arg

load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Print pipeline telemetry summary for a given model")
    add_embedder_arg(parser)
    args = parser.parse_args()

    data_dir = Path(os.getenv("DATA_DIR", "/app/data"))
    embeddings_dir = data_dir / "embeddings"

    metadata = PipelineMetadata.load(args.model, embeddings_dir)
    metadata.print_telemetry_summary()


if __name__ == "__main__":
    main()
