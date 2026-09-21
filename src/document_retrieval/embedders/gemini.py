from google import genai
from google.genai import types as genai_types
from google.oauth2 import service_account

_VERTEX_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]

def _load_vertex_credentials():
    raw = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
    if raw:
        try:
            info = json.loads(raw.strip())
        except json.JSONDecodeError:
            info = json.loads(base64.b64decode(raw.strip()).decode())
        return service_account.Credentials.from_service_account_info(info, scopes=_VERTEX_SCOPES)
    path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if path:
        return service_account.Credentials.from_service_account_file(path, scopes=_VERTEX_SCOPES)
    return None


class GeminiBiEncoderPageEmbedder(PageEmbedder):
    def __init__(self, model_name: str, engine: Engine, device: torch.device, data_dir: Path):
        super().__init__(engine)
        self.model_name = model_name
        self.data_dir = data_dir
        self.client = genai.Client(
            vertexai=True,
            project=os.environ.get("GCP_PROJECT_ID"),
            location=os.environ.get("VERTEX_LOCATION", "us"),
            credentials=_load_vertex_credentials(),
        )
        self.metadata = PipelineMetadata.load(model_name, data_dir / "embeddings")

    def _preprocess_batch(self, page_ids: list[int]):
        with Session(self.engine) as session:
            pages = session.execute(
                select(Page.id, Page.image_path).where(Page.id.in_(page_ids))
            ).all()

        successful_page_ids, failed_page_ids, images = [], [], []
        for page_id, image_path in pages:
            try:
                images.append(load_image(image_path))
                successful_page_ids.append(page_id)
            except Exception as e:
                failed_page_ids.append(page_id)
                print(f"WARNING: page image could not be loaded for {image_path}: {e}")

        return images, successful_page_ids, failed_page_ids

    @timefunction
    def _embedding_pipeline(self, images: list[Image]) -> torch.Tensor:
        config = genai_types.EmbedContentConfig(output_dimensionality=1536)

        def embed_one(img):
            result = self.client.models.embed_content(
                model=self.model_name, contents=img, config=config
            )
            return result.embeddings[0].values

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            all_values = list(executor.map(embed_one, images))

        return torch.tensor(all_values)

    @timefunction
    def embed_pages(self, page_ids: list[int], batch_size: int = 64):
        i = 0
        while i < len(page_ids):
            batch_ids = page_ids[i : i + batch_size]
            images, successful_ids, _ = self._preprocess_batch(batch_ids)

            if not successful_ids:
                print(
                    f"WARNING: no page images could be loaded for page batch {batch_ids}. skipping embedding generation for batch."
                )
            else:
                with Timer() as timer:
                    embeddings = self._embedding_pipeline(images)

                batch_telemetry = EmbeddingBatchTelemetry(timer, successful_ids, embeddings.shape)
                batch_metadata = BatchMetadata(len(self.metadata.batches), successful_ids, batch_telemetry)
                self.metadata.add_batch(batch_metadata)
                self.metadata.save()

                torch.save(
                    {"embeddings": embeddings, "page_ids": successful_ids},
                    self.metadata.embeddings_path / f"batch_{batch_metadata.id}.pt",
                )

            i += batch_size

        self.metadata.save()
        self.metadata.print_telemetry_summary()

    def embed_queries(self, queries: list[str]) -> list[list[float]]:
        config = genai_types.EmbedContentConfig(output_dimensionality=1536)
        all_values = []
        for q in queries:
            result = self.client.models.embed_content(
                model=self.model_name,
                contents=f"Represent this retrieval query for finding relevant documents: {q}",
                config=config,
            )
            all_values.append(result.embeddings[0].values)
        return all_values
