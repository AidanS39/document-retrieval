from sklearn.metrics.pairwise import cosine_similarity

class TfIdfIndexer:
    def __init__(self):
        self.embeddings = None
        self.page_ids: list[int] = []

    def index_pages(self, embeddings, valid_ids: list[int]):
        self.embeddings = embeddings
        self.page_ids = valid_ids

    def retrieve(self, query_embeddings, top_k: int = 25):
        all_scores = cosine_similarity(query_embeddings, self.embeddings)
        results = []
        for scores in all_scores:
            top_indices = sorted(range(len(self.page_ids)), key=lambda i: scores[i], reverse=True)[:top_k]
            results.append([(self.page_ids[i], float(scores[i])) for i in top_indices])
        return results


