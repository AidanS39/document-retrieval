import bm25s

class BM25Indexer:
    def __init__(self):
        self.index = bm25s.BM25()
        self.page_ids: list[int] = []

    def index_pages(self, tokenized_texts, valid_ids: list[int]):
        self.page_ids = valid_ids
        self.index.index(tokenized_texts)

    def retrieve(self, tokenized_queries, top_k: int = 25):
        results_indices, scores = self.index.retrieve(tokenized_queries, k=top_k)
        results = []
        for query_i, result_indices in enumerate(results_indices):
            results.append([
                (self.page_ids[page_i], float(scores[query_i][result_i]))
                for result_i, page_i in enumerate(result_indices)
            ])
        return results

