import math

class NDCG:
    @staticmethod
    def cg(ranking: list[dict], gold_scores: dict[int, float], k: int) -> float:
        cg = 0.0
        ranking = sorted(ranking, key=lambda rank: rank["position"])

        for rank in ranking[:k]:
            gold_score = gold_scores[rank["page_id"]]
            cg += gold_score

        return cg

    @staticmethod
    def dcg(ranking: list[dict], gold_scores: dict[int, float], k: int) -> float:
        dcg = 0.0
        ranking = sorted(ranking, key=lambda rank: rank["position"])

        for rank in ranking[:k]:
            gold_score = gold_scores[rank["page_id"]]
            position = math.log2(rank["position"] + 1)
            dcg += gold_score / position

        return dcg

    @staticmethod
    def idcg(gold_scores: dict[int, float], k: int) -> float:
        gold_score_values = sorted(list(gold_scores.values()), reverse=True)

        idcg = 0.0
        # positions are 1-based to match rank["position"] in dcg
        for position, gold_score in enumerate(gold_score_values[:k], start=1):
            idcg += gold_score / math.log2(position + 1)

        return idcg

    @staticmethod
    def ndcg(ranking: list[dict], gold_scores: dict[int, float], k: int) -> float:
        dcg_score = NDCG.dcg(ranking, gold_scores, k)
        idcg_score = NDCG.idcg(gold_scores, k)

        # no relevant pages for this query, so there is nothing to normalize against
        if idcg_score == 0:
            return 0.0

        ndcg = dcg_score / idcg_score

        return ndcg
