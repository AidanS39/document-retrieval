import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from sqlalchemy import select, func
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from .models import EvaluationAnnotation


class RetrievalSystem:
    def __init__(
        self,
        id: int,
        name: str,
        embed_model: str,
        embed_paradigm: str,
        search_method: str,
        top_k: int,
        created_at: Optional[str] = None,
        rankings: Optional[dict[str, list[dict[str, int]]]] = None,
    ):
        self.id = id
        self.name = name
        self.embed_model = embed_model
        self.embed_paradigm = embed_paradigm
        self.search_method = search_method
        self.top_k = top_k
        self.created_at = created_at or datetime.now(timezone.utc).isoformat()
        self.rankings: dict[str, list[dict[str, int]]] = rankings or {}

    def add_ranking(self, query: str, ranks: list[dict[str, int]]):
        self.rankings[query] = ranks

    def export_to_json(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"system_{self.id}.json"
        data = {
            "id": self.id,
            "name": self.name,
            "embed_model": self.embed_model,
            "embed_paradigm": self.embed_paradigm,
            "search_method": self.search_method,
            "top_k": self.top_k,
            "created_at": self.created_at,
            "rankings": self.rankings,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return path

    @classmethod
    def import_from_json(cls, path: Path) -> "RetrievalSystem":
        with open(path) as f:
            data = json.load(f)
        return cls(
            id=data["id"],
            name=data["name"],
            embed_model=data["embed_model"],
            embed_paradigm=data["embed_paradigm"],
            search_method=data["search_method"],
            top_k=data.get("top_k", 10),
            created_at=data.get("created_at"),
            rankings=data.get("rankings", {}),
        )


class QueryPool:
    def __init__(
        self,
        top_k: int,
        system_ids: list[int],
        queries: list[dict],
        created_at: Optional[str] = None,
    ):
        self.top_k = top_k
        self.system_ids = system_ids
        self.queries = queries  # [{"id": int, "query": str, "pooled_page_ids": list[int]}]
        self.created_at = created_at or datetime.now(timezone.utc).isoformat()

    @classmethod
    def from_systems(
        cls,
        systems: list[RetrievalSystem],
        queries: list[str],
        top_k: int,
    ) -> "QueryPool":
        query_entries = []
        for query_id, query in enumerate(queries, start=1):
            pooled: set[int] = set()
            for system in systems:
                for rank in system.rankings.get(query, [])[:top_k]:
                    pooled.add(rank["page_id"])
            query_entries.append({
                "id": query_id,
                "query": query,
                "pooled_page_ids": sorted(pooled),
            })
        return cls(
            top_k=top_k,
            system_ids=[s.id for s in systems],
            queries=query_entries,
        )

    def get_pooled_page_ids(self, query_id: int) -> list[int]:
        for q in self.queries:
            if q["id"] == query_id:
                return q["pooled_page_ids"]
        return []

    def export_to_json(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "created_at": self.created_at,
            "top_k": self.top_k,
            "system_ids": self.system_ids,
            "queries": self.queries,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return path

    @classmethod
    def import_from_json(cls, path: Path) -> "QueryPool":
        with open(path) as f:
            data = json.load(f)
        return cls(
            top_k=data["top_k"],
            system_ids=data["system_ids"],
            queries=data["queries"],
            created_at=data.get("created_at"),
        )


class AnnotationStore:
    def __init__(self, engine):
        self.engine = engine

    def submit(self, annotator: str, query_id: int, query: str, page_id: int, score: int):
        stmt = (
            pg_insert(EvaluationAnnotation)
            .values(
                annotator=annotator,
                query_id=query_id,
                query=query,
                page_id=page_id,
                score=score,
            )
            .on_conflict_do_update(
                index_elements=["annotator", "query_id", "page_id"],
                set_={"score": score, "submitted_at": func.now()},
            )
        )
        with Session(self.engine) as session:
            session.execute(stmt)
            session.commit()

    def get_annotations(
        self,
        query_id: int,
        before: Optional[datetime] = None,
        after: Optional[datetime] = None,
    ) -> dict[str, dict[int, int]]:
        stmt = select(
            EvaluationAnnotation.annotator,
            EvaluationAnnotation.page_id,
            EvaluationAnnotation.score,
        ).where(EvaluationAnnotation.query_id == query_id)
        if before is not None:
            stmt = stmt.where(EvaluationAnnotation.submitted_at < before)
        if after is not None:
            stmt = stmt.where(EvaluationAnnotation.submitted_at > after)
        with Session(self.engine) as session:
            rows = session.execute(stmt).all()
        result: dict[str, dict[int, int]] = {}
        for annotator, page_id, score in rows:
            result.setdefault(annotator, {})[page_id] = score
        return result

    def get_annotated_queries(
        self,
        before: Optional[datetime] = None,
        after: Optional[datetime] = None,
    ) -> list[dict]:
        stmt = select(
            EvaluationAnnotation.query_id,
            EvaluationAnnotation.query,
            EvaluationAnnotation.page_id,
        ).distinct()
        if before is not None:
            stmt = stmt.where(EvaluationAnnotation.submitted_at < before)
        if after is not None:
            stmt = stmt.where(EvaluationAnnotation.submitted_at > after)
        with Session(self.engine) as session:
            rows = session.execute(stmt).all()
        queries: dict[int, dict] = {}
        for query_id, query, page_id in rows:
            if query_id not in queries:
                queries[query_id] = {"query_id": query_id, "query": query, "annotated_page_ids": []}
            queries[query_id]["annotated_page_ids"].append(page_id)
        return sorted(queries.values(), key=lambda q: q["query_id"])

    def get_all_annotators(self) -> list[str]:
        with Session(self.engine) as session:
            return list(session.execute(
                select(EvaluationAnnotation.annotator).distinct()
            ).scalars().all())

    def is_complete(self, pool: QueryPool, annotator: str) -> bool:
        for q in pool.queries:
            scored = self.get_annotations(q["id"]).get(annotator, {})
            if not all(pid in scored for pid in q["pooled_page_ids"]):
                return False
        return True


class NDCGComputer:
    def __init__(self, store: AnnotationStore):
        self.store = store

    def gold_scores(
        self,
        query_id: int,
        pooled_page_ids: list[int],
        aggregate: str = "mean",
        before: Optional[datetime] = None,
        after: Optional[datetime] = None,
    ) -> dict[int, float]:
        annotations = self.store.get_annotations(query_id, before=before, after=after)
        result: dict[int, float] = {}
        for page_id in pooled_page_ids:
            scores = [ann[page_id] for ann in annotations.values() if page_id in ann]
            if scores:
                avg = sum(scores) / len(scores)
                result[page_id] = float(round(avg)) if aggregate == "majority" else avg
            else:
                result[page_id] = 0.0
        return result

    def dcg(self, ranking: list[dict], gold: dict[int, float], k: int) -> float:
        total = 0.0
        for rank in ranking[:k]:
            rel = gold.get(rank["page_id"], 0.0)
            total += rel / math.log2(rank["position"] + 1)
        return total

    def idcg(self, pooled_page_ids: list[int], gold: dict[int, float], k: int) -> float:
        sorted_scores = sorted(
            (gold.get(pid, 0.0) for pid in pooled_page_ids), reverse=True
        )
        total = 0.0
        for i, rel in enumerate(sorted_scores[:k]):
            total += rel / math.log2(i + 2)
        return total

    def ndcg(
        self,
        ranking: list[dict],
        gold: dict[int, float],
        pooled_page_ids: list[int],
        k: int,
    ) -> float:
        idcg_val = self.idcg(pooled_page_ids, gold, k)
        if idcg_val == 0.0:
            return 0.0
        return self.dcg(ranking, gold, k) / idcg_val

    def compute(
        self,
        pool: QueryPool,
        systems: list[RetrievalSystem],
        k: int,
        aggregate: str = "mean",
        before: Optional[datetime] = None,
        after: Optional[datetime] = None,
        include_zero_idcg: bool = False,
    ) -> dict:
        results = {
            "computed_at": datetime.now(timezone.utc).isoformat(),
            "k": k,
            "score_aggregate": aggregate,
            "annotation_before": before.isoformat() if before else None,
            "annotation_after": after.isoformat() if after else None,
            "systems": [],
        }
        for system in systems:
            per_query = []
            for q in pool.queries:
                query_id = q["id"]
                pooled = q["pooled_page_ids"]
                gold = self.gold_scores(query_id, pooled, aggregate, before=before, after=after)
                ranking = system.rankings.get(q["query"], [])
                dcg_val = self.dcg(ranking, gold, k)
                idcg_val = self.idcg(pooled, gold, k)
                ndcg_val = dcg_val / idcg_val if idcg_val > 0 else 0.0
                per_query.append({
                    "query_id": query_id,
                    "query": q["query"],
                    "ndcg": ndcg_val,
                    "dcg": dcg_val,
                    "idcg": idcg_val,
                })
            scoreable = per_query if include_zero_idcg else [r for r in per_query if r["idcg"] > 0]
            mean_ndcg = sum(r["ndcg"] for r in scoreable) / len(scoreable) if scoreable else 0.0
            results["systems"].append({
                "system_id": system.id,
                "name": system.name,
                "mean_ndcg": mean_ndcg,
                "per_query": per_query,
            })
        return results

    def compute_pool_free(
        self,
        systems: list[RetrievalSystem],
        k: int,
        aggregate: str = "mean",
        before: Optional[datetime] = None,
        after: Optional[datetime] = None,
        include_zero_idcg: bool = False,
    ) -> dict:
        queries = self.store.get_annotated_queries(before=before, after=after)
        results = {
            "computed_at": datetime.now(timezone.utc).isoformat(),
            "k": k,
            "score_aggregate": aggregate,
            "annotation_before": before.isoformat() if before else None,
            "annotation_after": after.isoformat() if after else None,
            "pool_free": True,
            "systems": [],
        }
        for system in systems:
            per_query = []
            for q in queries:
                query_id = q["query_id"]
                annotated_page_ids = q["annotated_page_ids"]
                gold = self.gold_scores(query_id, annotated_page_ids, aggregate, before=before, after=after)
                ranking = system.rankings.get(q["query"], [])
                dcg_val = self.dcg(ranking, gold, k)
                idcg_val = self.idcg(annotated_page_ids, gold, k)
                ndcg_val = dcg_val / idcg_val if idcg_val > 0 else 0.0
                per_query.append({
                    "query_id": query_id,
                    "query": q["query"],
                    "ndcg": ndcg_val,
                    "dcg": dcg_val,
                    "idcg": idcg_val,
                })
            scoreable = per_query if include_zero_idcg else [r for r in per_query if r["idcg"] > 0]
            mean_ndcg = sum(r["ndcg"] for r in scoreable) / len(scoreable) if scoreable else 0.0
            results["systems"].append({
                "system_id": system.id,
                "name": system.name,
                "mean_ndcg": mean_ndcg,
                "per_query": per_query,
            })
        return results

    def export_results(self, results: dict, out_dir: Path) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = out_dir / f"ndcg_results_{timestamp}.json"
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        return path
