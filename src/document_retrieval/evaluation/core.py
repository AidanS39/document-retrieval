import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from document_retrieval.models import Query

class RetrievalSystem:
    # system_{id}.json structure:
    # {
    #   "id": int,
    #   "name": str,
    #   "embed_model": str,
    #   "embed_paradigm": str,
    #   "search_method": str,
    #   "top_k": int,
    #   "created_at": str,  # ISO 8601 timestamp
    #   "rankings": {
    #     "<query.id>": [  # query table primary key, stored as a string since json keys are strings
    #       {"page_id": int, "position": int},
    #       ...
    #     ],
    #     ...
    #   }
    # }
    def __init__(
        self,
        id: int,
        name: str,
        embed_model: str,
        embed_paradigm: str,
        search_method: str,
        top_k: int,
        created_at: Optional[str] = None,
        rankings: Optional[dict[int, list[dict[str, int]]]] = None,
    ):
        self.id = id
        self.name = name
        self.embed_model = embed_model
        self.embed_paradigm = embed_paradigm
        self.search_method = search_method
        self.top_k = top_k
        self.created_at = created_at or datetime.now(timezone.utc).isoformat()
        self.rankings: dict[int, list[dict[str, int]]] = rankings or {}  # keyed by query.id

    def add_ranking(self, query_id: int, ranks: list[dict[str, int]]):
        self.rankings[query_id] = ranks

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
            rankings={int(query_id): ranks for query_id, ranks in data.get("rankings", {}).items()},
        )


class QueryPool:
    def __init__(
        self,
        top_k: int,
        system_ids: list[int],
        queries: list[dict], # [{"id": int, "query": str, "pooled_page_ids": list[int]}]
        created_at: Optional[str] = None,
    ):
        self.top_k = top_k
        self.system_ids = system_ids
        self.queries = queries
        self.created_at = created_at or datetime.now(timezone.utc).isoformat()

    @classmethod
    def from_systems(
        cls,
        systems: list[RetrievalSystem],
        queries: list[Query],
        top_k: int,
    ) -> "QueryPool":
        query_entries = []
        for query in queries:
            pooled: set[int] = set()
            for system in systems:
                ranks = sorted(system.rankings.get(query.id, []), key=lambda rank: rank["position"])
                for rank in ranks[:top_k]:
                    pooled.add(rank["page_id"])
            query_entries.append({
                "id": query.id,
                "query": query.text,
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
