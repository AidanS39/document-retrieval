import pprint
from math import sqrt
import argparse
import csv
import os
from collections import Counter, defaultdict
from itertools import combinations, combinations_with_replacement
from pathlib import Path
import scipy.stats as stats
import numpy as np

from dotenv import load_dotenv
from sqlalchemy import select, func, Engine
from sqlalchemy.engine import URL, create_engine
from sqlalchemy.orm import Session

from document_retrieval.evaluation import QueryPool, RetrievalSystem
from document_retrieval.evaluation.data import load_annotation_data_2
from document_retrieval.evaluation.annotations import compute_pairwise_metric
from document_retrieval.evaluation.retrieval import NDCG
from document_retrieval.evaluation.annotations import agreement, mae, mse, rmse, mean_bias, cohens_kappa, kendalls_tau, aaps
from document_retrieval.models import Annotator, EvaluationAnnotation, Query

load_dotenv()

SCORE_LABELS = {
    0: "Not relevant", 
    1: "Marginally", 
    2: "Relevant", 
    3: "Highly relevant"
}
ROLE_DISPLAY = {
    None: "all",
    "human": "human",
    "ai": "ai"
}


def arguments(eval_dir):
    parser = argparse.ArgumentParser(description="Display statistics about evaluation annotations")
    parser.add_argument(
        "--pool",
        type=Path,
        default=eval_dir / "query_pool.json",
        help="Path to query_pool.json for completion stats (default: $DATA_DIR/evaluation/query_pool.json)",
    )
    parser.add_argument(
        "--systems-dir",
        type=Path,
        default=eval_dir / "systems",
        help="Directory containing system_*.json files for NDCG display (default: $DATA_DIR/evaluation/systems)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Cutoff for NDCG@k (default: 5)",
    )
    parser.add_argument(
        "--aggregate",
        choices=["mean", "majority"],
        default="mean",
        help="How to aggregate annotator scores into gold scores for continuous NDCG (default: mean)",
    )
    parser.add_argument(
        "--ndcg-mode",
        choices=["discrete", "continuous"],
        default="discrete",
        dest="ndcg_mode",
        help="discrete: compute NDCG per annotator then average; continuous: average scores first then compute NDCG (default: discrete)",
    )
    parser.add_argument(
        "--annotators",
        choices=["all", "human", "ai"],
        default="all",
        help="Restrict all statistics to human annotators, AI annotators, or both (default: all)",
    )
    parser.add_argument(
        "--agree-metric",
        choices=["exact", "rmse"],
        default="exact",
        dest="agree_metric",
        help="Metric for observed agreement: exact match rate or RMSE of score differences (default: exact)",
    )

    parser.add_argument(
        "--exclude",
        nargs="+",
        default=[],
        metavar="NAME",
        help="Annotator names to exclude from all calculations. Example: --exclude \"Alice Smith\" \"Bob Jones\"",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        metavar="PATH",
        help="Write per-annotator score distribution to a CSV file at this path",
    )
    args = parser.parse_args()

    return args

def hr(char="=", width=64):
    print(char * width)


def section(title: str):
    print()
    print(f"--- {title} ---")

def role_pair_key(a_role: str, b_role: str) -> tuple[str, str]:
    # order-independent key so (human, ai) and (ai, human) land in the same group
    return (a_role, b_role) if a_role <= b_role else (b_role, a_role)

def weighted_kappa(a_scores: list[int], b_scores: list[int]) -> float | None:
    # quadratic weights: kappa = 1 - observed MSE / MSE expected if both annotators scored independently,
    # each keeping their own score distribution
    n = len(a_scores)
    a_counts = Counter(a_scores)
    b_counts = Counter(b_scores)
    observed = sum((a - b) ** 2 for a, b in zip(a_scores, b_scores)) / n
    expected = sum(
        a_count * b_count * (a - b) ** 2
        for a, a_count in a_counts.items()
        for b, b_count in b_counts.items()
    ) / n ** 2

    # both annotators gave every page the same single score, so chance agreement is perfect and kappa is undefined
    if expected == 0:
        return None

    return 1 - observed / expected

# def kendalls_tau(a_scores: list[int], b_scores: list[int]) -> float | None:
#     tau, _ = stats.kendalltau(a_scores, b_scores, variant="b")
#     return None if np.isnan(tau) else float(tau)

def fleiss_kappa(ratings: list[list[int]], categories: list[int]) -> float | None:
    """
    Generalized Fleiss' kappa supporting varying numbers of raters per subject.
    ratings[i] is the list of scores assigned to subject i by all raters who rated it.
    Returns None if kappa is undefined (fewer than 2 raters on every subject, or all
    raters always agree on the same single category).
    """
    cat_index = {c: i for i, c in enumerate(categories)}

    # counts[i][j] = number of raters assigning category j to subject i
    counts = np.zeros((len(ratings), len(categories)), dtype=int)
    for i, subject_ratings in enumerate(ratings):
        for score in subject_ratings:
            if score in cat_index:
                counts[i, cat_index[score]] += 1

    n_per_subject = counts.sum(axis=1)

    # subjects rated by fewer than 2 raters contribute no agreement information
    valid = n_per_subject >= 2
    if not valid.any():
        return None

    counts = counts[valid]
    n_per_subject = n_per_subject[valid]

    # marginal proportion of each category across all ratings
    p_j = counts.sum(axis=0) / n_per_subject.sum()

    # expected agreement by chance
    p_e = float((p_j ** 2).sum())
    if p_e == 1.0:
        return None

    # observed agreement, generalized for varying rater counts per subject:
    # P_bar = Σ_i Σ_j n_ij*(n_ij-1)  /  Σ_i n_i*(n_i-1)
    agreement_numerator   = int((counts * (counts - 1)).sum())
    agreement_denominator = int((n_per_subject * (n_per_subject - 1)).sum())
    if agreement_denominator == 0:
        return None

    p_bar = agreement_numerator / agreement_denominator
    return (p_bar - p_e) / (1 - p_e)

def krippendorffs_alpha(ratings: list[list[int]], categories: list[int]) -> float | None:
    """
    Krippendorff's alpha with quadratic (interval) metric.
    ratings[i] is the list of scores assigned to subject i by all raters who rated it.
    Returns None if alpha is undefined (no subjects with 2+ raters, or zero expected disagreement).
    """
    cat_index = {c: j for j, c in enumerate(categories)}
    C = len(categories)

    counts = np.zeros((len(ratings), C), dtype=float)
    for i, subject_ratings in enumerate(ratings):
        for score in subject_ratings:
            if score in cat_index:
                counts[i, cat_index[score]] += 1

    n_per_subject = counts.sum(axis=1)
    valid = n_per_subject >= 2
    if not valid.any():
        return None

    counts = counts[valid]
    n_per_subject = n_per_subject[valid]

    # coincidence matrix: O[k][l] = weighted count of (k,l) rating pairs across all subjects
    # weight per subject i is 1/(n_i-1) so each subject contributes n_i total to O
    O = np.zeros((C, C), dtype=float)
    for i in range(len(counts)):
        w = 1.0 / (n_per_subject[i] - 1)
        for k in range(C):
            for l in range(C):
                if k == l:
                    O[k, l] += w * counts[i, k] * (counts[i, k] - 1)
                else:
                    O[k, l] += w * counts[i, k] * counts[i, l]

    n = O.sum()  # equals total number of ratings Σ_i n_i
    if n == 0:
        return None

    n_k = O.sum(axis=1)  # marginal totals per category

    # quadratic distance matrix
    d = np.array([[(categories[k] - categories[l]) ** 2 for l in range(C)] for k in range(C)], dtype=float)

    D_o = float((d * O).sum()) / n
    D_e = float((d * np.outer(n_k, n_k)).sum()) / (n * (n - 1))

    if D_e == 0:
        return None

    return 1 - D_o / D_e

def annotator_report(query_pool: QueryPool, role: str | None, engine, csv_path: Path | None = None):
    with Session(engine) as session:
        active_ids = set(session.scalars(select(Query.id).where(Query.active)))

        annotator_stmt = select(Annotator.id, Annotator.name, Annotator.role)
        annotation_stmt = (
            select(
                Annotator.name,
                EvaluationAnnotation.query_id,
                EvaluationAnnotation.page_id,
                EvaluationAnnotation.score,
            )
            .join(Annotator, Annotator.id == EvaluationAnnotation.annotator_id)
            .join(Query, Query.id == EvaluationAnnotation.query_id)
            .where(Query.active)
        )
        if role is not None:
            annotator_stmt = annotator_stmt.where(Annotator.role == role)
            annotation_stmt = annotation_stmt.where(Annotator.role == role)

        # annotator name -> role, annotator name -> id
        annotator_roles: dict[str, str] = {}
        annotator_ids: dict[str, int] = {}
        for ann_id, ann_name, ann_role in session.execute(annotator_stmt).tuples().all():
            annotator_roles[ann_name] = ann_role
            annotator_ids[ann_name] = ann_id
        rows = session.execute(annotation_stmt).all()

    annotator_names = sorted(annotator_roles)
    roles = sorted(set(annotator_roles.values()))

    # only active queries in the pool count towards completion and agreement
    pooled_queries = [q for q in query_pool.queries if q["id"] in active_ids]
    query_page_pool = {(q["id"], page_id) for q in pooled_queries for page_id in q["pooled_page_ids"]}
    total_pages = len(query_page_pool)

    # (query_id, page_id) -> annotator name -> score
    page_scores: dict[tuple[int, int], dict[str, int]] = defaultdict(dict)
    # annotator name -> set of (query_id, page_id) they scored
    annotator_pages: dict[str, set[tuple[int, int]]] = defaultdict(set)
    # role -> score -> number of annotations with that score
    score_counts_by_role: dict[str, Counter] = {r: Counter() for r in roles}
    # annotator name -> score -> count
    score_counts_by_annotator: dict[str, Counter] = {n: Counter() for n in annotator_names}
    outside_pool = 0
    for name, query_id, page_id, score in rows:
        if (query_id, page_id) not in query_page_pool:
            outside_pool += 1
            continue
        page_scores[(query_id, page_id)][name] = score
        annotator_pages[name].add((query_id, page_id))
        score_counts_by_role[annotator_roles[name]][score] += 1
        score_counts_by_annotator[name][score] += 1

    total_annotations = len(rows) - outside_pool
    ann_col = max([len(name) for name in annotator_names] + [10])

    # === Totals ===
    section("Totals")
    print(f"  Annotator filter   : {role or 'all'}")
    print(f"  Annotators         : {len(annotator_names)}")
    print(f"  Annotations        : {total_annotations}")
    print(f"  Pool               : {len(pooled_queries)} active queries, {total_pages} pages")
    if outside_pool:
        print(f"  ({outside_pool} annotation(s) on pages outside the active query pool were not counted)")

    # === Annotations per Query ===
    section("Annotations per Query  (pages annotated by at least one annotator)")
    print(f"  {'ID':>4}  {'Annotated':>9}  {'Pct':>6}  Query")
    print("  " + "-" * 70)
    for q in pooled_queries:
        pooled = q["pooled_page_ids"]
        annotated = sum(1 for page_id in pooled if (q["id"], page_id) in page_scores)
        pct = annotated / len(pooled) * 100 if pooled else 0.0
        query_text = q["query"]
        truncated = query_text[:44] + "..." if len(query_text) > 44 else query_text
        print(f"  {q['id']:>4}  {annotated:>4}/{len(pooled):<4}  {pct:>5.1f}%  {truncated}")

    # === Annotations per Annotator ===
    section("Annotations per Annotator")
    print(f"  {'Annotator':<{ann_col}}  {'Annotated':>11}  {'Pct':>6}")
    print("  " + "-" * (ann_col + 23))
    for name in annotator_names:
        annotated = len(annotator_pages[name])
        pct = annotated / total_pages * 100 if total_pages else 0.0
        print(f"  {name:<{ann_col}}  {annotated:>5}/{total_pages:<5}  {pct:>5.1f}%")

    # === Score Distribution ===
    section("Score Distribution")
    role_col = max([len(r) for r in roles] + [len("all")])
    print(f"  {'Role':<{role_col}}  {'N':>6}  " + "  ".join(f"{score:>14}" for score in SCORE_LABELS) + f"  {'Mean':>5}")
    print("  " + "-" * (role_col + 16 * len(SCORE_LABELS) + 17))

    def print_distribution(label: str, counts: Counter):
        n = sum(counts.values())
        if n == 0:
            print(f"  {label:<{role_col}}  {n:>6}  " + "  ".join(f"{'-':>14}" for _ in SCORE_LABELS) + f"  {'-':>5}")
            return
        cells = "  ".join(f"{counts[score]:>6} ({counts[score] / n * 100:>5.1f}%)" for score in SCORE_LABELS)
        mean = sum(score * count for score, count in counts.items()) / n
        print(f"  {label:<{role_col}}  {n:>6}  {cells}  {mean:>5.2f}")

    print_distribution("all", sum(score_counts_by_role.values(), Counter()))
    for r in roles:
        print_distribution(r, score_counts_by_role[r])
    print()
    print("  " + "  ".join(f"{score} = {label}" for score, label in SCORE_LABELS.items()))

    # === Per-Annotator Score Distribution ===
    section("Score Distribution by Annotator")
    print(f"  {'Annotator':<{ann_col}}  {'Role':<{role_col}}  {'N':>6}  " + "  ".join(f"{score:>14}" for score in SCORE_LABELS) + f"  {'Mean':>5}")
    print("  " + "-" * (ann_col + role_col + 16 * len(SCORE_LABELS) + 20))
    for name in annotator_names:
        counts = score_counts_by_annotator[name]
        n = sum(counts.values())
        r = annotator_roles[name]
        if n == 0:
            cells = "  ".join(f"{'-':>14}" for _ in SCORE_LABELS)
            print(f"  {name:<{ann_col}}  {r:<{role_col}}  {n:>6}  {cells}  {'-':>5}")
        else:
            cells = "  ".join(f"{counts[score]:>6} ({counts[score] / n * 100:>5.1f}%)" for score in SCORE_LABELS)
            mean = sum(score * count for score, count in counts.items()) / n
            print(f"  {name:<{ann_col}}  {r:<{role_col}}  {n:>6}  {cells}  {mean:>5.2f}")

    if csv_path is not None:
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "name", "role", "score", "total"])
            for name in annotator_names:
                ann_id = annotator_ids[name]
                r = annotator_roles[name]
                counts = score_counts_by_annotator[name]
                for score in SCORE_LABELS:
                    writer.writerow([ann_id, name, r, score, counts[score]])
        print(f"\n  Score distribution written to {csv_path}")

    # === Pairwise Agreement ===
    # every pair of annotators that scored the same (query, page) contributes one |score difference|,
    # which is pooled overall and into the group for that pair's roles
    role_pairs = list(combinations_with_replacement(roles, 2))
    diffs_by_role_pair: dict[tuple[str, str], list[int]] = {pair: [] for pair in role_pairs}
    all_diffs: list[int] = []
    multi_annotated = 0

    for scores in page_scores.values():
        if len(scores) < 2:
            continue
        multi_annotated += 1
        for (a, a_score), (b, b_score) in combinations(scores.items(), 2):
            diff = abs(a_score - b_score)
            all_diffs.append(diff)
            role_pair = role_pair_key(annotator_roles[a], annotator_roles[b])
            diffs_by_role_pair[role_pair].append(diff)

    section("Pairwise Agreement  (score differences between annotators on the same page)")
    print(f"  Pages scored by 2+ annotators : {multi_annotated}")
    if not all_diffs:
        print("  No pages scored by multiple annotators yet.")
        return

    def print_row(label: str, diffs: list[int]):
        if diffs:
            mae = f"{sum(diffs) / len(diffs):>6.3f}"
            mse = f"{sum(d ** 2 for d in diffs) / len(diffs):>6.3f}"
        else:
            mae = mse = f"{'-':>6}"
        print(f"  {label:<{pair_col}}  {len(diffs):>6}  {mae}  {mse}")

    pair_col = max([len(f"{a} - {b}") for a, b in role_pairs] + [len("all")])
    print()
    print(f"  {'Roles':<{pair_col}}  {'Pairs':>6}  {'MAE':>6}  {'MSE':>6}")
    print("  " + "-" * (pair_col + 25))
    print_row("all", all_diffs)
    for role_pair in role_pairs:
        print_row(f"{role_pair[0]} - {role_pair[1]}", diffs_by_role_pair[role_pair])
    print()
    print("  Pairs: score differences pooled across pages.")

def get_gold_scores(query_pool: QueryPool, aggregate: str, role: str | None, engine, exclude_names: list[str] = []) -> dict[int, dict[int, float]]:
    # create gold scores dictionary: query_id -> page_id -> aggregated score
    gold_scores: dict[int, dict[int, float]] = {}
    with Session(engine) as session:
        active_ids = set(session.scalars(select(Query.id).where(Query.active)))

        for query in query_pool.queries:
            if query["id"] not in active_ids:
                continue

            stmt = (
                select(EvaluationAnnotation.page_id, EvaluationAnnotation.score)
                .join(Annotator, Annotator.id == EvaluationAnnotation.annotator_id)
                .where(EvaluationAnnotation.query_id == query["id"])
                .where(EvaluationAnnotation.page_id.in_(query["pooled_page_ids"]))
            )
            if role is not None:
                stmt = stmt.where(Annotator.role == role)
            if exclude_names:
                stmt = stmt.where(Annotator.name.not_in(exclude_names))

            scores_by_page: dict[int, list[int]] = defaultdict(list)
            for page_id, score in session.execute(stmt):
                scores_by_page[page_id].append(score)

            # unannotated pooled pages count as not relevant
            page_scores = {page_id: 0.0 for page_id in query["pooled_page_ids"]}
            for page_id, scores in scores_by_page.items():
                if aggregate == "majority":
                    # mode of the scores; on a tie, average the tied scores
                    counts = Counter(scores)
                    top = max(counts.values())
                    modes = [s for s, c in counts.items() if c == top]
                    page_scores[page_id] = sum(modes) / len(modes)
                else:
                    page_scores[page_id] = sum(scores) / len(scores)
            gold_scores[query["id"]] = page_scores

    return gold_scores

def get_per_annotator_gold_scores(query_pool: QueryPool, role: str | None, engine, exclude_names: list[str] = []) -> dict[str, dict[int, dict[int, int]]]:
    """Returns annotator_name -> query_id -> page_id -> score. Unannotated pooled pages default to 0.
    Only includes queries the annotator scored at least one page for."""
    with Session(engine) as session:
        active_ids = set(session.scalars(select(Query.id).where(Query.active)))

        stmt = (
            select(Annotator.name, EvaluationAnnotation.query_id, EvaluationAnnotation.page_id, EvaluationAnnotation.score)
            .join(Annotator, Annotator.id == EvaluationAnnotation.annotator_id)
            .join(Query, Query.id == EvaluationAnnotation.query_id)
            .where(Query.active)
        )
        if role is not None:
            stmt = stmt.where(Annotator.role == role)
        if exclude_names:
            stmt = stmt.where(Annotator.name.not_in(exclude_names))
        rows = session.execute(stmt).all()

    # annotator_name -> query_id -> page_id -> score (only scored pages)
    raw: dict[str, dict[int, dict[int, int]]] = defaultdict(lambda: defaultdict(dict))
    for ann_name, query_id, page_id, score in rows:
        raw[ann_name][query_id][page_id] = score

    # fill in unscored pooled pages with 0 for each query the annotator touched
    result: dict[str, dict[int, dict[int, int]]] = {}
    pooled_pages_by_query = {q["id"]: q["pooled_page_ids"] for q in query_pool.queries if q["id"] in active_ids}
    for ann_name, query_scores in raw.items():
        result[ann_name] = {}
        for query_id, scored_pages in query_scores.items():
            if query_id not in pooled_pages_by_query:
                continue
            page_scores = {page_id: 0 for page_id in pooled_pages_by_query[query_id]}
            page_scores.update(scored_pages)
            result[ann_name][query_id] = page_scores

    return result

def calculate_system_ndcgs(system: RetrievalSystem, gold_scores: dict[int, dict[int, float]], k: int, active_query_ids: list[int]) -> list[float]:
    ndcgs = []
    for query_id in system.rankings:
        if query_id in active_query_ids:
            ranking = system.rankings[query_id]
            query_gold_scores = gold_scores[query_id]
            ndcg = NDCG.ndcg(ranking, query_gold_scores, k)
            ndcgs.append(ndcg)

    return ndcgs

class SystemResult:
    def __init__(self, system: RetrievalSystem, ndcgs: list[float], avg_ndcg: float):
        self.system = system
        self.ndcgs = ndcgs
        self.avg_ndcg = avg_ndcg

def report_ndcgs(systems: list[RetrievalSystem], query_pool: QueryPool, roles: list[str | None], aggregate: str, k: int, engine, ndcg_mode: str = "discrete", exclude_names: list[str] = []):

    stmt = select(Query.id).where(Query.active == True)
    with Session(engine) as session:
        active_query_ids = list(session.execute(stmt).scalars().all())

    for role in roles:
        if ndcg_mode == "continuous":
            gold_scores = get_gold_scores(query_pool, aggregate, role, engine, exclude_names)
            scorable_query_ids = [qid for qid in active_query_ids if NDCG.idcg(gold_scores[qid], k) > 0]

            ann_stmt = select(func.count(func.distinct(Annotator.id))).join(
                EvaluationAnnotation, EvaluationAnnotation.annotator_id == Annotator.id
            ).join(Query, Query.id == EvaluationAnnotation.query_id).where(Query.active)
            if role is not None:
                ann_stmt = ann_stmt.where(Annotator.role == role)
            if exclude_names:
                ann_stmt = ann_stmt.where(Annotator.name.not_in(exclude_names))
            with Session(engine) as session:
                n_annotators = session.scalar(ann_stmt)

            system_results = []
            for system in systems:
                ndcgs = calculate_system_ndcgs(system, gold_scores, k, scorable_query_ids)
                avg_ndcg = sum(ndcgs) / len(ndcgs) if ndcgs else 0.0
                system_results.append(SystemResult(system, ndcgs, avg_ndcg))

            mode_label = f"continuous, {aggregate} aggregate, {n_annotators} annotators, {len(scorable_query_ids)} scoreable queries"
        else:  # discrete
            per_annotator_scores = get_per_annotator_gold_scores(query_pool, role, engine, exclude_names)

            system_results = []
            for system in systems:
                annotator_ndcgs = []
                for ann_gold_scores in per_annotator_scores.values():
                    scorable = [
                        qid for qid in ann_gold_scores
                        if qid in system.rankings and NDCG.idcg(ann_gold_scores[qid], k) > 0
                    ]
                    if not scorable:
                        continue
                    ndcgs = [NDCG.ndcg(system.rankings[qid], ann_gold_scores[qid], k) for qid in scorable]
                    annotator_ndcgs.append(sum(ndcgs) / len(ndcgs))
                avg_ndcg = sum(annotator_ndcgs) / len(annotator_ndcgs) if annotator_ndcgs else 0.0
                system_results.append(SystemResult(system, annotator_ndcgs, avg_ndcg))

            mode_label = f"discrete, {len(per_annotator_scores)} annotators"

        system_results = sorted(system_results, key=lambda result: result.avg_ndcg, reverse=True)
        print()
        section(f"Average NDCG@{k} by System: {ROLE_DISPLAY[role]} ({mode_label})")
        print(f" {"System":^35} {"NDCG@" + str(k):>6}")
        print(" " + "-" * 42)
        for result in system_results:
            print(f" {result.system.name:<35} {result.avg_ndcg:>6.4f}")
        print()

def load_systems(systems_dir: Path) -> list[RetrievalSystem]:
    systems = []

    if systems_dir.exists():
        system_files = sorted(systems_dir.glob("system_*.json"))
        if system_files:
            systems = [RetrievalSystem.import_from_json(p) for p in system_files]

    return systems

def build_roles_list(annotators: str) -> list[str | None]:
    if annotators == "all":
        return [None, "human", "ai"]
    return [annotators]


def load_annotation_data(query_pool: QueryPool, engine, exclude_names: list[str] = []) -> tuple:
    """Fetch and prepare annotation data shared by all report functions.

    Returns: (humans, ais, human_scores, ai_scores, human_query_pages_by_annotator)
      humans/ais                  : lists of Annotator (excluding excluded_names, restricted to those with annotations)
      human_scores                : (query_id, page_id) -> {annotator_id: score}
      ai_scores                   : (query_id, page_id) -> {annotator_id: score}
      human_query_pages_by_annotator    : annotator_id -> set of (query_id, page_id)
    Only pooled pairs from active queries are included.
    """
    with Session(engine) as session:
        active_query_ids = set(session.scalars(select(Query.id).where(Query.active)))

    # set of all (query_id, page_id) pairs from query pool
    query_page_pool: set[tuple[int, int]] = {
        (query["id"], page_id)
        for query in query_pool.queries
        if query["id"] in active_query_ids
        for page_id in query["pooled_page_ids"]
    }

    with Session(engine) as session:
        all_annotators = session.execute(select(Annotator)).scalars().all()

        ann_rows = session.execute(
            select(
                EvaluationAnnotation.annotator_id,
                EvaluationAnnotation.query_id,
                EvaluationAnnotation.page_id,
                EvaluationAnnotation.score,
                Annotator.role,
            )
            .join(Annotator, EvaluationAnnotation.annotator_id == Annotator.id)
            .join(Query, EvaluationAnnotation.query_id == Query.id)
            .where(Query.active)
        ).all()

    # filter excluded annotators
    excluded_ids = {a.id for a in all_annotators if a.name in exclude_names}
    all_annotators = [a for a in all_annotators if a.id not in excluded_ids]
    ann_rows = [row for row in ann_rows if row.annotator_id not in excluded_ids]
    
    # filter annotators that do not have any annotations
    annotator_ids_with_annotations = {row.annotator_id for row in ann_rows}
    all_annotators = [a for a in all_annotators if a.id in annotator_ids_with_annotations]

    humans = [a for a in all_annotators if a.role == "human"]
    ais    = [a for a in all_annotators if a.role == "ai"]

    human_scores: dict[tuple[int, int], dict[int, int]] = defaultdict(dict)
    ai_scores:    dict[tuple[int, int], dict[int, int]] = defaultdict(dict)
    human_query_pages_by_annotator: dict[int, set[tuple[int, int]]] = defaultdict(set)

    for row in ann_rows:
        query_page = (row.query_id, row.page_id)
        if query_page not in query_page_pool:
            continue
        if row.role == "human":
            human_scores[query_page][row.annotator_id] = row.score
            human_query_pages_by_annotator[row.annotator_id].add(query_page)
        elif row.role == "ai":
            ai_scores[query_page][row.annotator_id] = row.score

    return humans, ais, human_scores, ai_scores, human_query_pages_by_annotator


def report_fleiss_kappa(query_pool: QueryPool, engine, exclude_names: list[str] = []):
    humans, ais, human_scores, ai_scores, _ = load_annotation_data(query_pool, engine, exclude_names)

    categories = list(SCORE_LABELS.keys())  # [0, 1, 2, 3]

    section("Fleiss' Kappa")
    if exclude_names:
        print(f"  Excluded from calculations: {', '.join(exclude_names)}")
    print(f"  {'Rater group':<30}  {'Fleiss\' Kappa':>13}")
    print("  " + "-" * 45)

    # Baseline: humans only, restricted to pairs with at least 2 human raters
    human_ratings = [
        list(scores.values())
        for scores in human_scores.values()
        if len(scores) >= 2
    ]
    human_kappa = fleiss_kappa(human_ratings, categories)
    kappa_str = f"{human_kappa:.4f}" if human_kappa is not None else "N/A"
    print(f"  {'humans only':<30}  {kappa_str:>13}")

    # Humans + each AI: subjects are pairs annotated by the AI and at least one human
    for ai in ais:
        ratings = []
        for query_page, ai_query_page_scores in ai_scores.items():
            if ai.id not in ai_query_page_scores:
                continue
            human_query_page_scores = list(human_scores[query_page].values())
            if not human_query_page_scores:
                continue
            ratings.append(human_query_page_scores + [ai_query_page_scores[ai.id]])

        if not ratings:
            print(f"  {ai.name:<30}  {'N/A':>13}  (no overlapping pairs with humans)")
            continue

        kappa = fleiss_kappa(ratings, categories)
        kappa_str = f"{kappa:.4f}" if kappa is not None else "N/A"
        print(f"  {ai.name:<30}  {kappa_str:>13}")

    print()
    print("  Fleiss' kappa measures agreement among raters beyond what is expected by chance.")
    print("  1.0 = perfect agreement, 0.0 = agreement no better than chance, < 0 = worse than chance.")
    print("  Each AI row shows agreement when that AI is added to the full human rater pool.")
    print()


def report_krippendorffs_alpha(query_pool: QueryPool, engine, exclude_names: list[str] = []):
    humans, ais, human_scores, ai_scores, _ = load_annotation_data(query_pool, engine, exclude_names)

    categories = list(SCORE_LABELS.keys())  # [0, 1, 2, 3]

    section("Krippendorff's Alpha  (quadratic/interval metric)")
    if exclude_names:
        print(f"  Excluded from calculations: {', '.join(exclude_names)}")
    print(f"  {'Rater group':<30}  {'Alpha':>13}")
    print("  " + "-" * 45)

    # Baseline: humans only, restricted to subjects with at least 2 human raters
    human_ratings = [
        list(scores.values())
        for scores in human_scores.values()
        if len(scores) >= 2
    ]
    alpha = krippendorffs_alpha(human_ratings, categories)
    alpha_str = f"{alpha:.4f}" if alpha is not None else "N/A"
    print(f"  {'humans only':<30}  {alpha_str:>13}")

    # Humans + each AI: subjects annotated by that AI and at least one human
    for ai in ais:
        ratings = []
        for query_page, ai_query_page_scores in ai_scores.items():
            if ai.id not in ai_query_page_scores:
                continue
            human_query_page_scores = list(human_scores[query_page].values())
            if not human_query_page_scores:
                continue
            ratings.append(human_query_page_scores + [ai_query_page_scores[ai.id]])

        if not ratings:
            print(f"  {ai.name:<30}  {'N/A':>13}  (no overlapping pairs with humans)")
            continue

        alpha = krippendorffs_alpha(ratings, categories)
        alpha_str = f"{alpha:.4f}" if alpha is not None else "N/A"
        print(f"  {ai.name:<30}  {alpha_str:>13}")

    print()
    print("  Krippendorff's alpha measures agreement corrected for chance, using a coincidence matrix")
    print("  that handles varying rater counts per subject. Quadratic metric penalizes larger disagreements.")
    print("  1.0 = perfect agreement, 0.0 = chance level, < 0 = worse than chance.")
    print("  Each AI row shows agreement when that AI is added to the full human rater pool.")
    print()


def report_cohens_kappa(query_pool: QueryPool, engine, exclude_names: list[str] = []):
    humans, ais, human_scores, ai_scores, human_query_pages_by_annotator = load_annotation_data(query_pool, engine, exclude_names)

    section("Cohen's Kappa  (quadratic-weighted, pairwise)")
    if exclude_names:
        print(f"  Excluded from calculations: {', '.join(exclude_names)}")
    print(f"  {'Rater group':<30}  {'Cohen\'s Kappa':>13}")
    print("  " + "-" * 45)

    # Baseline: average pairwise kappa across all human-human pairs
    human_kappas: list[float] = []
    undefined_kappas = 0
    for a, b in combinations(humans, 2):
        shared = sorted(human_query_pages_by_annotator[a.id] & human_query_pages_by_annotator[b.id])
        if not shared:
            continue
        scores_a = [human_scores[query_page][a.id] for query_page in shared]
        scores_b = [human_scores[query_page][b.id] for query_page in shared]
        score_pairs = list(zip(scores_a, scores_b))

        kappa = cohens_kappa(score_pairs)
        # kappa = weighted_kappa(
        #     [human_scores[query_page][a.id] for query_page in shared],
        #     [human_scores[query_page][b.id] for query_page in shared],
        # )
        if kappa is None:
            undefined_kappas += 1
            continue
        human_kappas.append(kappa)

    if human_kappas:
        kappa_str = f"{sum(human_kappas) / len(human_kappas):.4f}"
    else:
        kappa_str = "N/A"
    print(f"  {'humans only':<30}  {kappa_str:>13}")

    # Each AI: average pairwise kappa between that AI and each human
    for ai in ais:
        ai_query_pages = {query_page for query_page in ai_scores if ai.id in ai_scores[query_page]}
        kappas: list[float] = []
        for human in humans:
            shared = sorted(human_query_pages_by_annotator[human.id] & ai_query_pages)
            if not shared:
                continue
            scores_a = [human_scores[query_page][human.id] for query_page in shared]
            scores_b = [ai_scores[query_page][ai.id] for query_page in shared]
            score_pairs = list(zip(scores_a, scores_b))
            kappa = cohens_kappa(score_pairs)
            # kappa = weighted_kappa(
            #     [human_scores[query_page][human.id] for query_page in shared],
            #     [ai_scores[query_page][ai.id] for query_page in shared],
            # )
            if kappa is None:
                undefined_kappas += 1
                continue
            kappas.append(kappa)

        if not kappas:
            kappa_str = "N/A"
        else:
            kappa_str = f"{sum(kappas) / len(kappas):.4f}"
        print(f"  {ai.name:<30}  {kappa_str:>13}")

    print()
    print("  Cohen's kappa measures pairwise agreement between two raters beyond chance.")
    print("  Values shown are the mean kappa across all relevant annotator pairs.")
    print("  1.0 = perfect agreement, 0.0 = chance level, < 0 = worse than chance.")
    if undefined_kappas:
        print(f"  ({undefined_kappas} pair(s) where all shared pages got the same score were skipped as undefined)")
    print()


def report_kendalls_tau(query_pool: QueryPool, engine, exclude_names: list[str] = []):
    humans, ais, human_scores, ai_scores, human_query_pages_by_annotator = load_annotation_data(query_pool, engine, exclude_names)

    section("Kendall's Tau-b  (pairwise)")
    if exclude_names:
        print(f"  Excluded from calculations: {', '.join(exclude_names)}")
    print(f"  {'Rater group':<30}  {'Tau-b':>13}")
    print("  " + "-" * 45)

    # Baseline: average pairwise tau-b across all human-human pairs
    human_taus: list[float] = []
    undefined_taus = 0
    for a, b in combinations(humans, 2):
        shared = sorted(human_query_pages_by_annotator[a.id] & human_query_pages_by_annotator[b.id])
        if not shared:
            continue

        scores_a = [human_scores[query_page][a.id] for query_page in shared]
        scores_b = [human_scores[query_page][b.id] for query_page in shared]
        score_pairs = list(zip(scores_a, scores_b))
        
        tau = kendalls_tau(score_pairs)
        # tau = kendalls_tau(
        #     [human_scores[query_page][a.id] for query_page in shared],
        #     [human_scores[query_page][b.id] for query_page in shared],
        # )
        if tau is None:
            undefined_taus += 1
            continue
        human_taus.append(tau)

    tau_str = f"{sum(human_taus) / len(human_taus):.4f}" if human_taus else "N/A"
    print(f"  {'humans only':<30}  {tau_str:>13}")

    # Each AI: average pairwise tau-b between that AI and each human
    for ai in ais:
        ai_query_pages = {query_page for query_page in ai_scores if ai.id in ai_scores[query_page]}
        taus: list[float] = []
        for human in humans:
            shared = sorted(human_query_pages_by_annotator[human.id] & ai_query_pages)
            if not shared:
                continue
            scores_a = [human_scores[query_page][human.id] for query_page in shared]
            scores_b = [ai_scores[query_page][ai.id] for query_page in shared]
            score_pairs = list(zip(scores_a, scores_b))

            tau = kendalls_tau(score_pairs)
            # tau = kendalls_tau(
            #     [human_scores[query_page][human.id] for query_page in shared],
            #     [ai_scores[query_page][ai.id] for query_page in shared],
            # )
            if tau is None:
                undefined_taus += 1
                continue
            taus.append(tau)

        tau_str = f"{sum(taus) / len(taus):.4f}" if taus else "N/A"
        print(f"  {ai.name:<30}  {tau_str:>13}")

    print()
    print("  Kendall's tau-b measures rank correlation between two raters' scores, corrected for ties.")
    print("  Values shown are the mean tau-b across all relevant annotator pairs.")
    print("  1.0 = perfect agreement, -1.0 = perfect disagreement, 0.0 = no correlation.")
    if undefined_taus:
        print(f"  ({undefined_taus} pair(s) where all shared pages got the same score were skipped as undefined)")
    print()


def report_agree(query_pool: QueryPool, engine, exclude_names: list[str] = [], metric: str = "exact"):
    humans, ais, human_scores, ai_scores, human_query_pages_by_annotator = load_annotation_data(query_pool, engine, exclude_names)

    if metric == "rmse":
        def pairwise_score(scores_a: list[int], scores_b: list[int]) -> float:
            score_pairs = list(zip(scores_a, scores_b))
            return -rmse(score_pairs)
            # return -sqrt(sum((a - b) ** 2 for a, b in zip(scores_a, scores_b)) / len(scores_a))
        metric_label = "negative RMSE"
        col_header = "-RMSE"
        footnote = "  Observed agreement (-RMSE): negative root mean squared score difference between two raters on shared pages."
        footnote2 = "  Higher is better (0.0 = perfect agreement). Values shown are the mean across all relevant annotator pairs."
    else:
        def pairwise_score(scores_a: list[int], scores_b: list[int]) -> float:
            score_pairs = list(zip(scores_a, scores_b))
            return agreement(score_pairs)
            # return sum(a == b for a, b in zip(scores_a, scores_b)) / len(scores_a)
        metric_label = "exact match"
        col_header = "Agreement"
        footnote = "  Observed agreement: fraction of shared pages where two raters gave the exact same score."
        footnote2 = "  Unlike kappa, this is not corrected for chance agreement. Values shown are the mean across all relevant annotator pairs."

    section(f"Observed Agreement  ({metric_label}, pairwise)")
    if exclude_names:
        print(f"  Excluded from calculations: {', '.join(exclude_names)}")
    print(f"  {'Rater group':<30}  {col_header:>10}")
    print("  " + "-" * 43)

    # Baseline: mean pairwise score across all human-human pairs
    human_agreements: list[float] = []
    for a, b in combinations(humans, 2):
        shared = sorted(human_query_pages_by_annotator[a.id] & human_query_pages_by_annotator[b.id])
        if not shared:
            continue
        human_agreements.append(pairwise_score(
            [human_scores[query_page][a.id] for query_page in shared],
            [human_scores[query_page][b.id] for query_page in shared],
        ))

    agree_str = f"{sum(human_agreements) / len(human_agreements):.4f}" if human_agreements else "N/A"
    print(f"  {'humans only':<30}  {agree_str:>10}")

    # Each AI: mean pairwise score between that AI and each human
    for ai in ais:
        ai_query_pages = {query_page for query_page in ai_scores if ai.id in ai_scores[query_page]}
        agreements: list[float] = []
        for human in humans:
            shared = sorted(human_query_pages_by_annotator[human.id] & ai_query_pages)
            if not shared:
                continue
            agreements.append(pairwise_score(
                [human_scores[query_page][human.id] for query_page in shared],
                [ai_scores[query_page][ai.id] for query_page in shared],
            ))

        agree_str = f"{sum(agreements) / len(agreements):.4f}" if agreements else "N/A"
        print(f"  {ai.name:<30}  {agree_str:>10}")

    print()
    print(footnote)
    print(footnote2)
    print()


def report_mean_bias(query_pool: QueryPool, engine, exclude_names: list[str] = []):
    humans, ais, human_scores, ai_scores, human_query_pages_by_annotator = load_annotation_data(query_pool, engine, exclude_names)

    section("Mean Bias  (mean signed score difference, pairwise)")
    if exclude_names:
        print(f"  Excluded from calculations: {', '.join(exclude_names)}")
    print(f"  {'Rater group':<30}  {'Bias':>10}")
    print("  " + "-" * 43)

    # Baseline: mean of pairwise mean(a - b) across all human-human pairs
    human_biases: list[float] = []
    for a, b in combinations(humans, 2):
        shared = sorted(human_query_pages_by_annotator[a.id] & human_query_pages_by_annotator[b.id])
        if not shared:
            continue

        scores_a = [human_scores[qp][a.id] for qp in shared]
        scores_b = [human_scores[qp][b.id] for qp in shared]
        score_pairs = list(zip(scores_a, scores_b))

        human_biases.append(mean_bias(score_pairs))
        # human_biases.append(
        #     sum(human_scores[qp][a.id] - human_scores[qp][b.id] for qp in shared) / len(shared)
        # )

    bias_str = f"{sum(human_biases) / len(human_biases):.4f}" if human_biases else "N/A"
    print(f"  {'humans only':<30}  {bias_str:>10}")

    # Each AI: mean of (AI score - human score) across all humans
    for ai in ais:
        ai_query_pages = {qp for qp in ai_scores if ai.id in ai_scores[qp]}
        biases: list[float] = []
        for human in humans:
            shared = sorted(human_query_pages_by_annotator[human.id] & ai_query_pages)
            if not shared:
                continue
            scores_a = [ai_scores[qp][ai.id] for qp in shared]
            scores_b = [human_scores[qp][human.id] for qp in shared]
            score_pairs = list(zip(scores_a, scores_b))

            biases.append(mean_bias(score_pairs))
            # biases.append(
            #     sum(ai_scores[qp][ai.id] - human_scores[qp][human.id] for qp in shared) / len(shared)
            # )

        bias_str = f"{sum(biases) / len(biases):.4f}" if biases else "N/A"
        print(f"  {ai.name:<30}  {bias_str:>10}")

    print()
    print("  Mean bias: mean signed score difference between two raters on shared pages.")
    print("  For AI rows: positive = AI scores higher than the human on average.")
    print("  For humans only: expected near 0 across all pairs; non-zero indicates ordering effects.")
    print()


def report_mean_absolute_error(query_pool: QueryPool, engine, exclude_names: list[str] = []):
    humans, ais, human_scores, ai_scores, human_query_pages_by_annotator = load_annotation_data(query_pool, engine, exclude_names)

    section("Mean Absolute Error  (pairwise)")
    if exclude_names:
        print(f"  Excluded from calculations: {', '.join(exclude_names)}")
    print(f"  {'Rater group':<30}  {'MAE':>10}")
    print("  " + "-" * 43)

    # Baseline: mean pairwise MAE across all human-human pairs
    human_maes: list[float] = []
    for a, b in combinations(humans, 2):
        shared = sorted(human_query_pages_by_annotator[a.id] & human_query_pages_by_annotator[b.id])
        if not shared:
            continue
        scores_a = [human_scores[qp][a.id] for qp in shared]
        scores_b = [human_scores[qp][b.id] for qp in shared]
        score_pairs = list(zip(scores_a, scores_b))
        human_maes.append(mae(score_pairs))
        # human_maes.append(
        #     sum(abs(human_scores[qp][a.id] - human_scores[qp][b.id]) for qp in shared) / len(shared)
        # )

    mae_str = f"{sum(human_maes) / len(human_maes):.4f}" if human_maes else "N/A"
    print(f"  {'humans only':<30}  {mae_str:>10}")

    # Each AI: mean |AI score - human score| across all humans
    for ai in ais:
        ai_query_pages = {qp for qp in ai_scores if ai.id in ai_scores[qp]}
        maes: list[float] = []
        for human in humans:
            shared = sorted(human_query_pages_by_annotator[human.id] & ai_query_pages)
            if not shared:
                continue
            scores_a = [human_scores[qp][human.id] for qp in shared]
            scores_b = [ai_scores[qp][ai.id] for qp in shared]
            score_pairs = list(zip(scores_a, scores_b))
            maes.append(mae(score_pairs))
            # maes.append(
            #     sum(abs(ai_scores[qp][ai.id] - human_scores[qp][human.id]) for qp in shared) / len(shared)
            # )

        mae_str = f"{sum(maes) / len(maes):.4f}" if maes else "N/A"
        print(f"  {ai.name:<30}  {mae_str:>10}")

    print()
    print("  MAE: mean absolute score difference between two raters on shared pages.")
    print("  Lower is better (0.0 = perfect agreement). Values shown are the mean across all relevant annotator pairs.")
    print()


def report_aap(query_pool: QueryPool, engine, cost_benefit_penalty: float = 0.2, q: float = 0.05, exclude_names: list[str] = []):
    humans, ais, human_scores, ai_scores, human_query_pages_by_annotator = load_annotation_data(query_pool, engine, exclude_names)

    all_alignments: dict[int, list] = {}

    for excluded_human in humans:

        all_alignments[excluded_human.id] = []

        for query_page in human_query_pages_by_annotator[excluded_human.id]:
            other_human_scores = {
                ann_id: score
                for ann_id, score in human_scores[query_page].items()
                if ann_id != excluded_human.id
            }

            if not other_human_scores:
                print(f"  Warning: no co-annotators for query_page {query_page} when excluding {excluded_human.name}, skipping page")
                continue

            excluded_human_score = human_scores[query_page][excluded_human.id]

            excluded_human_errors: list[float] = []
            ai_errors: dict[int, list[float]] = {ai.id: [] for ai in ais}

            for score in other_human_scores.values():

                excluded_human_errors.append(score - excluded_human_score)

                for ai in ais:
                    ai_score = ai_scores[query_page][ai.id]
                    ai_errors[ai.id].append(score - ai_score)
            
            # using negative RMSE so that higher = better
            excluded_human_alignment = (-1) * sqrt(sum([e**2 for e in excluded_human_errors]) / len(excluded_human_errors)) - cost_benefit_penalty
            
            ai_alignments = {}
            
            for ai in ais:
                ai_alignment = (-1) * sqrt(sum([e**2 for e in ai_errors[ai.id]]) / len(ai_errors[ai.id]))
                ai_alignments[ai.id] = ai_alignment

            all_alignments[excluded_human.id].append({"alignment": excluded_human_alignment, "ai_alignments": ai_alignments})

    all_human_scores = {}
    all_ai_scores = {}
    for human in humans:

        all_human_scores[human.id] = []
        all_ai_scores[human.id] = {ai.id: [] for ai in ais}

        for human_align_scores in all_alignments[human.id]:
            human_alignment = human_align_scores["alignment"]
            ai_align_scores = human_align_scores["ai_alignments"]
            all_human_scores[human.id].append(human_alignment)
            for ai in ais:
                ai_alignment = ai_align_scores[ai.id]
                all_ai_scores[human.id][ai.id].append(ai_alignment)

    # calculate t-stat and p-value
    all_stats = {}
    for human in humans:
        all_stats[human.id] = {}
        for ai in ais:
            t_stat, p_value = stats.ttest_rel(
                np.array(all_ai_scores[human.id][ai.id]), 
                np.array(all_human_scores[human.id]), 
                alternative='greater'
            )
            all_stats[human.id][ai.id] = {"t_stat": t_stat, "p_value": p_value}

    # pprint.pprint(all_stats)

    harmonic_constant = sum([(1/i) for i in range(1,len(humans)+1)])

    win_rates = {}
    for ai in ais:
        # print(f"FOR AI: {ai.name}")
        largest_k = 0
        # human ids sorted by p_value
        sorted_humans_by_p_value = sorted(humans, key=lambda human: all_stats[human.id][ai.id]["p_value"])
        for k, human in enumerate(sorted_humans_by_p_value):
            p_value = all_stats[human.id][ai.id]["p_value"]
            by_compare = ((k+1)*q) / (len(humans) * harmonic_constant)
            # print(f"HUMAN: {human.name}")
            # print(f"p_value {p_value}")
            # print(f"by_compare {by_compare}")
            # print()
            if p_value <= by_compare:
                largest_k = k+1

        win_rate = largest_k / len(humans)
        win_rates[ai.id] = win_rate
    
    # calculating aaps (rho)
    all_advantages: dict[int, dict] = {}
    for human in humans:
        all_advantages[human.id] = {ai.id: [] for ai in ais}

        for human_align_scores in all_alignments[human.id]:
            human_alignment = human_align_scores["alignment"]
            ai_align_scores = human_align_scores["ai_alignments"]

            for ai in ais:
                ai_alignment = ai_align_scores[ai.id]

                ai_has_advantage = False
                if ai_alignment >= human_alignment:
                    ai_has_advantage = True
                all_advantages[human.id][ai.id].append(ai_has_advantage)



    all_aps: dict[int, dict] = {}
    for human in humans:
        if all_alignments[human.id]:

            all_aps[human.id] = {ai.id: [] for ai in ais}

            for ai in ais:
                advantages = all_advantages[human.id][ai.id]
                all_aps[human.id][ai.id] = sum([1 for advantage in advantages if advantage is True]) / len(advantages)

    # pprint.pprint(all_aps)

    aaps: dict[int, float] = {}
    for ai in ais:
        aaps[ai.id] = 0
        for human in humans:
            aaps[ai.id] += all_aps[human.id][ai.id]
        aaps[ai.id] = aaps[ai.id] / len(humans)

    section("AAP  (Average Advantage Probability)")
    if exclude_names:
        print(f"  Excluded from calculations: {', '.join(exclude_names)}")
    print(f"  {'AI':<30}  {'Win Rate':>10}  {'AAP (ρ)':>10}")
    print("  " + "-" * 55)
    for ai in ais:
        print(f"  {ai.name:<30}  {win_rates[ai.id]:>10.4f}  {aaps[ai.id]:>10.4f}")
    print()
    print("  Win Rate: fraction of human annotators the AI outperformed (Benjamini-Yekutieli corrected).")
    print("  AAP (ρ): average probability the AI's alignment exceeded the held-out human's alignment.")
    print()

def display_excluded(excluded: list[Annotator]):
    print("Excluded: ", end="")
    for i, rater in enumerate(excluded):
        print(rater, end="")
        if i < len(excluded) - 1:
            print(", ", end="")
    print("")


def display_metrics(metrics: list[tuple[str, dict[str, float]]]):
    raters = metrics[0][1].keys()
    metric_names = [metrics[i][0] for i in range(len(metrics))]

    # header
    print(("=" * 160))
    print(f" {'Rater':<30}", end="")
    for metric_name in metric_names:
        print(f" {metric_name:<18}", end="")
    print("\n" + ("=" * 160))

    for rater in raters:
        print(f" {rater:<30}", end="")
        for metric in metrics:
            try: 
                print(f" {metric[1][rater]:<18.4f}", end="")
            except KeyError:
                print(f" {'---':<18}", end="")

        print("")

    print("")



def main():
    eval_dir = Path(os.getenv("DATA_DIR", "/app/data")) / "evaluation"

    args = arguments(eval_dir)

    role = None if args.annotators == "all" else str(args.annotators)

    roles = build_roles_list(args.annotators)

    query_pool = QueryPool.import_from_json(args.pool)

    if args.k > query_pool.top_k:
        print("Error: chosen k must be less than or equal to the chosen query pool's top k.")
        return -1

    conn_url = URL.create(
        drivername=os.getenv("DB_DRIVER", "postgresql"),
        username=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
    port=int(os.getenv("DB_PORT", "5432")),
        database=os.getenv("DB_DATABASE"),
    )
    engine = create_engine(conn_url)

    # load systems
    systems = load_systems(args.systems_dir)

    excluded = args.exclude
    # report_agree(query_pool, engine, excluded, args.agree_metric)
    # report_mean_bias(query_pool, engine, excluded)
    # report_mean_absolute_error(query_pool, engine, excluded)
    report_fleiss_kappa(query_pool, engine, excluded)
    report_krippendorffs_alpha(query_pool, engine, excluded)
    # report_cohens_kappa(query_pool, engine, excluded)
    # report_kendalls_tau(query_pool, engine, excluded)
    # report_aap(query_pool, engine, cost_benefit_penalty=0.00, exclude_names=excluded)

    # annotator_report(query_pool, role, engine, csv_path=args.csv)
    # report_ndcgs(systems, query_pool, roles, args.aggregate, args.k, engine, args.ndcg_mode, excluded)

    print()
    hr()

    # configurations
    cost_benefit_penalty = 0.20
    q = 0.05

    annotators, query_pages, annotations = load_annotation_data_2(query_pool, engine, excluded)

    all_metric_scores = []
    
    all_metric_scores.append(("Agreement", compute_pairwise_metric(annotators, query_pages, annotations, agreement)))
    all_metric_scores.append(("Mean Bias", compute_pairwise_metric(annotators, query_pages, annotations, mean_bias)))
    all_metric_scores.append(("MAE", compute_pairwise_metric(annotators, query_pages, annotations, mae)))
    all_metric_scores.append(("Cohen's Kappa", compute_pairwise_metric(annotators, query_pages, annotations, cohens_kappa)))
    all_metric_scores.append(("Kendall's Tau-b", compute_pairwise_metric(annotators, query_pages, annotations, kendalls_tau)))

    all_win_rates, all_aaps = aaps(annotators, query_pages, annotations, cost_benefit_penalty=cost_benefit_penalty, q=q)
    all_metric_scores.append(("Win Rate", all_win_rates))
    all_metric_scores.append(("AAP", all_aaps))

    print("Measures for Inter-rater Agreement")
    display_excluded(excluded)
    display_metrics(all_metric_scores)



if __name__ == "__main__":
    main()
