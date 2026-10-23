import pprint
import math
from collections import Counter, defaultdict
import scipy.stats as stats
import numpy as np
from itertools import combinations

from .core import QueryPool
from .data import human_ai_split

def mse(score_pairs: list[tuple[int, int]]) -> float:
    return sum((a - b) ** 2 for a, b in score_pairs) / len(score_pairs)

def rmse(score_pairs: list[tuple[int, int]]) -> float:
    return math.sqrt(mse(score_pairs))

def mae(score_pairs: list[tuple[int, int]]) -> float:
    return sum(abs(a - b) for a, b in score_pairs) / len(score_pairs)

def agreement(score_pairs: list[tuple[int, int]]) -> float:
    return sum(a == b for a, b in score_pairs) / len(score_pairs)

def mean_bias(score_pairs: list[tuple[int, int]]) -> float:
    return sum(a - b for a, b in score_pairs) / len(score_pairs)

def cohens_kappa(score_pairs: list[tuple[int, int]]) -> float | None:
    a_scores, b_scores = zip(*score_pairs)
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

def kendalls_tau(score_pairs: list[tuple[int, int]]) -> float | None:
    a_scores, b_scores = zip(*score_pairs)
    tau, _ = stats.kendalltau(a_scores, b_scores, variant="b")
    return None if np.isnan(tau) else float(tau)

# computes avg pairwise metric for each annotator
# for all humans: all possible combinations of human, human pair
# for each ai A: every human, A pair
def compute_pairwise_metric(annotators, query_pages, annotations, pairwise_metric_func):

    avg_metrics: dict[str, float] = dict()

    humans, ais = human_ai_split(annotators)

    human_combos = combinations(humans, r=2)

    human_metrics = []

    for human_pair in human_combos:
        human_a = human_pair[0]
        human_b = human_pair[1]

        a_scores = [annotations[(query_id, page_id, human_a.id)] for query_id, page_id in query_pages]
        b_scores = [annotations[(query_id, page_id, human_b.id)] for query_id, page_id in query_pages]
        score_pairs = list(zip(a_scores, b_scores))

        human_metrics.append(pairwise_metric_func(score_pairs))

    avg_metrics["humans"] = sum(human_metrics) / len(human_metrics)

    for ai in ais:

        ai_metrics = []

        ai_scores = [annotations[(query_id, page_id, ai.id)] for query_id, page_id in query_pages]

        for human in humans:
            human_scores = [annotations[(query_id, page_id, human.id)] for query_id, page_id in query_pages]
            score_pairs = list(zip(ai_scores, human_scores))

            ai_metrics.append(pairwise_metric_func(score_pairs))

        avg_metrics[ai.name] = sum(ai_metrics) / len(ai_metrics)

    return avg_metrics

# aap based metrics

def aaps(annotators, query_pages, annotations, cost_benefit_penalty: float = 0.10, q: float = 0.05):
    humans, ais = human_ai_split(annotators)

    all_alignments: dict[tuple[int, int], list] = defaultdict(list)

    for excluded_human in humans:
        included_humans = [human for human in humans if human is not excluded_human]

        for query_id, page_id in query_pages:
            
            # get scores of current (query, page) pair for all annotators
            scores: dict[int, float] = dict()
            
            for annotator in annotators:
                if (query_id, page_id, annotator.id) not in annotations.keys():
                    print("error: annotation not found for annotator")
                    exit()
                else:
                    scores[annotator.id] = annotations[(query_id, page_id, annotator.id)]

            excluded_included_score_pairs = list(zip(
                [scores[excluded_human.id] for _ in included_humans],
                [scores[human.id] for human in included_humans]
            ))

            excluded_human_alignment = -rmse(excluded_included_score_pairs) - cost_benefit_penalty
            all_alignments[(excluded_human.id, excluded_human.id)].append(excluded_human_alignment)

            for ai in ais:
                ai_included_score_pairs = list(zip(
                    [scores[ai.id] for _ in included_humans],
                    [scores[human.id] for human in included_humans]
                ))

                ai_alignment = -rmse(ai_included_score_pairs)
                all_alignments[(excluded_human.id, ai.id)].append(ai_alignment)


    # calculate t-stat and p-value
    all_stats = dict()
    for human in humans:
        for ai in ais:
            t_stat, p_value = stats.ttest_rel(
                np.array(all_alignments[(human.id, ai.id)]), 
                np.array(all_alignments[(human.id, human.id)]), 
                alternative='greater'
            )
            all_stats[(human.id, ai.id)] = {"t_stat": t_stat, "p_value": p_value}

    harmonic_constant = sum([(1/i) for i in range(1,len(humans)+1)])

    win_rates = dict()
    for ai in ais:
        largest_k = 0
        # human ids sorted by p_value
        sorted_humans_by_p_value = sorted(humans, key=lambda human: all_stats[(human.id,ai.id)]["p_value"])
        for k, human in enumerate(sorted_humans_by_p_value):
            p_value = all_stats[(human.id, ai.id)]["p_value"]
            by_compare = ((k+1)*q) / (len(humans) * harmonic_constant)

            if p_value <= by_compare:
                largest_k = k+1

        win_rate = largest_k / len(humans)
        win_rates[ai.name] = win_rate
    
    # calculating aaps (rho)
    all_advantages: dict[tuple[int, int], list] = defaultdict(list)

    for human in humans:
        for ai in ais:
            for ai_alignment, human_alignment in zip(all_alignments[(human.id, ai.id)], all_alignments[(human.id, human.id)]):
                ai_has_advantage = False
                if ai_alignment >= human_alignment:
                    ai_has_advantage = True
                all_advantages[(human.id, ai.id)].append(ai_has_advantage)



    all_aps: dict[tuple[int, int], float] = {}
    for human in humans:
        for ai in ais:
            advantages = all_advantages[(human.id, ai.id)]
            all_aps[(human.id, ai.id)] = sum([1 for advantage in advantages if advantage is True]) / len(advantages)
    
    aaps: dict[int, float] = {}
    for ai in ais:
        aaps[ai.name] = sum(all_aps[(human.id, ai.id)] for human in humans) / len(humans)

    return win_rates, aaps
