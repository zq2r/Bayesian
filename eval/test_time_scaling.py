"""Small, model-agnostic test-time scaling evaluator for BayesianPRM.

The model and dataset are intentionally left as injection points.  This file
evaluates a pool of already generated candidates, which keeps generation and
PRM scoring separate and makes the 20-repeat protocol easy to reuse.

Expected record format::

    {
        "solutions_splits": [["step 1", "step 2"], ...],
        "prm_scores": [[0.8, 0.7], ...],
        "labels": [1, 0, ...]
    }

For Bayesian head-level output, ``prm_scores`` can be replaced with
``prm_mu_heads`` and ``prm_rel_weights``.  Labels are optional when the caller
only needs selected candidate indices; accuracy is then omitted from the
summary.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any, Iterable, Sequence


# Keep the original evaluation protocol explicit: every N is averaged over 20
# independently shuffled candidate subsets unless the caller overrides it.
DEFAULT_REPEATS = 20
DEFAULT_N_GRID = (1, 2, 4, 8, 16)


def mean_std(values: Iterable[float]) -> tuple[float, float]:
    values = [float(value) for value in values]
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], 0.0
    return float(statistics.mean(values)), float(statistics.pstdev(values))


def build_permutations(num_items: int, pool_size: int, seed: int, repeat: int) -> list[list[int]]:
    """Build deterministic, independent candidate orders for one repeat."""
    rng = random.Random(seed + 1_000_003 * repeat)
    permutations = []
    for _ in range(num_items):
        order = list(range(pool_size))
        rng.shuffle(order)
        permutations.append(order)
    return permutations


def bayesian_step_score(mu_heads: Sequence[float], rel_weights: Sequence[float], beta2: float) -> float:
    """Apply BayesianPRM's conservative head weighting to one process step."""
    if len(mu_heads) != len(rel_weights) or not mu_heads:
        raise ValueError("mu_heads and rel_weights must be non-empty and aligned")
    if beta2 <= 0:
        raise ValueError("beta2 must be positive")

    mus = [float(value) for value in mu_heads]
    rels = [max(float(value), 1e-6) for value in rel_weights]
    # Keep the expression in log-space for numerical stability.
    log_weights = [math.log(weight) - mu / beta2 for mu, weight in zip(mus, rels)]
    peak = max(log_weights)
    weights = [math.exp(value - peak) for value in log_weights]
    normalizer = sum(weights)
    return sum(weight * mu for weight, mu in zip(weights, mus)) / normalizer


def trajectory_score(candidate: dict[str, Any], beta2: float = 0.1) -> float:
    """Return one scalar score for a candidate solution."""
    if "prm_scores" in candidate:
        step_scores = candidate["prm_scores"]
    else:
        mu_heads = candidate.get("prm_mu_heads")
        rel_weights = candidate.get("prm_rel_weights")
        if mu_heads is None or rel_weights is None:
            raise ValueError("candidate needs prm_scores or Bayesian head-level scores")
        if len(mu_heads) != len(rel_weights):
            raise ValueError("prm_mu_heads and prm_rel_weights must have equal step counts")
        step_scores = [bayesian_step_score(mu, rel, beta2) for mu, rel in zip(mu_heads, rel_weights)]

    if isinstance(step_scores, (int, float)):
        step_scores = [step_scores]
    if not step_scores:
        raise ValueError("candidate has no process rewards")
    return float(statistics.mean(float(value) for value in step_scores))


def _candidate_pool(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize candidate-level arrays while retaining the source record."""
    solutions = record.get("solutions_splits")
    if not isinstance(solutions, list) or not solutions:
        raise ValueError("each record needs a non-empty solutions_splits list")

    def values(name: str) -> list[Any] | None:
        value = record.get(name)
        return value if isinstance(value, list) else None

    scores = values("prm_scores")
    mu_heads = values("prm_mu_heads")
    rel_weights = values("prm_rel_weights")
    if scores is None and (mu_heads is None or rel_weights is None):
        raise ValueError("each record needs prm_scores or Bayesian head-level scores")

    count = len(solutions)
    if scores is not None and len(scores) != count:
        raise ValueError("prm_scores must align with solutions_splits")
    if mu_heads is not None and len(mu_heads) != count:
        raise ValueError("prm_mu_heads must align with solutions_splits")
    if rel_weights is not None and len(rel_weights) != count:
        raise ValueError("prm_rel_weights must align with solutions_splits")

    return [
        {
            "solution": solutions[index],
            **({"prm_scores": scores[index]} if scores is not None else {}),
            **({"prm_mu_heads": mu_heads[index]} if mu_heads is not None else {}),
            **({"prm_rel_weights": rel_weights[index]} if rel_weights is not None else {}),
        }
        for index in range(count)
    ]


def select_best_candidate(candidates: Sequence[dict[str, Any]], indices: Sequence[int], beta2: float) -> int:
    """Select the highest-scoring candidate, preserving first-on-tie behavior."""
    return max(indices, key=lambda index: trajectory_score(candidates[index], beta2))


def evaluate_test_time_scaling(
    records: Sequence[dict[str, Any]],
    n_grid: Sequence[int] = DEFAULT_N_GRID,
    repeats: int = DEFAULT_REPEATS,
    seed: int = 42,
    beta2: float = 0.1,
) -> dict[str, Any]:
    """Evaluate random-subset best-of-N scaling and return JSON-ready results."""
    if not records:
        raise ValueError("records must not be empty")
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    if beta2 <= 0:
        raise ValueError("beta2 must be positive")
    ns = list(dict.fromkeys(int(n) for n in n_grid))
    if not ns or ns[0] < 1:
        raise ValueError("n_grid must contain positive integers")

    pools = [_candidate_pool(record) for record in records]
    pool_size = min(len(pool) for pool in pools)
    if max(ns) > pool_size:
        raise ValueError(f"n_grid requests {max(ns)} candidates, but pool has {pool_size}")

    output: dict[str, Any] = {
        "method": "bayesian_prm_test_time_scaling",
        "num_items": len(records),
        "n_grid": ns,
        "pool_max_n": pool_size,
        "repeats": repeats,
        "seed": seed,
        "beta2": beta2,
        "results": {},
    }

    for n in ns:
        repeat_results = []
        for repeat in range(repeats):
            permutations = build_permutations(len(pools), pool_size, seed, repeat)
            selected_indices = []
            correct = 0
            oracle_correct = 0
            labeled = True
            for record, candidates, order in zip(records, pools, permutations):
                ids = order[:n]
                selected = select_best_candidate(candidates, ids, beta2)
                selected_indices.append(selected)
                labels = record.get("labels")
                if not isinstance(labels, list) or len(labels) < pool_size:
                    labeled = False
                else:
                    oracle_correct += int(any(int(labels[index]) == 1 for index in ids))
                    correct += int(int(labels[selected]) == 1)

            item: dict[str, Any] = {
                "repeat": repeat,
                "selected_indices": selected_indices,
            }
            if labeled:
                item.update({
                    "accuracy": correct / len(records),
                    "correct": correct,
                    "oracle_accuracy": oracle_correct / len(records),
                    "oracle_correct": oracle_correct,
                    "total": len(records),
                })
            repeat_results.append(item)

        result: dict[str, Any] = {"repeat_results": repeat_results}
        accuracies = [item["accuracy"] for item in repeat_results if "accuracy" in item]
        if accuracies:
            result["accuracy_mean"], result["accuracy_std"] = mean_std(accuracies)
            oracle = [item["oracle_accuracy"] for item in repeat_results]
            result["oracle_accuracy_mean"], result["oracle_accuracy_std"] = mean_std(oracle)
        output["results"][str(n)] = result
    return output


# Generation and loading stay intentionally empty until a concrete dataset and
# model are selected.  A caller can inject these functions around the evaluator.
DATASET = None
MODEL = None


def generate_candidates(model: Any, sample: Any, num_candidates: int) -> list[Any]:
    raise NotImplementedError("plug the task-specific dataset/model generation here")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--n-grid", default=','.join(map(str, DEFAULT_N_GRID)))
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--beta2", type=float, default=0.1)
    args = parser.parse_args()

    with args.input_json.open(encoding="utf-8") as stream:
        records = json.load(stream)
    if not isinstance(records, list):
        raise ValueError("input JSON must contain a list of records")

    result = evaluate_test_time_scaling(
        records,
        n_grid=[int(value.strip()) for value in args.n_grid.split(",") if value.strip()],
        repeats=args.repeats,
        seed=args.seed,
        beta2=args.beta2,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)

    for n in result["n_grid"]:
        summary = result["results"][str(n)]
        if "accuracy_mean" in summary:
            print(f"N={n:<2d} accuracy={summary['accuracy_mean']:.4f} +/- {summary['accuracy_std']:.4f}")
        else:
            print(f"N={n:<2d} selected candidates saved")


if __name__ == "__main__":
    main()
