"""Compute BayesianPRM calibration errors against Monte-Carlo rollout labels.

The model and dataset are deliberately not loaded here.  The inference step
produces prediction JSON, while a separate rollout/judge step produces one
Monte-Carlo target per prefix.  This module only aligns the two files and
computes the calibration metrics.

Prediction records can be either flat::

    {"prefix_id": "p0", "prm_scores": [0.4, 0.6],
     "prm_mu_rel": [0.5, 0.7]}

or grouped by question::

    {"prefix_ids": ["p0"], "prm_scores": [[0.4, 0.6]],
     "prm_mu_rel": [[0.5, 0.7]]}

MC-label records need ``prefix_id`` and either ``success_prob`` /
``target_success_prob`` or the counts ``mc_correct`` and ``mc_total``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence


DEFAULT_NUM_BINS = 10
MODEL = None
DATASET = None


def require_probability(value: Any, *, field: str, prefix_id: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{prefix_id}: {field} must be numeric")
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{prefix_id}: {field} must be finite and in [0, 1]")
    return value


def fixed_width_bins(
    predictions: Sequence[float],
    targets: Sequence[float],
    num_bins: int,
) -> list[dict[str, Any]]:
    bins: list[list[int]] = [[] for _ in range(num_bins)]
    for index, prediction in enumerate(predictions):
        bin_index = min(int(prediction * num_bins), num_bins - 1)
        bins[bin_index].append(index)

    output = []
    for bin_index, indices in enumerate(bins):
        lower = bin_index / num_bins
        upper = (bin_index + 1) / num_bins
        if indices:
            mean_prediction = sum(predictions[i] for i in indices) / len(indices)
            mean_target = sum(targets[i] for i in indices) / len(indices)
            gap = abs(mean_prediction - mean_target)
        else:
            mean_prediction = mean_target = gap = None
        output.append({
            "bin_index": bin_index,
            "lower": lower,
            "upper": upper,
            "right_closed": bin_index == num_bins - 1,
            "count": len(indices),
            "mean_prediction": mean_prediction,
            "mean_target": mean_target,
            "absolute_gap": gap,
        })
    return output


def adaptive_equal_count_bins(
    predictions: Sequence[float],
    targets: Sequence[float],
    num_bins: int,
) -> list[dict[str, Any]]:
    order = sorted(range(len(predictions)), key=lambda i: (predictions[i], i))
    base_size, remainder = divmod(len(order), num_bins)
    output = []
    cursor = 0
    for bin_index in range(num_bins):
        size = base_size + int(bin_index < remainder)
        indices = order[cursor : cursor + size]
        cursor += size
        if not indices:
            output.append({
                "bin_index": bin_index,
                "count": 0,
                "min_prediction": None,
                "max_prediction": None,
                "mean_prediction": None,
                "mean_target": None,
                "absolute_gap": None,
            })
            continue
        values = [predictions[i] for i in indices]
        target_values = [targets[i] for i in indices]
        mean_prediction = sum(values) / len(values)
        mean_target = sum(target_values) / len(target_values)
        output.append({
            "bin_index": bin_index,
            "count": len(indices),
            "min_prediction": min(values),
            "max_prediction": max(values),
            "mean_prediction": mean_prediction,
            "mean_target": mean_target,
            "absolute_gap": abs(mean_prediction - mean_target),
        })
    return output


def _weighted_ce(bins: Iterable[dict[str, Any]], total: int) -> float:
    return sum(
        (int(item["count"]) / total) * float(item["absolute_gap"])
        for item in bins
        if int(item["count"]) > 0
    )


def _average_ce(bins: Iterable[dict[str, Any]]) -> float:
    gaps = [
        float(item["absolute_gap"])
        for item in bins
        if int(item["count"]) > 0
    ]
    if not gaps:
        raise ValueError("No non-empty calibration bins")
    return sum(gaps) / len(gaps)


def monte_carlo_label(rollout_labels: Sequence[Any]) -> float:
    """Convert binary rollout outcomes into the Monte-Carlo target K/N."""
    if not rollout_labels:
        raise ValueError("rollout_labels must not be empty")
    if any(label not in (0, 1, False, True) for label in rollout_labels):
        raise ValueError("rollout_labels must be binary")
    return sum(int(label) for label in rollout_labels) / len(rollout_labels)


def compute_metrics(
    predictions: Sequence[float],
    targets: Sequence[float],
    num_bins: int = DEFAULT_NUM_BINS,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Return KWDK-style calibration metrics and bin diagnostics."""
    if len(predictions) != len(targets) or not predictions:
        raise ValueError("predictions and targets must be non-empty and aligned")
    if num_bins <= 1:
        raise ValueError("num_bins must be greater than 1")

    predictions = [require_probability(p, field="prediction", prefix_id=str(i)) for i, p in enumerate(predictions)]
    targets = [require_probability(t, field="target", prefix_id=str(i)) for i, t in enumerate(targets)]
    errors = [prediction - target for prediction, target in zip(predictions, targets)]
    fixed = fixed_width_bins(predictions, targets, num_bins)
    adaptive = adaptive_equal_count_bins(predictions, targets, num_bins)
    metrics = {
        "brier": sum(error * error for error in errors) / len(errors),
        "positive_brier": sum(max(error, 0.0) ** 2 for error in errors) / len(errors),
        "ece": _weighted_ce(fixed, len(errors)),
        "average_ce": _average_ce(fixed),
        # Kept for parity with the existing calibration reports.
        "adaptive_ce": _weighted_ce(adaptive, len(errors)),
    }
    return metrics, {
        "fixed_width_bins": fixed,
        "adaptive_equal_count_bins": adaptive,
    }


def _load_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"No records found in {path}")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ValueError(f"{path} must contain a JSON list or JSONL objects")
    return value


def _terminal(value: Any, prefix_id: str, field: str) -> float:
    if isinstance(value, list):
        if not value:
            raise ValueError(f"{prefix_id}: {field} is empty")
        value = value[-1]
    return require_probability(value, field=field, prefix_id=prefix_id)


def _prediction_field(item: dict[str, Any], variant: str) -> str:
    fields = {
        "reliability": ("prm_mu_rel",),
        "final": ("prm_mu", "prm_scores"),
    }
    for field in fields[variant]:
        if field in item:
            return field
    raise ValueError(f"prediction record has no {variant} field")


def flatten_predictions(records: Sequence[dict[str, Any]], variant: str) -> dict[str, float]:
    """Extract terminal per-prefix predictions from flat or grouped output."""
    if variant not in {"reliability", "final"}:
        raise ValueError("variant must be reliability or final")
    output: dict[str, float] = {}
    for item_index, item in enumerate(records):
        if isinstance(item.get("prefix_id"), str):
            prefix_id = item["prefix_id"]
            if prefix_id in output:
                raise ValueError(f"duplicate prediction prefix_id: {prefix_id}")
            field = _prediction_field(item, variant)
            output[prefix_id] = _terminal(item[field], prefix_id, field)
            continue

        prefix_ids = item.get("prefix_ids")
        if not isinstance(prefix_ids, list) or not prefix_ids:
            raise ValueError(f"prediction item {item_index} needs prefix_id or prefix_ids")
        field = _prediction_field(item, variant)
        values = item[field]
        if not isinstance(values, list) or len(values) != len(prefix_ids):
            raise ValueError(f"prediction item {item_index}: {field} does not align with prefix_ids")
        for prefix_id, value in zip(prefix_ids, values):
            if not isinstance(prefix_id, str) or not prefix_id:
                raise ValueError(f"prediction item {item_index}: invalid prefix_id")
            if prefix_id in output:
                raise ValueError(f"duplicate prediction prefix_id: {prefix_id}")
            output[prefix_id] = _terminal(value, prefix_id, field)
    if not output:
        raise ValueError("No predictions found")
    return output


def _mc_target(record: dict[str, Any]) -> float:
    prefix_id = str(record.get("prefix_id", ""))
    if "target_success_prob" in record:
        target = require_probability(
            record["target_success_prob"],
            field="target_success_prob",
            prefix_id=prefix_id,
        )
    elif "success_prob" in record:
        target = require_probability(record["success_prob"], field="success_prob", prefix_id=prefix_id)
    elif "mc_correct" in record and "mc_total" in record:
        total = int(record["mc_total"])
        correct = int(record["mc_correct"])
        if total <= 0 or not 0 <= correct <= total:
            raise ValueError(f"{prefix_id}: invalid MC counts")
        target = correct / total
    else:
        labels = record.get("rollout_labels", record.get("labels"))
        if not isinstance(labels, list) or not labels:
            raise ValueError(f"{prefix_id}: missing MC probability or rollout labels")
        try:
            target = monte_carlo_label(labels)
        except ValueError as exc:
            raise ValueError(f"{prefix_id}: {exc}") from exc

    if "mc_correct" in record and "mc_total" in record:
        correct, total = int(record["mc_correct"]), int(record["mc_total"])
        if total <= 0 or not 0 <= correct <= total or not math.isclose(
            target, correct / total, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(f"{prefix_id}: MC probability is inconsistent with K/N")
    return target


def load_mc_targets(records: Sequence[dict[str, Any]]) -> dict[str, float]:
    targets: dict[str, float] = {}
    for index, record in enumerate(records):
        prefix_id = record.get("prefix_id")
        if not isinstance(prefix_id, str) or not prefix_id:
            raise ValueError(f"MC record {index}: missing prefix_id")
        if prefix_id in targets:
            raise ValueError(f"duplicate MC prefix_id: {prefix_id}")
        targets[prefix_id] = _mc_target(record)
    if not targets:
        raise ValueError("No MC labels found")
    return targets


def evaluate_calibration(
    prediction_records: Sequence[dict[str, Any]],
    mc_records: Sequence[dict[str, Any]],
    *,
    variants: Sequence[str] = ("reliability", "final"),
    num_bins: int = DEFAULT_NUM_BINS,
) -> dict[str, Any]:
    targets = load_mc_targets(mc_records)
    models: dict[str, Any] = {}
    for variant in variants:
        predictions = flatten_predictions(prediction_records, variant)
        if set(predictions) != set(targets):
            missing = sorted(set(targets) - set(predictions))
            extra = sorted(set(predictions) - set(targets))
            raise ValueError(f"prediction/MC prefix mismatch: missing={missing[:3]}, extra={extra[:3]}")
        ordered_predictions = [predictions[prefix_id] for prefix_id in targets]
        ordered_targets = [targets[prefix_id] for prefix_id in targets]
        metrics, diagnostics = compute_metrics(ordered_predictions, ordered_targets, num_bins)
        models[variant] = {
            "num_prefixes": len(targets),
            **metrics,
            "bin_diagnostics": diagnostics,
        }
    return {
        "schema_version": 1,
        "num_bins": num_bins,
        "target": "Monte-Carlo rollout success probability K/N",
        "models": models,
    }


# Descriptive alias for callers that prefer the full metric name.
compute_calibration_metrics = compute_metrics


def _write_csv(result: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["variant", "num_prefixes", "brier", "positive_brier", "ece", "average_ce", "adaptive_ce"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for variant, metrics in result["models"].items():
            row = {"variant": variant}
            row.update({field: metrics[field] for field in fields[1:]})
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--mc-labels", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--variant", choices=("reliability", "final", "both"), default="both")
    parser.add_argument("--num-bins", type=int, default=DEFAULT_NUM_BINS)
    args = parser.parse_args()

    prediction_records = _load_records(args.predictions)
    mc_records = _load_records(args.mc_labels)
    variants = ("reliability", "final") if args.variant == "both" else (args.variant,)
    result = evaluate_calibration(prediction_records, mc_records, variants=variants, num_bins=args.num_bins)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.output_csv:
        _write_csv(result, args.output_csv)
    for variant, metrics in result["models"].items():
        print(
            f"{variant}: Brier={metrics['brier']:.6f} "
            f"PositiveBrier={metrics['positive_brier']:.6f} "
            f"ECE={metrics['ece']:.6f} AverageCE={metrics['average_ce']:.6f}"
        )


if __name__ == "__main__":
    main()
