"""Metrics and slot matching for v8 action-slot training/evaluation."""
from __future__ import annotations

from itertools import permutations
from typing import Any

NO_LABEL = -1


def match_action_slots(predicted: list[dict[str, Any]], truth: list[dict[str, Any]]) -> list[tuple[int, int]]:
    """Return ``(predicted_index, truth_index)`` matches using a tiny assignment search.

    The action-slot count is fixed at K=8, so brute-force permutation search is
    faster to maintain than bringing in SciPy.
    """
    if not predicted or not truth:
        return []
    pred_indices = list(range(len(predicted)))
    truth_indices = list(range(len(truth)))
    if len(pred_indices) >= len(truth_indices):
        best_score = float("-inf")
        best_matches: list[tuple[int, int]] = []
        for chosen in permutations(pred_indices, len(truth_indices)):
            matches = list(zip(chosen, truth_indices))
            score = sum(_match_score(predicted[pred_i], truth[truth_i]) for pred_i, truth_i in matches)
            if score > best_score:
                best_score = score
                best_matches = matches
        return best_matches

    best_score = float("-inf")
    best_matches = []
    for chosen_truth in permutations(truth_indices, len(pred_indices)):
        matches = list(zip(pred_indices, chosen_truth))
        score = sum(_match_score(predicted[pred_i], truth[truth_i]) for pred_i, truth_i in matches)
        if score > best_score:
            best_score = score
            best_matches = matches
    return sorted(best_matches, key=lambda item: item[1])


def compute_action_metrics(
    truth_batches: list[list[dict[str, Any]]],
    predicted_batches: list[list[dict[str, Any]]],
    *,
    fire_threshold: float = 0.5,
) -> dict[str, float]:
    true_positive = 0
    predicted_positive = 0
    truth_positive = 0
    source_hits = 0
    target_hits = 0
    target_top2_hits = 0
    target_top3_hits = 0
    amount_hits = 0
    amount_non_all_in_hits = 0
    amount_non_all_in_total = 0
    duplicate_source_batches = 0
    fire_slots = 0
    total_slots = 0

    for truth, predicted in zip(truth_batches, predicted_batches):
        fired = [action for action in predicted if float(action.get("fire", 1.0)) >= fire_threshold]
        predicted_positive += len(fired)
        truth_positive += len(truth)
        fire_slots += len(fired)
        total_slots += max(len(predicted), len(fired), 1)
        sources = [action.get("source") for action in fired]
        duplicate_source_batches += int(len(sources) != len(set(sources)))
        matches = match_action_slots(fired, truth)
        for pred_index, truth_index in matches:
            pred = fired[pred_index]
            expected = truth[truth_index]
            source_ok = pred.get("source") == expected.get("source")
            target_ok = pred.get("target") == expected.get("target")
            amount_ok = pred.get("amount") == expected.get("amount")
            if source_ok:
                source_hits += 1
            if target_ok:
                target_hits += 1
            if _contains_target(pred, expected.get("target"), 2):
                target_top2_hits += 1
            if _contains_target(pred, expected.get("target"), 3):
                target_top3_hits += 1
            if amount_ok:
                amount_hits += 1
            if expected.get("amount") != 0:
                amount_non_all_in_total += 1
                amount_non_all_in_hits += int(amount_ok)
            true_positive += int(source_ok and target_ok)

    precision = _safe_div(true_positive, predicted_positive)
    recall = _safe_div(true_positive, truth_positive)
    return {
        "actionPrecision": precision,
        "actionRecall": recall,
        "actionF1": _safe_div(2.0 * precision * recall, precision + recall),
        "sourceTop1": _safe_div(source_hits, truth_positive),
        "targetTop1": _safe_div(target_hits, truth_positive),
        "targetTop2": _safe_div(target_top2_hits, truth_positive),
        "targetTop3": _safe_div(target_top3_hits, truth_positive),
        "amountAccuracy": _safe_div(amount_hits, truth_positive),
        "amountNonAllInAccuracy": _safe_div(amount_non_all_in_hits, amount_non_all_in_total),
        "duplicateSourceRate": _safe_div(duplicate_source_batches, len(predicted_batches)),
        "fireRate": _safe_div(fire_slots, total_slots),
    }


def _match_score(predicted: dict[str, Any], truth: dict[str, Any]) -> float:
    score = float(predicted.get("fire", 1.0))
    score += 4.0 if predicted.get("source") == truth.get("source") else 0.0
    score += 2.0 if predicted.get("target") == truth.get("target") else 0.0
    score += 1.0 if predicted.get("amount") == truth.get("amount") else 0.0
    return score


def _contains_target(predicted: dict[str, Any], target: Any, top_k: int) -> bool:
    candidates = predicted.get("target_topk")
    if isinstance(candidates, list):
        return target in candidates[:top_k]
    return predicted.get("target") == target


def _safe_div(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else float(numerator) / float(denominator)
