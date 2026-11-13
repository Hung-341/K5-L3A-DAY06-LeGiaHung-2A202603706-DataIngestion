"""Evaluate retrieval hits and summarize benchmark metrics."""
import re
import unicodedata
from typing import Any


def _normalize(value: str) -> str:
    value = unicodedata.normalize("NFKC", value or "")
    return re.sub(r"\s+", " ", value).strip().casefold()


def _gold_items(question: dict[str, Any]) -> list[dict[str, Any]]:
    alternatives = question.get("gold_evidence")
    if alternatives:
        return alternatives

    return [{
        "source_id": question["gold_source_id"],
        "page": question["gold_page"],
        "evidence_quote": question["evidence_quote"],
    }]


def _matches_gold(hit: dict[str, Any], gold_items: list[dict[str, Any]]) -> bool:
    hit_text = _normalize(hit.get("text", ""))

    for gold in gold_items:
        same_source = hit.get("document_id") == gold.get("source_id")
        same_page = hit.get("page") == gold.get("page")
        quote = _normalize(gold.get("evidence_quote", ""))

        if same_source and same_page and quote and quote in hit_text:
            return True

    return False


def evaluate_query(
    question: dict[str, Any],
    hits: list[dict[str, Any]],
) -> dict[str, Any]:
    """Mark exact gold evidence matches and preserve the full retrieval trace."""
    gold_items = _gold_items(question)
    traced_hits = []

    for rank, hit in enumerate(hits, start=1):
        traced_hits.append({
            **hit,
            "rank": rank,
            "is_gold_evidence": _matches_gold(hit, gold_items),
        })

    return {
        "question_id": question["id"],
        "split": question["split"],
        "question": question["question"],
        "expected_answer": question["expected_answer"],
        "gold_evidence": gold_items,
        "hits": traced_hits,
        "hit_at_1": bool(
            traced_hits and traced_hits[0]["is_gold_evidence"]
        ),
        "evidence_hit_at_3": any(
            hit["is_gold_evidence"] for hit in traced_hits[:3]
        ),
    }


def summarize(traces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate retrieval metrics by strategy and benchmark split."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}

    for trace in traces:
        key = (trace["strategy"], trace["split"])
        groups.setdefault(key, []).append(trace)

    summary = []
    for (strategy, split), rows in sorted(groups.items()):
        total = len(rows)
        hit1 = sum(bool(row["hit_at_1"]) for row in rows)
        hit3 = sum(bool(row["evidence_hit_at_3"]) for row in rows)
        summary.append({
            "strategy": strategy,
            "split": split,
            "question_count": total,
            "hit_at_1_n": hit1,
            "hit_at_1": hit1 / total,
            "evidence_hit_at_3_n": hit3,
            "evidence_hit_at_3": hit3 / total,
        })

    return summary