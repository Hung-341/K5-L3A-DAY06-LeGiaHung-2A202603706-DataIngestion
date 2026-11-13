"""Pipeline entry point: ingest, chunk, embed, store, retrieve and evaluate A/B/C.

Run from starter/:  python -m src.run
Relative paths in .env and in the manifest are resolved from the repository root.
"""

import csv
import json
import os
import platform
import statistics
import time
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .chunking import by_paragraph, fixed_size
from .embeddings import MyEmbedder
from .evaluation import evaluate_query, summarize
from .ingestion import load_pdf
from .models import Chunk, DocumentPage
from .store import PgVectorStore


STARTER_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STARTER_ROOT.parent
STRATEGIES = ("A", "B", "C")


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _expand_pages(spec: str) -> set[int]:
    pages: set[int] = set()
    for part in spec.split(","):
        if part.strip():
            bounds = [int(value) for value in part.strip().split("-", 1)]
            pages.update(range(bounds[0], bounds[-1] + 1))
    return pages


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _chunk_stats(chunks: list[Chunk]) -> dict[str, Any]:
    lengths = [len(chunk.text) for chunk in chunks]
    return {
        "chunk_count": len(chunks),
        "total_chars": sum(lengths),
        "min_chars": min(lengths),
        "median_chars": statistics.median(lengths),
        "mean_chars": round(statistics.mean(lengths), 1),
        "max_chars": max(lengths),
    }


def _package_version(name: str) -> str:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return ""


def load_corpus(
    manifest_path: Path,
    min_page_chars: int,
) -> tuple[list[DocumentPage], list[dict[str, Any]]]:
    """Load manifest pages; log the status of every physical page."""
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest missing: {manifest_path}")

    pages: list[DocumentPage] = []
    page_log: list[dict[str, Any]] = []

    with manifest_path.open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            source_id = row["source_id"].strip()
            source = row["source_path_or_url"].strip()
            pdf_path = _resolve(source)
            if not pdf_path.is_file():
                raise FileNotFoundError(f"PDF not found: {pdf_path}")

            metadata = json.loads(row.get("metadata_json") or "{}")
            metadata["title"] = row["title"]

            source_pages = load_pdf(pdf_path, source_id, source)
            selected = _expand_pages(row.get("pages", ""))
            if not selected:
                selected = {page.page for page in source_pages}

            declared = row.get("content_page_count", "").strip()
            if declared and int(declared) != len(selected):
                raise ValueError(
                    f"{source_id}: manifest content_page_count={declared} "
                    f"but 'pages' lists {len(selected)} pages."
                )

            for page in source_pages:
                chars = len(page.text)
                if page.page not in selected:
                    status = "excluded_by_manifest_rule"
                elif chars == 0:
                    status = "empty_text"
                elif chars < min_page_chars:
                    status = "short_text"
                else:
                    status = "included"

                page_log.append({
                    "document_id": source_id,
                    "page": page.page,
                    "chars": chars,
                    "status": status,
                    "first_line": page.text.split("\n", 1)[0][:120],
                })

                if status == "included":
                    page.metadata.update(metadata)
                    pages.append(page)

    return pages, page_log


def load_questions(questions_path: Path) -> list[dict[str, Any]]:
    required = (
        "id", "split", "question", "expected_answer",
        "gold_source_id", "gold_page", "evidence_quote",
    )
    questions: list[dict[str, Any]] = []
    with questions_path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            question = json.loads(line)
            missing = [field for field in required if not question.get(field)]
            if missing:
                raise ValueError(
                    f"Câu hỏi dòng {line_number} thiếu trường: {', '.join(missing)}"
                )
            questions.append(question)

    dev_count = sum(q["split"] == "dev" for q in questions)
    test_count = sum(q["split"] == "test" for q in questions)
    filter_count = sum(bool(q.get("filter")) for q in questions)
    filtered_test_count = sum(
        1 for q in questions if q["split"] == "test" and bool(q.get("filter"))
    )

    if len(questions) < 20 or dev_count < 15 or test_count < 5:
        raise ValueError(
            f"Benchmark needs >=20 questions, >=15 dev, >=5 test; "
            f"found {len(questions)} ({dev_count} dev, {test_count} test)."
        )
    if filter_count < 3 or filtered_test_count < 1:
        raise ValueError(
            "Benchmark needs >=3 filter questions, at least 1 of them in test."
        )
    return questions


def coherence_sample(
    pages: list[DocumentPage],
    chunks_by_strategy: dict[str, list[Chunk]],
    sample_size: int = 10,
) -> list[dict[str, Any]]:
    """Fixed positions chosen before scoring: midpoint of the first N pages
    sorted by (document_id, page). Same positions for every strategy."""
    positions = sorted(pages, key=lambda page: (page.document_id, page.page))
    rows = []
    for index, page in enumerate(positions[:sample_size], start=1):
        offset = len(page.text) // 2
        for strategy in STRATEGIES:
            chunk = next(
                (
                    chunk for chunk in chunks_by_strategy[strategy]
                    if chunk.document_id == page.document_id
                    and chunk.page == page.page
                    and chunk.metadata["char_start"] <= offset < chunk.metadata["char_end"]
                ),
                None,
            )
            rows.append({
                "position_id": f"P{index:02d}",
                "document_id": page.document_id,
                "page": page.page,
                "char_offset": offset,
                "strategy": strategy,
                "chunk_id": chunk.chunk_id if chunk else "",
                "chunk_chars": len(chunk.text) if chunk else 0,
                "chunk_text": chunk.text if chunk else "",
                "coherence_score_0_to_2": "",
                "reason": "",
            })
    return rows


def main() -> None:
    load_dotenv(STARTER_ROOT / ".env")
    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    timings: dict[str, float] = {}

    manifest_path = _resolve(os.getenv("MANIFEST_PATH", "data/manifest.csv"))
    questions_path = _resolve(os.getenv("QUESTIONS_PATH", "benchmark/questions.jsonl"))
    results_dir = _resolve(os.getenv("RESULTS_DIR", "results"))
    results_dir.mkdir(parents=True, exist_ok=True)

    min_page_chars = int(os.getenv("MIN_PAGE_CHARS", "100"))
    minimum_pages = int(os.getenv("MIN_CONTENT_PAGES", "30"))

    # 1. Ingestion.
    t0 = time.perf_counter()
    pages, page_log = load_corpus(manifest_path, min_page_chars)
    timings["ingest_seconds"] = round(time.perf_counter() - t0, 2)

    page_fields = ["document_id", "page", "chars", "status", "first_line"]
    _write_csv(results_dir / "ingestion_log.csv", page_log, page_fields)
    issues = [row for row in page_log if row["status"] in {"empty_text", "short_text"}]
    _write_csv(results_dir / "ingestion_issues.csv", issues, page_fields)

    if len(pages) < minimum_pages:
        raise ValueError(
            f"Only {len(pages)} content pages found; need >= {minimum_pages}."
        )

    # 2. Benchmark (validated on the full set, then restricted to the chosen split).
    all_questions = load_questions(questions_path)
    evaluation_split = os.getenv("EVALUATION_SPLIT", "dev").lower()
    if evaluation_split not in {"dev", "both"}:
        raise ValueError("EVALUATION_SPLIT must be 'dev' or 'both'.")
    questions = (
        [q for q in all_questions if q["split"] == "dev"]
        if evaluation_split == "dev" else all_questions
    )

    # 3. Chunking: A fixed-size, B paragraph, C = B chunks + metadata filter.
    fixed_size_chars = int(os.getenv("FIXED_CHUNK_SIZE", "500"))
    overlap_chars = int(os.getenv("FIXED_CHUNK_OVERLAP", "50"))
    paragraph_max_chars = int(os.getenv("PARAGRAPH_CHUNK_SIZE", "700"))

    t0 = time.perf_counter()
    chunks_by_strategy = {
        "A": [c for p in pages for c in fixed_size(p, fixed_size_chars, overlap_chars)],
        "B": [c for p in pages for c in by_paragraph(p, paragraph_max_chars)],
    }
    chunks_by_strategy["C"] = chunks_by_strategy["B"]
    timings["chunk_seconds"] = round(time.perf_counter() - t0, 2)

    # 4. Embeddings: one model for all strategies; C reuses B's vectors.
    model_name = os.getenv("EMBEDDING_MODEL", "").strip()
    if not model_name:
        raise ValueError("Set EMBEDDING_MODEL in .env.")
    embedder = MyEmbedder(model_name)

    vectors_by_strategy: dict[str, list[list[float]]] = {}
    tokens_by_strategy: dict[str, int] = {}
    for strategy in ("A", "B"):
        chunks = chunks_by_strategy[strategy]
        tokens_before = embedder.total_tokens
        t0 = time.perf_counter()
        vectors = embedder.embed_documents([chunk.text for chunk in chunks])
        timings[f"embed_{strategy}_seconds"] = round(time.perf_counter() - t0, 2)
        tokens_by_strategy[strategy] = embedder.total_tokens - tokens_before
        if not vectors or len(vectors) != len(chunks):
            raise ValueError(f"Strategy {strategy}: vector count does not match chunk count.")
        vectors_by_strategy[strategy] = vectors
    vectors_by_strategy["C"] = vectors_by_strategy["B"]
    tokens_by_strategy["C"] = 0

    # 5. Store, then re-ingest A once to check that reloading adds no duplicates.
    dimensions = len(vectors_by_strategy["A"][0])
    store = PgVectorStore(os.getenv("DATABASE_URL", ""), dimensions)
    store.create_schema()

    t0 = time.perf_counter()
    for strategy in STRATEGIES:
        store.upsert(strategy, chunks_by_strategy[strategy], vectors_by_strategy[strategy])
    timings["upsert_seconds"] = round(time.perf_counter() - t0, 2)

    stats_after_first = {s: store.stats(s) for s in STRATEGIES}
    store.upsert("A", chunks_by_strategy["A"], vectors_by_strategy["A"])
    reload_check = {
        "strategy": "A",
        "chunks_in_memory": len(chunks_by_strategy["A"]),
        "rows_after_first_load": stats_after_first["A"]["rows"],
        "rows_after_reload": store.stats("A")["rows"],
    }
    reload_check["no_duplicates"] = (
        reload_check["rows_after_first_load"]
        == reload_check["rows_after_reload"]
        == reload_check["chunks_in_memory"]
    )

    # 6. Retrieval top-3 with full traces.
    traces = []
    query_vectors: dict[str, list[float]] = {}
    tokens_before = embedder.total_tokens
    t0 = time.perf_counter()
    for question in questions:
        question_id = question["id"]
        if question_id not in query_vectors:
            query_vectors[question_id] = embedder.embed_query(question["question"])

        for strategy in STRATEGIES:
            filters = (question.get("filter") or {}) if strategy == "C" else {}
            hits = store.search(
                strategy=strategy,
                query_vector=query_vectors[question_id],
                top_k=3,
                filters=filters,
            )
            trace = evaluate_query(question, hits)
            trace["strategy"] = strategy
            trace["question_filter"] = question.get("filter") or {}
            trace["filter_applied"] = filters
            traces.append(trace)
    timings["retrieval_seconds"] = round(time.perf_counter() - t0, 2)
    query_tokens = embedder.total_tokens - tokens_before

    # 7. Outputs.
    with (results_dir / "traces.jsonl").open("w", encoding="utf-8") as file:
        for trace in traces:
            file.write(json.dumps(trace, ensure_ascii=False) + "\n")

    metrics = summarize(traces)
    (results_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    coherence_rows = coherence_sample(pages, chunks_by_strategy)
    _write_csv(results_dir / "coherence_labels.csv", coherence_rows, list(coherence_rows[0]))

    test_traces = [t for t in traces if t["split"] == "test"]
    if test_traces:
        metadata_rows, grounding_rows = [], []
        for trace in test_traces:
            for hit in trace["hits"]:
                required = hit["metadata"]
                metadata_rows.append({
                    "strategy": trace["strategy"],
                    "question_id": trace["question_id"],
                    "rank": hit["rank"],
                    "chunk_id": hit["chunk_id"],
                    "document_id": hit["document_id"],
                    "source": hit["source"],
                    "page": hit["page"],
                    "audience": required.get("audience", ""),
                    "year": required.get("year", ""),
                    "question_filter": json.dumps(trace["question_filter"], ensure_ascii=False),
                    "filter_applied": json.dumps(trace["filter_applied"], ensure_ascii=False),
                    "metadata_complete_and_correct_0_or_1": "",
                    "note": "",
                })
            grounding_rows.append({
                "strategy": trace["strategy"],
                "question_id": trace["question_id"],
                "question": trace["question"],
                "expected_answer": trace["expected_answer"],
                **{
                    f"top{hit['rank']}": f"[{hit['document_id']} p.{hit['page']}] {hit['text']}"
                    for hit in trace["hits"]
                },
                "grounding_score_0_to_2": "",
                "reason": "",
            })
        _write_csv(results_dir / "test_metadata_audit.csv", metadata_rows, list(metadata_rows[0]))
        grounding_fields = [
            "strategy", "question_id", "question", "expected_answer",
            "top1", "top2", "top3", "grounding_score_0_to_2", "reason",
        ]
        _write_csv(results_dir / "test_grounding_audit.csv", grounding_rows, grounding_fields)

    price = os.getenv("EMBEDDING_PRICE_PER_1M_TOKENS", "").strip()
    total_tokens = embedder.total_tokens
    run_summary = {
        "started_at_utc": started_at,
        "evaluation_split": evaluation_split,
        "question_count": len(questions),
        "content_pages": len(pages),
        "content_chars": sum(len(page.text) for page in pages),
        "ingestion_issue_pages": len(issues),
        "config": {
            "A": {"chunking": "fixed_size", "size": fixed_size_chars, "overlap": overlap_chars, "filter": False},
            "B": {"chunking": "paragraph", "max_size": paragraph_max_chars, "filter": False},
            "C": {"chunking": "paragraph (same chunks/vectors as B)", "max_size": paragraph_max_chars,
                  "filter": "question.filter when present, else none"},
        },
        "embedding": {
            "model": embedder.model_name,
            "dimensions": dimensions,
            "base_url": embedder.base_url,
            "document_tokens": {s: tokens_by_strategy[s] for s in STRATEGIES},
            "query_tokens": query_tokens,
            "total_tokens": total_tokens,
            "price_per_1m_tokens_usd": float(price) if price else None,
            "estimated_cost_usd": round(total_tokens * float(price) / 1_000_000, 6) if price else None,
        },
        "vector_store": {
            **store.versions(),
            "table": "day06_chunks",
            "metric": "cosine distance (<=>)",
            "search": "exact (sequential scan, no ANN index)",
        },
        "chunks": {s: _chunk_stats(chunks_by_strategy[s]) for s in STRATEGIES},
        "stored_rows": stats_after_first,
        "reload_check": reload_check,
        "timings_seconds": timings,
        "environment": {
            "python": platform.python_version(),
            "pypdf": _package_version("pypdf"),
            "psycopg": _package_version("psycopg"),
            "python-dotenv": _package_version("python-dotenv"),
        },
    }
    (results_dir / "run_summary.json").write_text(
        json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    for row in metrics:
        print(
            f"{row['strategy']} {row['split']}: "
            f"Hit@1 {row['hit_at_1_n']}/{row['question_count']}, "
            f"Evidence Hit@3 {row['evidence_hit_at_3_n']}/{row['question_count']}"
        )
    print(f"Ingestion issues: {len(issues)} page(s) -> {results_dir / 'ingestion_issues.csv'}")
    print(f"Reload check (no duplicates): {reload_check['no_duplicates']}")
    print(f"Outputs in: {results_dir}")
    print(
        f"Completed: {len(pages)} pages, {len(questions)} questions, "
        f"model={embedder.model_name} ({dimensions} dims), tokens={total_tokens}"
    )


if __name__ == "__main__":
    main()
