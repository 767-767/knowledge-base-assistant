"""Small local-only paper QA experiment; results require semantic review."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from tempfile import mkdtemp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import app
from evaluation.generation_stability import _load_rows, _write_rows, runtime_config_trace, select_cases


def find_paper(name: str, directories: list[Path]) -> Path:
    for directory in directories:
        path = directory / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"paper not found in supplied directories: {name}")


def replay_request(row: dict, model: str, no_thinking: bool = False) -> dict:
    """Change the model, not the saved evidence, prompts or sampling budget."""
    request = {**row["calls"][0]["request"], "model": model}
    if no_thinking:
        request["reasoning_effort"] = "none"
    return request


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--papers-dir", type=Path, action="append")
    parser.add_argument("--cases-file", type=Path, default=ROOT / "evaluation/benchmark/paper_pilot_cases.jsonl")
    parser.add_argument("--replay-trace", type=Path, help="Replay saved model calls without retrieval")
    parser.add_argument("--model", default="qwen3:4b-instruct")
    parser.add_argument("--no-thinking", action="store_true")
    parser.add_argument("--work-dir", type=Path, default=ROOT / "tmp/paper_pilot")
    parser.add_argument("--document-routing", action="store_true")
    parser.add_argument("--query-decomposition", action="store_true")
    parser.add_argument("--case-id", action="append")
    parser.add_argument("--use-case-sources", action="store_true", help="Limit each question to its annotated paper(s)")
    args = parser.parse_args()
    if args.papers_dir is None and args.replay_trace is None:
        parser.error("provide --papers-dir or --replay-trace")
    if args.replay_trace is not None and not args.replay_trace.is_file():
        parser.error(f"replay trace not found: {args.replay_trace}")
    if args.replay_trace is None and not args.cases_file.is_file():
        parser.error(f"cases file not found: {args.cases_file}")
    os.environ["HF_HUB_OFFLINE"] = "1"
    from openai import OpenAI

    # Deliberately ignore .env credentials: this pilot only calls local Ollama.
    config = app.RuntimeConfig(
        db_path=str(args.work_dir / "chroma"),
        llm_base_url="http://127.0.0.1:11434/v1",
        llm_model=args.model,
        document_routing=args.document_routing,
        query_decomposition=args.query_decomposition,
        hybrid_candidate_k=12,  # Inactive in dense mode; matches the first pilot trace.
    )
    client = OpenAI(base_url=config.llm_base_url, api_key="ollama", timeout=240, max_retries=0)
    client.models.retrieve(config.llm_model)
    if args.replay_trace is not None:
        args.work_dir.mkdir(parents=True, exist_ok=True)
        output = Path(mkdtemp(prefix="replay-", dir=args.work_dir)) / "trace.jsonl"
        rows = []
        print(f"TRACE {output}", flush=True)
        for original in select_cases(_load_rows(args.replay_trace), args.case_id):
            if not original.get("calls"):
                continue  # Deterministic table answers have no model call to replay.
            request = replay_request(original, args.model, args.no_thinking)
            started = time.perf_counter()
            response = client.chat.completions.create(**request)
            answer = response.choices[0].message.content or ""
            row = {**original, "answer": answer,
                   "source_trace": str(args.replay_trace.resolve()),
                   "latency_seconds": time.perf_counter() - started,
                   "runtime_config": {**original["runtime_config"], "llm_model": args.model},
                   "calls": [{"request": request, "response": response.model_dump()}]}
            rows.append(row)
            _write_rows(output, rows)
            print(f"{row['case_id']} {row['latency_seconds']:.1f}s "
                  f"{response.choices[0].finish_reason} {answer}", flush=True)
        return

    import chromadb
    from sentence_transformers import SentenceTransformer

    cases = _load_rows(args.cases_file)
    papers = sorted({name for case in cases for name in case["source_pages"]})
    cases = select_cases(cases, args.case_id)
    collection = chromadb.PersistentClient(path=config.db_path).get_or_create_collection(
        "knowledge_base", metadata={"hnsw:space": "cosine"}
    )
    calls = []

    def generate(**request):
        if args.no_thinking:
            request["reasoning_effort"] = "none"
        call = {"request": request}
        calls.append(call)
        response = client.chat.completions.create(**request)
        call["response"] = response.model_dump()
        return response

    recording_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=generate)))
    runtime = app.Runtime(config, recording_client, SentenceTransformer(config.embedding_model), collection)
    for name in papers:
        print(app.add_document_to_db(str(find_paper(name, args.papers_dir)), runtime), flush=True)

    # Each invocation keeps its own trace; a failed answer is not silently retried.
    mode = "routed" if args.document_routing else "dense"
    output = Path(mkdtemp(prefix=mode + "-", dir=args.work_dir)) / "trace.jsonl"
    rows = []
    print(f"TRACE {output}", flush=True)
    for case in cases:
        calls.clear()
        started = time.perf_counter()
        source_filter = list(case["source_pages"]) if args.use_case_sources else None
        result = app.query_knowledge(
            case["question"], runtime=runtime, return_contexts=True,
            source_filter=source_filter,
        )
        row = {
            "case_id": case["case_id"], "question": case["question"],
            "source_filter": source_filter,
            "runtime_config": runtime_config_trace(runtime),
            "latency_seconds": time.perf_counter() - started,
            "db_chunks": collection.count(), "calls": list(calls), **result,
        }
        rows.append(row)
        _write_rows(output, rows)
        print(f"{case['case_id']} {row['latency_seconds']:.1f}s {row['answer']}", flush=True)


if __name__ == "__main__":
    main()
