#!/usr/bin/env python3
"""Minimal image-only challenge run against DeepSeek vision."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.benchmark_loader import load_benchmark
from sci_rag_core import figure_reference_from_question
from sci_rag_vision import complete_vision, detail_clip, figure_clip, render_figure, vision_messages


def _find_pdf(documents: list[dict[str, object]], document_id: str, papers_dirs: list[Path]) -> tuple[Path, str]:
    document = next(item for item in documents if str(item["document_id"]) == document_id)
    filename = str(document["filename"])
    for directory in papers_dirs:
        path = directory / filename
        if path.is_file():
            return path, filename
    raise FileNotFoundError(filename)


def _row(case: dict[str, object], repeat: int, model: str, **values: object) -> dict[str, object]:
    return {
        "repeat": repeat,
        "case_id": case["case_id"],
        "document_id": case["document_id"],
        "question": case["question"],
        "answer": values.get("answer"),
        "model": model,
        "source": values.get("source"),
        "page": values.get("page"),
        "figure_reference": values.get("figure_reference"),
        "clip": values.get("clip"),
        "png_sha256": values.get("png_sha256"),
        "width": values.get("width"),
        "height": values.get("height"),
        "prompt_mode": values.get("prompt_mode", "full"),
        "detail_clip": values.get("detail_clip"),
        "detail_png_sha256": values.get("detail_png_sha256"),
        "detail_width": values.get("detail_width"),
        "detail_height": values.get("detail_height"),
        "latency": values.get("latency"),
        "error": values.get("error"),
    }


def run(args: argparse.Namespace) -> int:
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    benchmark = load_benchmark(args.manifest, papers_dir=args.papers_dirs, verify_files=True)
    cases = [case for case in benchmark["cases"] if case.get("requires_image") is True]
    if len(cases) != 10:
        raise ValueError(f"challenge image-only case count is {len(cases)}, expected 10")
    references = {}
    for case in cases:
        reference = figure_reference_from_question(str(case["question"]))
        if reference is None:
            raise ValueError(f"figure reference missing: {case['case_id']}")
        references[case["case_id"]] = reference

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    if output.is_file():
        for line in output.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
                if not record.get("error"):
                    completed.add((str(record.get("prompt_mode", "full")), int(record["repeat"]), str(record["case_id"])))
            except (ValueError, KeyError, json.JSONDecodeError):
                continue

    client = OpenAI(
        api_key=os.environ.get("DEEPSEEK_API_KEY"),
        base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
    )
    documents = benchmark["documents"]
    errors = 0
    with output.open("a", encoding="utf-8") as handle:
        for repeat in range(1, args.repeats + 1):
            for case in cases:
                key = (args.mode, repeat, str(case["case_id"]))
                if key in completed:
                    continue
                reference = references[case["case_id"]]
                figure_name = f"Extended Data Figure {reference[1]}" if reference[0] == "extended_data_figure" else f"Figure {reference[1]}"
                base = {"page": None, "figure_reference": figure_name, "source": None}
                try:
                    pdf_path, source = _find_pdf(documents, str(case["document_id"]), args.papers_dirs)
                    page_number = int(case["source_pages"][0])
                    image = render_figure(
                        pdf_path,
                        reference,
                        page_number=page_number,
                        include_detail=args.mode == "full+detail",
                    )
                except Exception as exc:
                    errors += 1
                    values = {**base, "error": f"{type(exc).__name__}: {exc}"}
                else:
                    started = time.perf_counter()
                    try:
                        detail = image.get("detail") if args.mode == "full+detail" else None
                        answer = complete_vision(
                            client,
                            args.model,
                            vision_messages(
                                str(case["question"]),
                                str(image["data_url"]),
                                str(detail["data_url"]) if detail else None,
                            ),
                        )
                        values = {
                            **base,
                            **image,
                            "source": source,
                            "page": page_number,
                            "answer": answer,
                            "prompt_mode": args.mode,
                            "detail_clip": detail["clip"] if detail else None,
                            "detail_png_sha256": detail["png_sha256"] if detail else None,
                            "detail_width": detail["width"] if detail else None,
                            "detail_height": detail["height"] if detail else None,
                            "latency": round(time.perf_counter() - started, 3),
                        }
                    except Exception as exc:
                        errors += 1
                        values = {**base, **image, "source": source, "page": page_number, "error": f"{type(exc).__name__}: {exc}"}
                        handle.write(json.dumps(_row(case, repeat, args.model, **values), ensure_ascii=False) + "\n")
                        handle.flush()
                        return 1
                handle.write(json.dumps(_row(case, repeat, args.model, **values), ensure_ascii=False) + "\n")
                handle.flush()
    return 1 if errors else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="evaluation/benchmark/manifest_challenge.json")
    parser.add_argument("--papers-dir", action="append", dest="papers_dirs", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--model", default="deepseek-v4-flash-vision-exp")
    parser.add_argument("--mode", choices=("full", "full+detail"), default="full")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    args.papers_dirs = [Path(value).expanduser().resolve() for value in args.papers_dirs]
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
