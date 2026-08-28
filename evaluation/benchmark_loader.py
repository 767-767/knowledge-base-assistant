#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Offline loader and validator for the multi-paper benchmark manifest."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
import hashlib
import json
from pathlib import Path
from typing import Any


BENCHMARK_DIR = Path(__file__).resolve().parent / "benchmark"
DEFAULT_MANIFEST = BENCHMARK_DIR / "manifest.json"


class BenchmarkValidationError(ValueError):
    """Raised when a benchmark manifest or case reference is invalid."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise BenchmarkValidationError(f"无法读取 JSON：{path}: {exc}") from exc
    if not isinstance(value, dict):
        raise BenchmarkValidationError(f"JSON 顶层必须是对象：{path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise BenchmarkValidationError(f"无法读取 JSONL：{path}: {exc}") from exc
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise BenchmarkValidationError(f"JSONL 第 {line_number} 行无效：{path}: {exc}") from exc
        if not isinstance(value, dict):
            raise BenchmarkValidationError(f"JSONL 第 {line_number} 行必须是对象：{path}")
        records.append(value)
    return records


def _case_id(value: Any) -> str:
    return str(value)


def _normalise_papers_dirs(
    papers_dir: str | Path | Sequence[str | Path],
) -> list[Path]:
    if isinstance(papers_dir, (str, Path)):
        return [Path(papers_dir).resolve()]
    return [Path(directory).resolve() for directory in papers_dir]


def _resolve_source_case(pointer: dict[str, Any], cases_path: Path) -> dict[str, Any]:
    source_testset = pointer.get("source_testset")
    source_case_id = pointer.get("source_case_id")
    if not source_testset or source_case_id is None:
        required = {"question", "ground_truth", "contexts"}
        missing = sorted(required - pointer.keys())
        if missing:
            raise BenchmarkValidationError(
                f"内联用例 {pointer.get('case_id', '?')} 缺少字段：{', '.join(missing)}"
            )
        return dict(pointer)

    source_path = (cases_path.parent / str(source_testset)).resolve()
    source_data = _load_json(source_path)
    source_cases = source_data.get("test_cases")
    if not isinstance(source_cases, list):
        raise BenchmarkValidationError(f"源测试集缺少 test_cases：{source_path}")
    target = _case_id(source_case_id)
    matching = [case for case in source_cases if _case_id(case.get("id")) == target]
    if len(matching) != 1:
        raise BenchmarkValidationError(
            f"用例 {pointer.get('case_id', '?')} 找不到唯一 source_case_id={source_case_id}：{source_path}"
        )
    resolved = dict(matching[0])
    resolved.update({key: value for key, value in pointer.items() if key not in {"source_testset", "source_case_id"}})
    return resolved


def _validate_resolved_case(case: dict[str, Any], case_id: str) -> None:
    for field in ("question", "ground_truth"):
        if not isinstance(case.get(field), str) or not case[field].strip():
            raise BenchmarkValidationError(f"用例 {case_id} 的 {field} 必须是非空字符串")
    contexts = case.get("contexts")
    if not isinstance(contexts, list) or not contexts or not all(
        isinstance(context, str) and context.strip() for context in contexts
    ):
        raise BenchmarkValidationError(f"用例 {case_id} 的 contexts 必须是非空字符串数组")
    required_facts = case.get("required_facts")
    if required_facts is not None and (
        not isinstance(required_facts, list)
        or not required_facts
        or not all(isinstance(fact, str) and fact.strip() for fact in required_facts)
    ):
        raise BenchmarkValidationError(f"用例 {case_id} 的 required_facts 必须是非空字符串数组")
    aliases = case.get("required_fact_aliases")
    if aliases is not None:
        if not isinstance(aliases, dict):
            raise BenchmarkValidationError(
                f"用例 {case_id} 的 required_fact_aliases 必须是对象"
            )
        unknown = sorted(set(aliases) - set(required_facts or []))
        if unknown:
            raise BenchmarkValidationError(
                f"用例 {case_id} 的 required_fact_aliases 含未知事实：{', '.join(unknown)}"
            )
        for fact, values in aliases.items():
            if (
                not isinstance(values, list)
                or not values
                or not all(isinstance(value, str) and value.strip() for value in values)
            ):
                raise BenchmarkValidationError(
                    f"用例 {case_id} 的事实 {fact} 别名必须是非空字符串数组"
                )


def load_benchmark(
    manifest_path: str | Path = DEFAULT_MANIFEST,
    papers_dir: str | Path | Sequence[str | Path] | None = None,
    verify_files: bool = False,
) -> dict[str, Any]:
    """Load and validate a benchmark without importing app/model dependencies."""

    manifest_path = Path(manifest_path).resolve()
    manifest = _load_json(manifest_path)
    benchmark_dir = manifest_path.parent
    documents = manifest.get("documents")
    if not isinstance(documents, list) or not documents:
        raise BenchmarkValidationError("manifest.documents 必须是非空数组")

    document_ids: set[str] = set()
    normalized_documents: list[dict[str, Any]] = []
    for document in documents:
        if not isinstance(document, dict):
            raise BenchmarkValidationError("manifest.documents 中每项必须是对象")
        missing = sorted({"document_id", "filename", "sha256"} - document.keys())
        if missing:
            raise BenchmarkValidationError(f"文档条目缺少字段：{', '.join(missing)}")
        document_id = str(document["document_id"])
        if document_id in document_ids:
            raise BenchmarkValidationError(f"重复的 document_id：{document_id}")
        document_ids.add(document_id)
        normalized = dict(document)
        normalized["document_id"] = document_id
        normalized["sha256"] = str(document["sha256"]).lower()
        normalized_documents.append(normalized)

    if verify_files:
        if papers_dir is None:
            raise BenchmarkValidationError("--verify-files 需要同时提供 --papers-dir")
        papers_dirs = _normalise_papers_dirs(papers_dir)
        if not papers_dirs:
            raise BenchmarkValidationError("--verify-files 至少需要一个 --papers-dir")
        for document in normalized_documents:
            candidates = [directory / document["filename"] for directory in papers_dirs]
            path = next((candidate for candidate in candidates if candidate.is_file()), None)
            if path is None:
                searched = "、".join(str(candidate) for candidate in candidates)
                raise BenchmarkValidationError(f"找不到论文文件：{searched}")
            actual = _sha256(path)
            if actual != document["sha256"]:
                raise BenchmarkValidationError(
                    f"SHA-256 不一致：{document['filename']}，清单={document['sha256']}，实际={actual}"
                )

    cases_name = manifest.get("cases_path")
    if not cases_name:
        raise BenchmarkValidationError("manifest 缺少 cases_path")
    cases_path = (benchmark_dir / str(cases_name)).resolve()
    pointers = _load_jsonl(cases_path)
    if not pointers:
        raise BenchmarkValidationError("cases.jsonl 为空")

    case_ids: set[str] = set()
    cases: list[dict[str, Any]] = []
    for pointer in pointers:
        if "case_id" not in pointer or "document_id" not in pointer:
            raise BenchmarkValidationError("每个 benchmark 用例必须包含 case_id 和 document_id")
        case_id = _case_id(pointer["case_id"])
        document_id = str(pointer["document_id"])
        if case_id in case_ids:
            raise BenchmarkValidationError(f"重复的 case_id：{case_id}")
        if document_id not in document_ids:
            raise BenchmarkValidationError(f"用例 {case_id} 引用了未知 document_id：{document_id}")
        case_ids.add(case_id)
        resolved = _resolve_source_case(pointer, cases_path)
        _validate_resolved_case(resolved, case_id)
        resolved["case_id"] = case_id
        resolved["document_id"] = document_id
        cases.append(resolved)

    return {
        "manifest": manifest,
        "documents": normalized_documents,
        "cases": cases,
        "cases_path": cases_path,
    }


def benchmark_summary(benchmark: dict[str, Any]) -> dict[str, Any]:
    """Return stable counts used by the CLI and tests."""

    documents = benchmark["documents"]
    cases = benchmark["cases"]
    counts = Counter(case["document_id"] for case in cases)
    minimum_documents = int(benchmark["manifest"].get("minimum_documents", 1))
    return {
        "benchmark_id": benchmark["manifest"].get("benchmark_id", ""),
        "documents": len(documents),
        "cases": len(cases),
        "cases_per_document": dict(sorted(counts.items())),
        "minimum_documents": minimum_documents,
        "complete": len(documents) >= minimum_documents,
    }
