#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Offline preflight checks for a saved RAGAS report.

This module never runs RAGAS or imports the application runtime. It checks
whether a report is internally complete enough for later interpretation and
labels claims that cannot be proven from the saved artifact alone.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.answer_audit import _load_cases  # noqa: E402


RAGAS_METRICS = ("nv_context_relevance", "faithfulness", "answer_relevancy")


def _case_id(record: dict[str, Any]) -> str | None:
    value = record.get("case_id", record.get("id"))
    return None if value is None else str(value)


def _load_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"报告顶层必须是对象：{source}")
    return payload


def _metric_check(summary: dict[str, Any], metric: str) -> tuple[dict[str, Any], list[str]]:
    issues: list[str] = []
    value = summary.get(metric)
    if not isinstance(value, dict):
        return {"present": False}, [f"summary 缺少指标 {metric}"]
    scores = value.get("scores")
    if not isinstance(scores, list):
        return {"present": True, "valid": False}, [f"指标 {metric} 缺少 scores 数组"]
    valid_count = sum(score is not None for score in scores)
    count_valid = value.get("count_valid")
    count_total = value.get("count_total")
    if count_total != len(scores):
        issues.append(f"指标 {metric} 的 count_total 与 scores 长度不一致")
    if count_valid != valid_count:
        issues.append(f"指标 {metric} 的 count_valid 与非空分数数量不一致")
    numeric_scores: list[float] = []
    for index, score in enumerate(scores):
        if score is None:
            continue
        try:
            numeric = float(score)
        except (TypeError, ValueError):
            issues.append(f"指标 {metric} 的 scores[{index}] 不是数值")
            continue
        if not math.isfinite(numeric):
            issues.append(f"指标 {metric} 的 scores[{index}] 不是有限数值")
            continue
        numeric_scores.append(numeric)
    mean = value.get("mean")
    if valid_count == 0:
        if mean is not None:
            issues.append(f"指标 {metric} 无有效分数但 mean 非空")
    else:
        try:
            mean_value = float(mean)
        except (TypeError, ValueError):
            mean_value = None
            issues.append(f"指标 {metric} 的 mean 不是数值")
        if mean_value is not None and (
            not math.isfinite(mean_value)
            or len(numeric_scores) != valid_count
            or not math.isclose(mean_value, sum(numeric_scores) / valid_count, abs_tol=1e-4)
        ):
            issues.append(f"指标 {metric} 的 mean 与 scores 平均值不一致")
    return {
        "present": True,
        "valid": not issues,
        "count_valid": valid_count,
        "count_total": len(scores),
    }, issues


def preflight_report(
    report: dict[str, Any],
    cases: Iterable[dict[str, Any]],
    *,
    require_complete: bool = False,
    require_trace: bool = False,
) -> dict[str, Any]:
    """Return artifact checks and explicit evidence/claim boundaries."""

    case_list = list(cases)
    expected_ids = [_case_id(case) for case in case_list]
    expected_ids_clean = [value for value in expected_ids if value is not None]
    result_list = report.get("results")
    if not isinstance(result_list, list):
        result_list = []
    result_ids = [_case_id(result) for result in result_list if isinstance(result, dict)]
    errors: list[str] = []
    warnings: list[str] = []
    checks: dict[str, Any] = {}

    duplicate_expected = sorted({value for value in expected_ids_clean if expected_ids_clean.count(value) > 1})
    duplicate_results = sorted({value for value in result_ids if result_ids.count(value) > 1})
    checks["input_case_count"] = len(expected_ids_clean)
    checks["report_result_count"] = len(result_list)
    checks["meta_case_count_matches_input"] = report.get("meta", {}).get("num_test_cases") == len(case_list)
    checks["result_count_matches_input"] = len(result_list) == len(case_list)
    checks["result_ids_unique"] = not duplicate_results and all(value is not None for value in result_ids)
    checks["input_case_ids_unique"] = not duplicate_expected
    checks["result_ids_match_input"] = set(result_ids) == set(expected_ids_clean)
    if duplicate_expected:
        errors.append(f"测试集存在重复 case id：{', '.join(duplicate_expected)}")
    if duplicate_results:
        errors.append(f"报告 results 存在重复 id：{', '.join(duplicate_results)}")
    if require_complete and not checks["result_count_matches_input"]:
        errors.append("报告结果数未覆盖测试集全部用例")
    if not checks["result_ids_match_input"]:
        errors.append("报告结果 ID 与测试集 ID 不一致")

    required_result_fields = ("question", "ground_truth", "answer", "scores")
    missing_fields: dict[str, list[str]] = {}
    for result in result_list:
        if not isinstance(result, dict):
            errors.append("results 中存在非对象记录")
            continue
        result_id = _case_id(result) or "?"
        missing = [field for field in required_result_fields if field not in result]
        if missing:
            missing_fields[result_id] = missing
    checks["required_result_fields_present"] = not missing_fields
    if missing_fields:
        errors.append("部分报告结果缺少 question/ground_truth/answer/scores")

    trace_fields = ("evaluated_contexts", "context_ids", "context_metadatas")
    generation_trace_fields = (
        "generation_contexts",
        "generation_context_ids",
        "generation_context_metadatas",
    )
    trace_presence = {
        field: bool(result_list) and all(isinstance(result, dict) and field in result for result in result_list)
        for field in trace_fields
    }
    checks["trace_fields_present"] = trace_presence
    generation_trace_presence = {
        field: bool(result_list) and all(isinstance(result, dict) and field in result for result in result_list)
        for field in generation_trace_fields
    }
    checks["generation_trace_fields_present"] = generation_trace_presence
    trace_lengths_ok = True
    generation_prefix_ok = True
    for result in result_list:
        if not isinstance(result, dict):
            continue
        contexts = result.get("evaluated_contexts")
        ids = result.get("context_ids")
        metadatas = result.get("context_metadatas")
        expected_length = result.get("num_contexts_evaluated")
        if contexts is not None and not isinstance(contexts, list):
            trace_lengths_ok = False
        if ids is not None and not isinstance(ids, list):
            trace_lengths_ok = False
        if metadatas is not None and not isinstance(metadatas, list):
            trace_lengths_ok = False
        if isinstance(contexts, list) and expected_length is not None and len(contexts) != expected_length:
            trace_lengths_ok = False
        if isinstance(ids, list) and isinstance(contexts, list) and len(ids) != len(contexts):
            trace_lengths_ok = False
        if isinstance(metadatas, list) and isinstance(contexts, list) and len(metadatas) != len(contexts):
            trace_lengths_ok = False
        generation_contexts = result.get("generation_contexts")
        generation_ids = result.get("generation_context_ids")
        generation_metas = result.get("generation_context_metadatas")
        if generation_contexts is not None and not isinstance(generation_contexts, list):
            generation_prefix_ok = False
        if generation_ids is not None and not isinstance(generation_ids, list):
            generation_prefix_ok = False
        if generation_metas is not None and not isinstance(generation_metas, list):
            generation_prefix_ok = False
        if isinstance(generation_contexts, list) and isinstance(contexts, list):
            if generation_contexts[: len(contexts)] != contexts:
                generation_prefix_ok = False
        if isinstance(generation_ids, list) and isinstance(ids, list):
            if generation_ids[: len(ids)] != ids:
                generation_prefix_ok = False
        if isinstance(generation_metas, list) and isinstance(metadatas, list):
            if generation_metas[: len(metadatas)] != metadatas:
                generation_prefix_ok = False
    checks["trace_lengths_consistent"] = trace_lengths_ok
    checks["evaluated_trace_is_generation_prefix"] = generation_prefix_ok
    if not all(trace_presence.values()):
        warnings.append("报告未完整保存 evaluated_contexts/context_ids/context_metadatas")
    if not all(generation_trace_presence.values()):
        warnings.append("报告未完整保存 generation_contexts/generation_context_ids/generation_context_metadatas")
    if not trace_lengths_ok:
        errors.append("报告上下文、ID、metadata 或 num_contexts_evaluated 长度不一致")
    if not generation_prefix_ok:
        errors.append("报告的 RAGAS evaluated trace 不是 generation trace 的稳定前缀")
    if require_trace and (
        not all(trace_presence.values())
        or not all(generation_trace_presence.values())
        or not trace_lengths_ok
        or not generation_prefix_ok
    ):
        errors.append("已要求 trace，但报告没有完整且一致的上下文 trace")

    meta = report.get("meta") if isinstance(report.get("meta"), dict) else {}
    checks["judge_model_recorded"] = bool(str(meta.get("judge_model") or "").strip())
    checks["generation_model_recorded"] = bool(
        str(meta.get("generation_model") or meta.get("answer_model") or "").strip()
    )
    checks["embedding_model_recorded"] = bool(str(meta.get("embedding_model") or "").strip())
    if not checks["generation_model_recorded"]:
        warnings.append("meta 未记录 generation_model，无法审计生成模型与 judge 是否相同")

    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    metric_checks: dict[str, Any] = {}
    for metric in RAGAS_METRICS:
        metric_checks[metric], metric_issues = _metric_check(summary, metric)
        errors.extend(metric_issues)
    checks["ragas_metrics"] = metric_checks

    checks["ground_truth_attached_to_results"] = bool(result_list) and all(
        isinstance(result, dict) and isinstance(result.get("ground_truth"), str) and result["ground_truth"].strip()
        for result in result_list
    )
    checks["reference_contexts_saved"] = bool(result_list) and all(
        isinstance(result, dict) and "reference_contexts" in result for result in result_list
    )
    checks["generation_contexts_saved_separately"] = bool(result_list) and all(
        isinstance(result, dict) and "generation_contexts" in result for result in result_list
    )
    if not checks["ground_truth_attached_to_results"]:
        errors.append("报告结果缺少可核对的 ground_truth")
    if not checks["reference_contexts_saved"]:
        warnings.append("报告未保存 reference_contexts，无法从 artifact 证明 RAGAS 的参考上下文输入")
    metric_inputs = meta.get("ragas_metric_input_columns")
    ground_truth_metric_usage = "not_provable_from_saved_report"
    if isinstance(metric_inputs, dict):
        reference_metrics = [
            metric
            for metric in RAGAS_METRICS
            if isinstance(metric_inputs.get(metric), list)
            and "reference" in metric_inputs[metric]
        ]
        checks["ragas_metric_input_columns_recorded"] = all(
            isinstance(metric_inputs.get(metric), list) for metric in RAGAS_METRICS
        )
        checks["ground_truth_required_by_any_ragas_metric"] = bool(reference_metrics)
        if checks["ragas_metric_input_columns_recorded"] and not reference_metrics:
            ground_truth_metric_usage = "not_used_by_declared_metrics"
            warnings.append(
                "报告记录的三项 RAGAS metric input columns 均不含 reference；这些指标不使用 ground_truth"
            )
        else:
            warnings.append("报告记录的 metric input columns 不足以证明每个 RAGAS 指标实际使用了 ground_truth")
    else:
        checks["ragas_metric_input_columns_recorded"] = False
        checks["ground_truth_required_by_any_ragas_metric"] = None
        warnings.append("仅凭报告中的 ground_truth 字段，不能证明每个 RAGAS 指标实际使用了 ground_truth")
    warnings.append("仅凭 evaluated_contexts，不能证明生成时上下文与 RAGAS 评估上下文完全一致")

    return {
        "schema": "sci-rag-ragas-preflight-v1",
        "evidence_level": "artifact_consistency_only",
        "ready_for_interpretation": not errors,
        "errors": errors,
        "warnings": warnings,
        "checks": checks,
        "claim_boundaries": {
            "ground_truth_metric_usage": ground_truth_metric_usage,
            "generation_vs_evaluation_context_identity": "not_provable_from_saved_report",
            "semantic_answer_correctness": "not_measured_by_preflight",
            "cross_paper_generalization": "not_measured_by_preflight",
        },
    }


def _print_summary(report: dict[str, Any]) -> None:
    print(f"RAGAS 预检：{'可继续解释' if report['ready_for_interpretation'] else '存在阻塞问题'}")
    print(f"错误：{len(report['errors'])}；警告：{len(report['warnings'])}")
    for issue in report["errors"]:
        print(f"- ERROR: {issue}")
    for issue in report["warnings"]:
        print(f"- WARNING: {issue}")


def main() -> int:
    parser = argparse.ArgumentParser(description="离线预检已保存的 RAGAS 报告")
    parser.add_argument("--report-json", required=True, help="已保存的 RAGAS JSON 报告")
    parser.add_argument("--testset", required=True, help="JSON/JSONL 测试集路径")
    parser.add_argument("--require-complete", action="store_true", help="要求报告覆盖测试集全部用例")
    parser.add_argument("--require-trace", action="store_true", help="要求报告保存完整上下文 trace")
    parser.add_argument("--json-out", help="可选：写入预检结果 JSON")
    args = parser.parse_args()
    report = preflight_report(
        _load_json(args.report_json),
        _load_cases(args.testset),
        require_complete=args.require_complete,
        require_trace=args.require_trace,
    )
    _print_summary(report)
    if args.json_out:
        output = Path(args.json_out)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"完整 JSON：{output}")
    return 0 if report["ready_for_interpretation"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
