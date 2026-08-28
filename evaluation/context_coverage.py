#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Deterministic required-fact coverage for retrieved benchmark contexts.

The matcher is intentionally lexical and auditable. Cross-language or surface
form equivalence must be declared per case through ``required_fact_aliases``;
no LLM or embedding model decides whether a fact is present.
"""

from __future__ import annotations

import html
import re
import unicodedata
from typing import Any, Iterable

from sci_rag_core import normalize_for_match


_COMPACT_PUNCTUATION_RE = re.compile(r"\s*([=,()<>*×^])\s*")


def _normalized_forms(value: Any) -> set[str]:
    """Return raw-markup and text-normalized forms for exact matching."""

    raw = unicodedata.normalize("NFKC", html.unescape(str(value))).casefold().strip()
    cleaned = normalize_for_match(value).strip()
    forms: set[str] = set()
    for form in (raw, cleaned):
        if not form:
            continue
        form = re.sub(r"\s+", " ", form)
        forms.add(form)
        forms.add(_COMPACT_PUNCTUATION_RE.sub(r"\1", form))
    return forms


def _contains_surface(text: str, surface: str) -> bool:
    """Match a surface with token boundaries around alphanumeric edges."""

    if not surface:
        return False
    prefix = r"(?<!\w)" if surface[0].isalnum() else ""
    suffix = r"(?!\w)" if surface[-1].isalnum() else ""
    return re.search(prefix + re.escape(surface) + suffix, text) is not None


def aliases_for_fact(case: dict[str, Any], fact: str) -> list[str]:
    """Return the canonical fact followed by explicitly declared aliases."""

    aliases = case.get("required_fact_aliases") or {}
    declared = aliases.get(fact, []) if isinstance(aliases, dict) else []
    return list(dict.fromkeys([fact, *(str(value) for value in declared)]))


def fact_is_present(
    fact: str,
    contexts: Iterable[str],
    aliases: Iterable[str] = (),
) -> bool:
    """Return whether any canonical/alias surface occurs in any context."""

    context_forms = [_normalized_forms(context) for context in contexts]
    surfaces = set()
    for value in (fact, *aliases):
        surfaces.update(_normalized_forms(value))
    return any(
        _contains_surface(context_form, surface)
        for forms in context_forms
        for context_form in forms
        for surface in surfaces
    )


def case_fact_coverage(case: dict[str, Any], contexts: Iterable[str]) -> dict[str, Any]:
    """Classify one case as full, partial, zero, or not scored."""

    required = [str(fact) for fact in case.get("required_facts") or []]
    context_list = [str(context) for context in contexts]
    if not required:
        return {
            "required_fact_coverage": None,
            "fact_coverage_status": "not_scored",
            "matched_required_facts": [],
            "missing_required_facts": [],
            "matched_required_fact_count": 0,
            "required_fact_count": 0,
        }

    matched = [
        fact
        for fact in required
        if fact_is_present(fact, context_list, aliases_for_fact(case, fact)[1:])
    ]
    missing = [fact for fact in required if fact not in matched]
    coverage = len(matched) / len(required)
    status = "full" if not missing else "partial" if matched else "zero"
    return {
        "required_fact_coverage": coverage,
        "fact_coverage_status": status,
        "matched_required_facts": matched,
        "missing_required_facts": missing,
        "matched_required_fact_count": len(matched),
        "required_fact_count": len(required),
    }


def aggregate_fact_coverage(rows: Iterable[dict[str, Any]]) -> dict[str, int | float | None]:
    """Return case-macro, fact-micro, and full/partial/zero rates."""

    scored = [row for row in rows if int(row.get("required_fact_count") or 0) > 0]
    if not scored:
        return {
            "required_fact_coverage_macro": None,
            "required_fact_coverage_micro": None,
            "full_fact_coverage_rate": None,
            "partial_fact_coverage_rate": None,
            "zero_fact_coverage_rate": None,
            "fact_scored_cases": 0,
            "required_fact_count": 0,
        }
    total_facts = sum(int(row["required_fact_count"]) for row in scored)
    matched_facts = sum(int(row["matched_required_fact_count"]) for row in scored)
    total_cases = len(scored)
    statuses = [str(row["fact_coverage_status"]) for row in scored]
    return {
        "required_fact_coverage_macro": sum(
            float(row["required_fact_coverage"]) for row in scored
        )
        / total_cases,
        "required_fact_coverage_micro": matched_facts / total_facts,
        "full_fact_coverage_rate": statuses.count("full") / total_cases,
        "partial_fact_coverage_rate": statuses.count("partial") / total_cases,
        "zero_fact_coverage_rate": statuses.count("zero") / total_cases,
        "fact_scored_cases": total_cases,
        "required_fact_count": total_facts,
    }


def unsupported_gold_facts(case: dict[str, Any]) -> list[str]:
    """Return facts whose canonical/alias surfaces are absent from gold contexts."""

    coverage = case_fact_coverage(case, case.get("contexts") or [])
    return list(coverage["missing_required_facts"])
