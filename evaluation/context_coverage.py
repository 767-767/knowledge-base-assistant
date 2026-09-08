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

from sci_rag_core import normalize_for_match, parse_markdown_table


_COMPACT_PUNCTUATION_RE = re.compile(r"\s*([=,()<>≤≥*×^])\s*")
_PARENTHETICAL_OPERATOR_RE = re.compile(r"\(\s*([<>≤≥=]\s*[^()]*)\s*\)")
_FACT_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[./-][a-z0-9]+)*", re.IGNORECASE)


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
        forms.add(re.sub(r"(?<=\d),(?=\d)", "", form))
        # PDF prose often wraps a threshold in parentheses while the
        # annotation writes it inline (``entropy (≥ 4.5)`` vs
        # ``entropy ≥4.5``). Keep an equivalent operator form.
        operator_form = _PARENTHETICAL_OPERATOR_RE.sub(r" \1 ", form)
        operator_form = re.sub(r"\s+", " ", operator_form).strip()
        forms.add(operator_form)
        forms.add(_COMPACT_PUNCTUATION_RE.sub(r"\1", operator_form))
        # Table extraction may put a unit in parentheses before the value
        # (``Avg. Time (s) 50.91``), while annotations keep it after the
        # value. Preserve an equivalent form without guessing the unit.
        unit_after_value = re.sub(
            r"\(\s*([A-Za-z%]+)\s*\)\s*([+-]?(?:\d+(?:\.\d+)?|\.\d+))",
            r"\2 \1",
            form,
        )
        unit_after_value = re.sub(r"\s+", " ", unit_after_value).strip()
        forms.add(unit_after_value)
        forms.add(_COMPACT_PUNCTUATION_RE.sub(r"\1", unit_after_value))
        # PDF list extraction can put a bullet hyphen on its own line between
        # two words (``independent - facts``); keep a comparison form without
        # that layout artifact while preserving real hyphenated terms.
        forms.add(re.sub(r"(?<=[A-Za-z])\s+-\s+(?=[A-Za-z])", " ", form))
    return forms


def _context_forms(value: Any) -> set[str]:
    """Return whole-context and line-local forms without merging table rows."""

    text = str(value or "")
    forms = set(_normalized_forms(text))
    for line in text.splitlines():
        if not line.strip():
            continue
        forms.update(_normalized_forms(line))
        # Structured evidence uses ``field=value`` (or a colon/semicolon)
        # and Markdown tables use pipe-separated cells. Normalize each line
        # independently so a fact cannot borrow values from another row.
        structured = re.sub(r"\s*(?:=|:|;|；)\s*", " ", line)
        forms.update(_normalized_forms(structured))
        if "|" in line:
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            cells = [cell for cell in cells if cell and not re.fullmatch(r":?-{3,}:?", cell)]
            if cells:
                forms.update(_normalized_forms(" ".join(cells)))
    return forms


def _contains_surface(text: str, surface: str) -> bool:
    """Match a surface with token boundaries around alphanumeric edges."""

    if not surface:
        return False
    # ``\w`` treats CJK characters as word characters, so applying an ASCII
    # word boundary to a one-character Chinese fact such as ``块`` would fail
    # inside natural Chinese prose (``相关块并...``). Keep boundaries for
    # ASCII/numeric surfaces to avoid partial matches such as ``42`` in ``420``;
    # use substring matching for non-ASCII surfaces.
    ascii_start = surface[0].isascii() and surface[0].isalnum()
    ascii_end = surface[-1].isascii() and surface[-1].isalnum()
    prefix = r"(?<!\w)" if ascii_start else ""
    suffix = r"(?!\w)" if ascii_end else ""
    return re.search(prefix + re.escape(surface) + suffix, text) is not None


def _contains_ordered_fact_tokens(text: str, surface: str) -> bool:
    """Match a multi-token fact within one line without crossing rows."""

    tokens = _FACT_TOKEN_RE.findall(surface.casefold())
    if len(tokens) < 2:
        return False
    position = 0
    for token in tokens:
        match = re.search(
            rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])",
            text[position:].casefold(),
        )
        if match is None:
            return False
        position += match.end()
    return True


def _table_fact_present(fact: str, aliases: Iterable[str], context: str) -> bool:
    """Match facts expressed as a table row/column/value relationship."""

    parsed = parse_markdown_table(context)
    if parsed is None:
        return False
    headers, rows = parsed
    surfaces = set()
    for value in (fact, *aliases):
        surfaces.update(_normalized_forms(value))
    for row in rows:
        cells = [normalize_for_match(cell) for cell in row]
        labels = [
            cell
            for cell in cells
            if cell and not re.search(r"[0-9]", cell)
        ]
        row_label = labels[0] if labels else ""
        for index, value in enumerate(cells):
            if not value:
                continue
            header = normalize_for_match(headers[index]) if index < len(headers) else ""
            if not header:
                continue
            header_variants = {header}
            header_variants.add(re.sub(r"\btable\b", "tables", header))
            for header_variant in header_variants:
                combinations = (
                    f"{value} {row_label}",
                    f"{value} {row_label} {header_variant}",
                    f"{row_label} {value} {header_variant}",
                    f"{header_variant} {row_label} {value}",
                    f"{header_variant} {value} {row_label}",
                )
                if any(
                    _contains_ordered_fact_tokens(
                        normalize_for_match(candidate), surface
                    )
                    for candidate in combinations
                    for surface in surfaces
                ):
                    return True
    return False


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

    context_list = [str(context) for context in contexts]
    context_forms = [_context_forms(context) for context in context_list]
    surfaces = set()
    for value in (fact, *aliases):
        surfaces.update(_normalized_forms(value))
    if any(
        _contains_surface(context_form, surface)
        for forms in context_forms
        for context_form in forms
        for surface in surfaces
    ):
        return True
    if any(
        _table_fact_present(fact, aliases, context)
        for context in context_list
    ):
        return True
    # Structured evidence may place a row label, field name, and value on the
    # same line (``WTQ: Dataset=...; Samples=13,706``).  Match that ordered
    # sequence line-locally; never search the whole multi-row table.
    for context in context_list:
        for line in str(context or "").splitlines():
            for line_form in _context_forms(line):
                if any(_contains_ordered_fact_tokens(line_form, surface) for surface in surfaces):
                    return True
    return False


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
