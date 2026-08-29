"""Side-effect-free document parsing, table normalization, and chunking.

The UI/runtime lives in :mod:`app`.  This module deliberately does not import
Gradio, ChromaDB, SentenceTransformers, OpenAI, or dotenv so that parsing and
retrieval-contract tests can run offline and without a model or API key.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import html
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable


TABLE_SEPARATOR_RE = re.compile(r"^\|(?:\s*:?-{3,}:?\s*\|)+$")
TABLE_NUMBER_RE = re.compile(r"\btable\s*(\d+)\b", re.IGNORECASE)
TABLE_CAPTION_RE = re.compile(r"^\s*table\s*(\d+)\b", re.IGNORECASE)
TABLE_QUESTION_RE = re.compile(
    r"\btables?\b|表\s*\d+|表格|(?:该|此|下|上|上述|以下)表|表(?:中|内|里|所示)",
    re.IGNORECASE,
)
TABLE_ROW_VALUE_QUESTION_RE = re.compile(
    r"数量|占比|比例|数值|值|values?|count|percentage|proportion|how\s+many",
    re.IGNORECASE,
)
TABLE_COMPARISON_QUESTION_RE = re.compile(
    r"相比|比较|各基线|基线模型|哪些指标|最优|最高|最低|优于|逊于|"
    r"compare|comparison|baselines?|which\s+metrics?|best|highest|lowest",
    re.IGNORECASE,
)
ENTITY_RE = re.compile(r"[A-Za-z0-9_*+\-]+(?:[-\s][A-Za-z0-9_*+\-]+)*")
HEADER_HINT_RE = re.compile(
    r"\b(?:model|dataset|setting|data|parameters?|time|metric|score|error|"
    r"accuracy|precision|recall|mse|f1(?:-score)?|meteor|rouge(?:-\d+)?|"
    r"berts?|l\s*2|h\s*1|pipe|ns\d*)\b",
    re.IGNORECASE,
)

EVIDENCE_ASCII_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_*+./-]*")
EVIDENCE_NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:[<>≤≥]=?\s*)?\d[\d,]*(?:\.\d+)?(?:\s*[%A-Za-z]+)?"
)
EVIDENCE_ENTITY_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:[A-Z]{2,}[A-Za-z0-9]*|[A-Z][A-Za-z0-9]*[-_][A-Za-z0-9_*.-]+|[A-Za-z0-9]+\*)(?![A-Za-z0-9])"
)

# These aliases are deliberately semantic rather than tied to one paper.  A
# user may ask for an English table header in Chinese, or abbreviate a header
# such as "Target property F1 score" to simply "F1 score".
TABLE_COLUMN_ALIASES = {
    "overall optimization score": (
        "overall optimization score",
        "overall optimization",
        "overall score",
        "整体优化得分",
        "整体优化分数",
        "总体优化得分",
    ),
    "target property f1 score": (
        "target property f1 score",
        "target property f1",
        "target-property f1",
        "f1 score",
        "目标属性 f1",
        "目标属性f1",
        "靶属性 f1",
        "靶属性f1",
    ),
    "fingerprint similarity": (
        "fingerprint similarity",
        "fingerprint sim",
        "指纹相似度",
    ),
    "reasoning lms score": (
        "reasoning lms score",
        "lms score",
        "推理 lms",
        "推理lms",
    ),
    "reasoning richness": (
        "reasoning richness",
        "richness",
        "推理丰富度",
    ),
    "target set": (
        "target set",
        "target sets",
        "靶点集合",
        "靶点集",
        "目标集合",
    ),
}


@dataclass
class Chunk:
    """A small serializable replacement for a LangChain Document."""

    page_content: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _restore_bold_boundaries(text: str) -> str:
    """Strip bold markers without joining adjacent alphanumeric tokens."""

    def replace(match: re.Match[str]) -> str:
        before = text[match.start() - 1] if match.start() else ""
        after = text[match.end()] if match.end() < len(text) else ""
        left = " " if before.isalnum() else ""
        right = " " if after.isalnum() else ""
        return f"{left}{match.group(1)}{right}"

    return re.sub(r"\*\*(.*?)\*\*", replace, text, flags=re.DOTALL)


def file_sha256(file_path: str | Path) -> str:
    """Return a stable content hash without loading the whole file in memory."""

    digest = hashlib.sha256()
    with open(file_path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_for_match(value: Any) -> str:
    """Normalize Markdown/HTML/Unicode decoration for entity comparisons.

    A single asterisk is meaningful (for example ``DrugR*``), while paired
    asterisks used by Markdown bold are decoration.  Superscript stars are
    normalized to the same ASCII representation.
    """

    text = html.unescape(str(value or ""))
    text = re.sub(r"<[^>]*>", " ", text)
    # Keep a token boundary when bold markup is adjacent to another token
    # (``**Baseline**MgNO`` is emitted by some PDF table converters).
    text = _restore_bold_boundaries(text)
    text = text.replace("∗", "*").replace("﹡", "*").replace("＊", "*")
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[_~`]+", "", text)
    text = re.sub(r"\s*\*\s*", "*", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().casefold()


def _evidence_query_tokens(question: str) -> set[str]:
    """Return stable ASCII tokens useful for matching evidence lines."""

    tokens = set()
    for token in EVIDENCE_ASCII_TOKEN_RE.findall(normalize_for_match(question)):
        token = token.strip("._/-")
        if len(token) >= 3 and not token.isdigit():
            tokens.add(token)
    return tokens


def build_evidence_ledger(
    question: str,
    texts: Iterable[str],
    metadatas: Iterable[dict[str, Any]] | None = None,
    *,
    max_lines: int = 32,
) -> list[str]:
    """Extract a compact, deterministic checklist from retrieved evidence.

    The ledger is only an index of literal lines already present in the
    retrieved chunks.  It intentionally does not infer facts, normalize
    values, or replace the full context.  Lines containing question terms are
    preferred; numeric and scientific-entity lines from the same relevant
    chunks are retained so thresholds, tool names, and dataset sizes are not
    silently lost during generation.
    """

    if max_lines <= 0:
        return []
    text_list = [str(text or "") for text in texts]
    metadata_list = list(metadatas or [])
    query_tokens = _evidence_query_tokens(question)
    candidates: list[tuple[int, int, int, str]] = []
    context_has_query: list[bool] = []

    for context_index, text in enumerate(text_list):
        lines = [line.strip() for line in re.split(r"\n+", text) if line.strip()]
        line_data: list[tuple[int, int, int, str]] = []
        context_query = False
        for line_index, line in enumerate(lines):
            normalized_line = normalize_for_match(line)
            query_hits = sum(1 for token in query_tokens if token in normalized_line)
            number_hits = len(EVIDENCE_NUMBER_RE.findall(line))
            entity_hits = len(EVIDENCE_ENTITY_RE.findall(line))
            if query_hits:
                context_query = True
            if query_hits or number_hits or entity_hits:
                score = query_hits * 4 + number_hits * 2 + entity_hits
                line_data.append((score, context_index, line_index, line))
        context_has_query.append(context_query)
        candidates.extend(line_data)

    # Prefer salient lines from contexts that contain at least one question
    # term.  Then fill any remaining budget from other retrieved contexts. The
    # fallback is important for translated questions: a Chinese question may
    # have no literal overlap with an English sentence that contains the
    # threshold or tool name needed for a complete answer.
    preferred = [row for row in candidates if context_has_query[row[1]]]
    preferred.sort(key=lambda row: (-row[0], row[1], row[2]))
    remaining = [row for row in candidates if not context_has_query[row[1]]]
    remaining.sort(key=lambda row: (-row[0], row[1], row[2]))
    ranked = preferred + remaining

    # Reserve up to two salient lines for every retrieved context before
    # filling the remaining budget by relevance.  A pure global sort can
    # otherwise spend the entire budget on duplicate table rows from one
    # chunk and silently drop a threshold/tool line that lives in a sibling
    # chunk of the same section.
    selected_rows: list[tuple[int, int, int, str]] = []
    selected_keys: set[tuple[int, int]] = set()
    for context_index in range(len(text_list)):
        context_rows = [row for row in ranked if row[1] == context_index]
        if not context_rows or len(selected_rows) >= max_lines:
            continue
        for row in context_rows[:2]:
            if len(selected_rows) >= max_lines:
                break
            selected_rows.append(row)
            selected_keys.add((row[1], row[2]))
    for row in ranked:
        if len(selected_rows) >= max_lines:
            break
        key = (row[1], row[2])
        if key in selected_keys:
            continue
        selected_rows.append(row)
        selected_keys.add(key)
    selected = {(row[1], row[2]): row for row in selected_rows}

    def render(row: tuple[int, int, int, str]) -> str:
        _, context_index, _, line = row
        metadata = metadata_list[context_index] if context_index < len(metadata_list) else {}
        source = metadata.get("source") if isinstance(metadata, dict) else None
        header = metadata.get("headers") if isinstance(metadata, dict) else None
        section = str(header).split(">")[-1].strip() if header else ""
        suffix = f"，{source}" if source else ""
        if section:
            suffix += f"，{section}"
        return f"【片段 {context_index + 1}{suffix}】{line}"

    return [
        render(selected[key])
        for key in sorted(selected, key=lambda value: (value[0], value[1]))
    ]


def supplement_answer_with_evidence(
    answer: str,
    question: str,
    ledger: Iterable[str],
    *,
    max_lines: int = 2,
) -> str:
    """Append literal high-signal evidence when a composite answer omits it.

    This is a conservative, model-free safeguard.  It only appends lines that
    already appeared in the ledger, contain at least two high-signal numeric
    or scientific markers, and are not fully represented in the generated
    answer.  The original answer is never rewritten, and the supplement is
    visibly labeled as a quotation for manual review rather than presented as
    an inference.
    """

    answer_text = str(answer or "").strip()
    if not answer_text or max_lines <= 0:
        return answer_text
    question_normalized = normalize_for_match(question)
    if not any(
        cue in question_normalized
        for cue in ("多少", "样本", "数量", "dataset", "pipeline", "管道", "阈值", "threshold")
    ):
        return answer_text
    answer_normalized = normalize_for_match(answer_text)
    query_tokens = _evidence_query_tokens(question)
    section_aliases = {
        "数据集": ("dataset", "data"),
        "显式推理": ("explicit-reasoning", "reasoning"),
        "推理": ("reasoning",),
        "强化学习": ("reinforcement", "rl"),
        "训练": ("training", "train"),
        "奖励": ("reward",),
        "管道": ("pipeline",),
    }
    for phrase, aliases in section_aliases.items():
        if phrase in normalize_for_match(question):
            query_tokens.update(aliases)
    rows: list[tuple[int, int, int, str, list[str]]] = []
    max_section_hits = 0
    for line_index, raw_line in enumerate(ledger):
        line = str(raw_line or "").strip()
        if not line:
            continue
        content = line.split("】", 1)[-1].strip()
        prefix = line.split("】", 1)[0]
        section = prefix.rsplit("，", 1)[-1] if "，" in prefix else ""
        section_hits = sum(1 for token in query_tokens if token in normalize_for_match(section))
        max_section_hits = max(max_section_hits, section_hits)
        if content.count("|") >= 2:
            continue
        # Skip heading/caption lines and bare hyphenated ordinals such as
        # ``COX-1/COX-2``.  They are useful entity text in the ledger but are
        # not reliable omitted quantities or thresholds for an answer
        # supplement.
        if content.startswith("#") or re.match(r"^\s*\*{0,2}table\s+\d+\b", content, re.IGNORECASE):
            continue
        if re.search(
            r"\btrain/dev/test\b|dataset statistics|statistical characterization|"
            r"token-level statistics|held[- ]?out",
            content,
            flags=re.IGNORECASE,
        ):
            continue
        numbers = [
            value
            for value in EVIDENCE_NUMBER_RE.findall(content)
            if not re.search(r",\s*[A-Za-z]", value)
            and (
                re.search(r"\d+\.\d+", value)
                or re.search(r"[%<>≤≥]", value)
                or len(re.sub(r"\D", "", value)) >= 2
            )
        ]
        entities = EVIDENCE_ENTITY_RE.findall(content)
        high_signal_entities = [
            entity
            for entity in entities
            if (
                len(re.sub(r"[^A-Za-z0-9]", "", entity)) >= 5
                and (
                    re.search(r"[A-Z0-9]", re.sub(r"[^A-Za-z0-9]", "", entity)[1:])
                    or re.sub(r"[^A-Za-z0-9]", "", entity).isupper()
                )
            )
            or (
                re.search(r"[-_*]", entity)
                and re.search(r"[A-Z0-9]", entity[1:])
            )
        ]
        number_hits = len(numbers)
        markers = [*numbers, *high_signal_entities]
        # A supplement is a last-resort check for omitted quantities,
        # thresholds, or named tools/models.  Entity-only prose is admitted
        # only when it contains at least two high-signal names, which keeps a
        # generic category list from being appended merely because it shares
        # a section heading with the question.
        if len(markers) < 2:
            continue
        missing = []
        for marker in markers:
            marker_normalized = normalize_for_match(marker)
            if marker_normalized in answer_normalized:
                continue
            numeric_match = re.match(
                r"[<>≤≥]?\s*\d[\d,]*(?:\.\d+)?",
                str(marker),
            )
            if numeric_match and normalize_for_match(numeric_match.group(0)) in answer_normalized:
                continue
            missing.append(marker)
        if not missing:
            continue
        query_hits = sum(1 for token in query_tokens if token in normalize_for_match(content))
        score = (
            number_hits * 10
            + query_hits * 4
            + section_hits * 6
            + len(high_signal_entities)
            - line_index * 0.001
        )
        rows.append((score, section_hits, line_index, line, missing))

    if not rows:
        return answer_text
    if max_section_hits:
        rows = [row for row in rows if row[1] == max_section_hits]
    if not rows:
        return answer_text
    rows.sort(key=lambda row: (-row[0], row[2]))
    selected = [row[3] for row in rows[:max_lines]]
    supplement = "\n\n【补充原文核对项】\n" + "\n".join(f"- {line}" for line in selected)
    return answer_text + supplement


def display_table_cell(value: Any) -> str:
    """Remove presentation markup while preserving the cell's value."""

    text = html.unescape(str(value or ""))
    text = re.sub(r"<[^>]*>", "", text)
    text = _restore_bold_boundaries(text)
    text = re.sub(r"[*_~`]+", "", text)
    text = text.replace("∗", "*").replace("﹡", "*").replace("＊", "*")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _caption_text(line: str) -> str:
    """Strip presentation markup before deciding whether a line is a caption."""

    text = html.unescape(str(line or ""))
    text = re.sub(r"<[^>]*>", "", text)
    text = re.sub(r"[*_~`]+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _is_table_caption(line: str) -> bool:
    return bool(TABLE_CAPTION_RE.match(_caption_text(line)))


def _caption_near(
    lines: list[str],
    header_idx: int,
    table_end_idx: int,
) -> tuple[str, int | None]:
    """Find a caption immediately before or after a Markdown table.

    ``pymupdf4llm`` commonly emits captions after the table, while LaTeX
    exports often place them before it.  Only blank lines may occur between a
    table and its caption; this prevents a prose cross-reference to another
    table from being attached accidentally.
    """

    for direction, start in ((-1, header_idx - 1), (1, table_end_idx)):
        candidate_idx = start
        blank_lines = 0
        while 0 <= candidate_idx < len(lines) and not lines[candidate_idx].strip():
            blank_lines += 1
            if blank_lines > 3:
                break
            candidate_idx += direction
        if 0 <= candidate_idx < len(lines) and _is_table_caption(lines[candidate_idx]):
            return lines[candidate_idx].strip(), candidate_idx
    return "未命名表格", None


def _clean_header_cell(value: Any) -> str:
    """Normalize a header cell while retaining readable metric names."""

    text = html.unescape(str(value or ""))
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)
    return display_table_cell(text)


def _looks_numeric_cell(value: Any) -> bool:
    text = display_table_cell(value)
    if not text:
        return False
    if text.casefold() in {"n/a", "na", "—", "-"}:
        return True
    return bool(re.fullmatch(r"[\s0-9.,+\-−×x*/()^<>eE]+", text))


def _looks_like_header_row(row: str) -> bool:
    """Recognize a second header row emitted below a spanning header."""

    cells = _split_markdown_row(row)
    if len(cells) < 2:
        return False
    hints = sum(bool(HEADER_HINT_RE.search(_clean_header_cell(cell))) for cell in cells)
    numeric = sum(_looks_numeric_cell(cell) for cell in cells)
    return hints >= 2 and numeric <= len(cells) // 2


def _looks_like_group_header_row(row: str) -> bool:
    """Require visible spanning/wrapping evidence before merging headers."""

    cells = [_clean_header_cell(cell) for cell in _split_markdown_row(row)]
    if not cells:
        return False
    if any(not cell for cell in cells):
        return True
    if any(left == right for left, right in zip(cells, cells[1:]) if left):
        return True
    return any(
        right and right[0].islower() and left and not _looks_numeric_cell(left)
        for left, right in zip(cells, cells[1:])
    )


def _join_markdown_cells(cells: list[str]) -> str:
    return "|" + "|".join(cells) + "|"


def _combine_header_rows(group_line: str, header_line: str) -> str:
    """Combine a spanning header row with the concrete metric header row."""

    group_cells = _split_markdown_row(group_line)
    metric_cells = _split_markdown_row(header_line)
    if not group_cells or len(group_cells) != len(metric_cells):
        return header_line
    clean_groups = [_clean_header_cell(cell) for cell in group_cells]
    clean_metrics = [_clean_header_cell(cell) for cell in metric_cells]
    if not any(clean_groups):
        return header_line

    spans: list[list[Any]] = []
    index = 0
    while index < len(clean_groups):
        label = clean_groups[index]
        if not label:
            index += 1
            continue
        end = index
        while end + 1 < len(clean_groups):
            next_label = clean_groups[end + 1]
            if not next_label or not next_label[0].islower():
                break
            # PDF line wrapping can split a word across adjacent cells (for
            # example ``Darcy s`` + ``mooth``).  Join a one-letter trailing
            # fragment directly; normal multi-word group labels retain a
            # separating space (``Darcy`` + ``rough``).
            last_token = label.rsplit(" ", 1)[-1]
            joiner = "" if len(last_token) == 1 and last_token.isalpha() else " "
            label = f"{label}{joiner}{next_label}"
            end += 1
        spans.append([index, end, label])
        index = end + 1

    # Some PDF table exporters place a final group label in the last cell of
    # a two-column metric group, leaving the preceding cell empty.  If that
    # pair repeats earlier in the header, move the label back one column.
    if spans:
        start, end, label = spans[-1]
        if start > 0 and not clean_groups[start - 1]:
            pair = tuple(normalize_for_match(value) for value in clean_metrics[start - 1 : start + 1])
            if len(pair) == 2 and any(
                tuple(normalize_for_match(value) for value in clean_metrics[pos : pos + 2]) == pair
                for pos in range(max(0, start - 2))
            ):
                spans[-1][0] = start - 1

    group_for_column = [""] * len(clean_metrics)
    for span_index, (start, _end, label) in enumerate(spans):
        next_start = spans[span_index + 1][0] if span_index + 1 < len(spans) else len(clean_metrics)
        for column in range(start, next_start):
            group_for_column[column] = label

    combined = []
    for group, metric in zip(group_for_column, clean_metrics):
        combined.append(f"{group} {metric}".strip() if group and metric else group or metric)
    return _join_markdown_cells(combined)


def _unit_header(value: Any) -> bool:
    text = _clean_header_cell(value)
    return "10" in text and any(symbol in text for symbol in ("×", "x", "^", "−", "-"))


def _canonicalize_unit_column(header_line: str, rows: list[str]) -> tuple[str, list[str]]:
    """Merge a standalone unit cell when data rows contain a blank column."""

    headers = _split_markdown_row(header_line)
    if not headers:
        return header_line, rows
    row_cells = [_split_markdown_row(row) for row in rows]
    for index in range(len(headers) - 1):
        if not _unit_header(headers[index + 1]):
            continue
        if not row_cells or not all(index < len(cells) and not display_table_cell(cells[index]) for cells in row_cells):
            continue
        headers = headers[:index] + [f"{_clean_header_cell(headers[index])} {_clean_header_cell(headers[index + 1])}"] + headers[index + 2 :]
        row_cells = [cells[:index] + cells[index + 1 :] for cells in row_cells]
        rows = [_join_markdown_cells(cells) for cells in row_cells]
        break
    return _join_markdown_cells(headers), rows


def _looks_like_layout_table(header_line: str, rows: list[str], caption_idx: int | None) -> bool:
    """Avoid indexing one-row URL/metadata layout tables as scientific tables."""

    if caption_idx is not None:
        return False
    cells = _split_markdown_row(header_line)
    cells.extend(cell for row in rows for cell in _split_markdown_row(row))
    return bool(cells) and any("http://" in cell or "https://" in cell for cell in cells) and not any(
        _looks_numeric_cell(cell) for cell in cells
    )


def _table_metadata(
    source: str,
    caption: str,
    ordinal: int,
    base_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    number_match = TABLE_NUMBER_RE.search(caption)
    number = number_match.group(1) if number_match else None
    table_id = f"table-{number}" if number else f"table-unnamed-{ordinal}"
    metadata = dict(base_metadata or {})
    metadata.update(
        {
            "source": source,
            "type": "table",
            "table_id": table_id,
            "table_number": int(number) if number else None,
            "table_caption": caption,
        }
    )
    return metadata


def _parse_gfm_tables(
    markdown_text: str,
    source: str,
    base_metadata: dict[str, Any] | None = None,
) -> tuple[list[Chunk], set[int]]:
    lines = markdown_text.splitlines()
    tables: list[Chunk] = []
    consumed: set[int] = set()
    i = 0
    while i < len(lines):
        if not TABLE_SEPARATOR_RE.fullmatch(lines[i].strip()):
            i += 1
            continue

        separator_idx = i
        header_idx = i - 1
        header_line = ""
        if header_idx >= 0:
            previous = lines[header_idx].strip()
            if previous.startswith("|") and previous.endswith("|"):
                header_line = lines[header_idx]
            else:
                header_idx = separator_idx

        rows: list[str] = []
        j = separator_idx + 1
        while j < len(lines):
            candidate = lines[j].strip()
            if not (candidate.startswith("|") and candidate.endswith("|")):
                break
            if TABLE_SEPARATOR_RE.fullmatch(candidate):
                break
            rows.append(lines[j])
            j += 1
        if not rows:
            i += 1
            continue

        data_rows = rows
        if (
            header_line
            and rows
            and _looks_like_group_header_row(header_line)
            and _looks_like_header_row(rows[0])
        ):
            header_line = _combine_header_rows(header_line, rows[0])
            data_rows = rows[1:]
        header_line, data_rows = _canonicalize_unit_column(header_line, data_rows)
        caption, caption_idx = _caption_near(lines, header_idx, j)
        if _looks_like_layout_table(header_line, data_rows, caption_idx):
            i = j
            continue
        start_idx = caption_idx if caption_idx is not None else header_idx
        end_idx = max(j, (caption_idx + 1) if caption_idx is not None else j)
        consumed.update(range(max(0, start_idx), end_idx))
        metadata = _table_metadata(source, caption, len(tables) + 1, base_metadata)
        metadata["headers"] = normalize_for_match(header_line)
        tables.append(
            Chunk(
                page_content="\n".join([header_line, lines[separator_idx]] + data_rows),
                metadata=metadata,
            )
        )
        i = j
    return tables, consumed


def _parse_fallback_tables(
    markdown_text: str,
    source: str,
    base_metadata: dict[str, Any] | None,
    existing: Iterable[Chunk],
) -> tuple[list[Chunk], set[int]]:
    """Extract paragraph-style ``Table N`` blocks when no GFM table exists."""

    existing_content = {chunk.page_content.strip() for chunk in existing}
    tables: list[Chunk] = []
    consumed: set[int] = set()
    lines = markdown_text.splitlines()
    pattern = re.compile(
        r"^\s*(?:\*\*)?Table\s*(\d+)\b.*?(?=\n\s*\n|\Z)",
        re.IGNORECASE | re.DOTALL | re.MULTILINE,
    )
    for match in pattern.finditer(markdown_text):
        block = match.group(0).strip()
        if not block or block in existing_content:
            continue
        # A prose sentence such as "Table 1 also presents ..." is a
        # cross-reference, not a table.  Require visible tabular delimiters
        # before creating a fallback table chunk; otherwise it can acquire a
        # misleading table number and hijack a later Table N query.
        block_lines = block.splitlines()
        if len(block_lines) < 2 or not any("|" in line for line in block_lines[1:]):
            continue
        start_line = markdown_text[: match.start()].count("\n")
        end_line = start_line + block.count("\n") + 1
        consumed.update(range(start_line, min(len(lines), end_line)))
        caption = block.splitlines()[0].strip()
        tables.append(
            Chunk(
                page_content=block,
                metadata=_table_metadata(
                    source, caption, len(tables) + len(list(existing)) + 1, base_metadata
                ),
            )
        )
        existing_content.add(block)
    return tables, consumed


def extract_tables(
    markdown_text: str,
    source: str,
    base_metadata: dict[str, Any] | None = None,
) -> tuple[list[Chunk], str]:
    """Return canonical table chunks and text with table spans removed.

    Removing table spans from the text stream prevents the old failure mode in
    which every table existed once as a normal text chunk and once as a table
    chunk.  The returned text can therefore be independently header-split.
    """

    tables, consumed = _parse_gfm_tables(markdown_text, source, base_metadata)
    if not tables:
        fallback, fallback_consumed = _parse_fallback_tables(
            markdown_text, source, base_metadata, tables
        )
        tables.extend(fallback)
        consumed.update(fallback_consumed)

    body_lines = ["" if idx in consumed else line for idx, line in enumerate(markdown_text.splitlines())]
    body = "\n".join(body_lines)
    return tables, body


def _split_text_chunks(
    content: str,
    metadata: dict[str, Any],
    chunk_size: int = 1024,
    chunk_overlap: int = 128,
) -> list[Chunk]:
    """Use LangChain splitters lazily; importing this module stays side-effect-free."""

    from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "H1"), ("##", "H2"), ("###", "H3"), ("####", "H4")],
        strip_headers=False,
    )
    fallback_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", "。", "；"],
    )
    chunks: list[Chunk] = []
    for header_chunk in header_splitter.split_text(content):
        header_meta = dict(metadata)
        header_parts = [
            f"{level}: {header_chunk.metadata[level]}"
            for level in ("H1", "H2", "H3", "H4")
            if level in header_chunk.metadata
        ]
        if header_parts:
            header_meta["headers"] = " > ".join(header_parts)
        text = header_chunk.page_content
        if len(text) > chunk_size:
            chunks.extend(
                Chunk(page_content=part, metadata=dict(header_meta))
                for part in fallback_splitter.split_text(text)
            )
        elif text.strip():
            chunks.append(Chunk(page_content=text, metadata=header_meta))
    return chunks


def split_to_chunks(
    documents: Iterable[Chunk],
    source: str,
    chunk_size: int = 1024,
    chunk_overlap: int = 128,
) -> list[Chunk]:
    """Split documents while keeping each table whole and canonical."""

    final_chunks: list[Chunk] = []
    for document in documents:
        base_metadata = dict(document.metadata)
        base_metadata["source"] = source
        tables, body = extract_tables(document.page_content, source, base_metadata)
        if body.strip():
            final_chunks.extend(
                _split_text_chunks(body, {**base_metadata, "type": "text"}, chunk_size, chunk_overlap)
            )
        final_chunks.extend(tables)
    return final_chunks


def extract_query_entities(question: str) -> list[str]:
    """Return candidate ASCII entities while excluding the table number."""

    candidates = []
    for entity in ENTITY_RE.findall(question):
        if re.fullmatch(r"table\s*\d+", entity, re.IGNORECASE) or entity.isdigit():
            continue
        candidates.append(entity)
    return candidates


def _table_row_label(row: str) -> str:
    cells = row.strip().strip("|").split("|")
    return cells[0] if cells else row


def select_row_entity(question: str, table_content: str) -> str:
    """Choose the question entity that actually appears in a table's first column."""

    data_rows = []
    lines = table_content.splitlines()
    separator_idx = next(
        (idx for idx, line in enumerate(lines) if TABLE_SEPARATOR_RE.fullmatch(line.strip())),
        0,
    )
    for row in lines[separator_idx + 1 :]:
        if row.strip().startswith("|") and row.strip().endswith("|"):
            data_rows.append(_table_row_label(row))
    normalized_labels = [normalize_for_match(label) for label in data_rows]
    candidates = sorted(extract_query_entities(question), key=len, reverse=True)
    for candidate in candidates:
        normalized = normalize_for_match(candidate)
        if normalized and any(
            normalized == label or normalized in label or label in normalized
            for label in normalized_labels
        ):
            return candidate
    return ""


def _split_markdown_row(row: str) -> list[str]:
    """Split a simple GFM row without treating escaped pipes as delimiters."""

    text = row.strip()
    if text.startswith("|"):
        text = text[1:]
    if text.endswith("|") and not text.endswith("\\|"):
        text = text[:-1]
    cells: list[str] = []
    current: list[str] = []
    escaped = False
    for char in text:
        if char == "|" and not escaped:
            cells.append("".join(current).strip())
            current = []
            continue
        if char == "\\" and not escaped:
            escaped = True
            current.append(char)
            continue
        escaped = False
        current.append(char)
    cells.append("".join(current).strip())
    return cells


def parse_markdown_table(content: str) -> tuple[list[str], list[list[str]]] | None:
    """Parse a canonical Markdown table into headers and data rows."""

    lines = content.splitlines()
    separator_idx = next(
        (idx for idx, line in enumerate(lines) if TABLE_SEPARATOR_RE.fullmatch(line.strip())),
        None,
    )
    if separator_idx is None or separator_idx == 0:
        return None
    header = _split_markdown_row(lines[separator_idx - 1])
    if not header:
        return None
    rows = []
    for line in lines[separator_idx + 1 :]:
        stripped = line.strip()
        if not (stripped.startswith("|") and stripped.endswith("|")):
            continue
        row = _split_markdown_row(line)
        if row:
            rows.append(row)
    return header, rows


def table_number_from_metadata(metadata: dict[str, Any]) -> int | None:
    """Read a table number from new or legacy Chroma metadata."""

    value = metadata.get("table_number")
    try:
        if value is not None:
            return int(value)
    except (TypeError, ValueError):
        pass
    caption_match = TABLE_NUMBER_RE.search(str(metadata.get("table_caption", "")))
    return int(caption_match.group(1)) if caption_match else None


def matching_table_indices(
    question: str,
    texts: list[str],
    metas: list[dict[str, Any]],
) -> list[int]:
    """Return canonical table chunks satisfying an explicit Table N filter."""

    table_indices = [idx for idx, meta in enumerate(metas) if meta.get("type") == "table"]
    table_number = table_number_from_question(question)
    if table_number is None:
        return table_indices
    return [
        idx
        for idx in table_indices
        if table_number_from_metadata(metas[idx]) == int(table_number)
    ]


def _match_table_column(question: str, headers: list[str]) -> tuple[int, str] | None:
    """Find the header requested by a question, including common aliases."""

    normalized_question = normalize_for_match(question)
    normalized_headers = [normalize_for_match(header) for header in headers]

    def without_unit(header: str) -> str:
        # Headers such as ``L2 Error (×10−2)`` are commonly asked about as
        # simply ``L2 Error``.  Keep the original header for display while
        # matching both forms against the question.
        return re.sub(r"\s*[\(\[][^)\]]*10[^)\]]*[\)\]]", "", header).strip()

    # Prefer a literal header match.  This avoids mapping a question about a
    # longer header to a shorter, ambiguous alias.
    literal_matches: list[tuple[int, str, int]] = []
    for idx, header in enumerate(normalized_headers):
        for candidate in {header, without_unit(header)}:
            if candidate and candidate in normalized_question:
                literal_matches.append((idx, header, len(candidate)))
    if literal_matches:
        idx, header, _ = max(literal_matches, key=lambda item: item[2])
        return idx, header

    alias_matches: list[tuple[int, str, int]] = []
    for idx, header in enumerate(normalized_headers):
        for canonical, aliases in TABLE_COLUMN_ALIASES.items():
            if header == canonical or header in canonical or canonical in header:
                for alias in aliases:
                    normalized_alias = normalize_for_match(alias)
                    if normalized_alias and normalized_alias in normalized_question:
                        alias_matches.append((idx, header, len(normalized_alias)))
    if alias_matches:
        idx, header, _ = max(alias_matches, key=lambda item: item[2])
        return idx, header
    return None


def extract_table_cell(
    question: str,
    table_content: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, str] | None:
    """Extract one row/column value when a question names both explicitly.

    The function is table-agnostic: it derives the row and column from the
    supplied Markdown table and only uses aliases for translating common
    natural-language header variants.
    """

    parsed = parse_markdown_table(table_content)
    if parsed is None:
        return None
    headers, rows = parsed
    column_match = _match_table_column(question, headers)
    if column_match is None:
        return None
    column_idx, column_header = column_match
    entity = select_row_entity(question, table_content)
    if not entity:
        return None
    target = normalize_for_match(entity)
    for row in rows:
        if not row:
            continue
        row_label = normalize_for_match(row[0])
        if not (target == row_label or target in row_label or row_label in target):
            continue
        if column_idx >= len(row):
            return None
        number = table_number_from_metadata(metadata or {})
        return {
            "table_number": str(number) if number is not None else "",
            "row": display_table_cell(row[0]),
            "column": display_table_cell(headers[column_idx]),
            "value": display_table_cell(row[column_idx]),
        }
    return None


def extract_table_row_values(
    question: str,
    table_content: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Extract all value columns for an explicitly named table row.

    Some questions ask for a row's count and percentage together without
    repeating the table's English column headers (for example, "数量和占比").
    In that case a single-column matcher cannot answer deterministically.  We
    return the row's non-label cells, preserving their parsed header names and
    literal values; this remains table-agnostic and does not infer units.
    """

    if not TABLE_ROW_VALUE_QUESTION_RE.search(question or ""):
        return None
    parsed = parse_markdown_table(table_content)
    if parsed is None:
        return None
    headers, rows = parsed
    entity = select_row_entity(question, table_content)
    if not entity:
        return None
    target = normalize_for_match(entity)
    for row in rows:
        if not row:
            continue
        row_label = normalize_for_match(row[0])
        if not (target == row_label or target in row_label or row_label in target):
            continue
        values: list[dict[str, str]] = []
        for column_idx, value in enumerate(row[1:], start=1):
            if column_idx >= len(headers):
                break
            values.append(
                {
                    "column": display_table_cell(headers[column_idx]),
                    "value": display_table_cell(value),
                }
            )
        if not values:
            return None
        number = table_number_from_metadata(metadata or {})
        return {
            "table_number": str(number) if number is not None else "",
            "row": display_table_cell(row[0]),
            "values": values,
        }
    return None


def find_table_cell_in_chunks(
    question: str,
    texts: list[str],
    metas: list[dict[str, Any]],
) -> tuple[int, dict[str, Any]] | None:
    """Find a requested table cell without crossing an explicit table boundary."""

    # Without an explicit table number, the same row/column may legitimately
    # occur in several tables.  Leave that ambiguous case to the normal
    # retrieval/generation path instead of silently choosing the first table.
    if table_number_from_question(question) is None:
        return None
    for idx in matching_table_indices(question, texts, metas):
        cell = extract_table_cell(question, texts[idx], metas[idx])
        if cell is not None:
            return idx, cell
        row_values = extract_table_row_values(question, texts[idx], metas[idx])
        if row_values is not None:
            return idx, row_values
    return None


def filter_table_rows_by_entity(content: str, entity: str) -> str | None:
    """Keep the header/separator and rows whose first cell matches the entity."""

    lines = content.splitlines()
    separator_idx = next(
        (idx for idx, line in enumerate(lines) if TABLE_SEPARATOR_RE.fullmatch(line.strip())),
        None,
    )
    if separator_idx is None:
        header, data = lines[:1], lines[1:]
    else:
        header, data = lines[: separator_idx + 1], lines[separator_idx + 1 :]
    target = normalize_for_match(entity)
    matched_rows = []
    for row in data:
        if not (row.strip().startswith("|") and row.strip().endswith("|")):
            continue
        label = normalize_for_match(_table_row_label(row))
        if target and (target == label or target in label or label in target):
            matched_rows.append(row)
    if not matched_rows:
        return None
    return "\n".join(header + matched_rows)


def is_table_question(question: str) -> bool:
    return bool(TABLE_QUESTION_RE.search(question or ""))


def is_comparative_table_question(question: str) -> bool:
    """Whether a table question needs multiple rows for comparison."""

    return bool(TABLE_COMPARISON_QUESTION_RE.search(question or ""))


def table_number_from_question(question: str) -> str | None:
    match = TABLE_NUMBER_RE.search(question or "")
    return match.group(1) if match else None


def _table_like_text(text: str) -> bool:
    lines = text.splitlines()
    return any(TABLE_SEPARATOR_RE.fullmatch(line.strip()) for line in lines)


def rerank_table_first(
    question: str,
    texts: list[str],
    metas: list[dict[str, Any]],
) -> tuple[list[int], str, list[str]]:
    """Return deterministic table-aware order and filtered text copies.

    Explicit table questions exclude non-table chunks that still contain a
    legacy GFM table.  This protects users before an old Chroma collection has
    been rebuilt with canonical chunks.
    """

    working_texts = list(texts)
    if not is_table_question(question):
        return list(range(len(working_texts))), "", working_texts

    table_idx = matching_table_indices(question, working_texts, metas)
    table_num = table_number_from_question(question)
    note = ""
    if table_num is not None:
        if table_idx:
            # Comparative questions (e.g. "DrugR compared with each
            # baseline") need the complete table.  Row filtering is reserved
            # for deterministic single-row lookups so baseline evidence is
            # not discarded before generation.
            if not is_comparative_table_question(question):
                for idx in table_idx:
                    entity = select_row_entity(question, working_texts[idx])
                    if entity:
                        filtered = filter_table_rows_by_entity(working_texts[idx], entity)
                        if filtered is not None:
                            working_texts[idx] = filtered
        else:
            # Never compensate for a missing Table N with another table.  The
            # previous fallback made Table 1 look like an answer to a Table 2
            # question whenever the requested chunk was not retrieved.
            note = f"未找到 Table {table_num} 的结构化表格，不能用其他表格替代。"

    other_idx = []
    for idx, meta in enumerate(metas):
        if meta.get("type") == "table":
            continue
        if table_num is not None and _table_like_text(working_texts[idx]):
            continue
        other_idx.append(idx)
    return table_idx + other_idx, note, working_texts
