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
TABLE_QUESTION_RE = re.compile(
    r"\btable\b|表\s*\d*|数值|多少|样本量|比率|\bn\s*=",
    re.IGNORECASE,
)
ENTITY_RE = re.compile(r"[A-Za-z0-9_*+\-]+(?:[-\s][A-Za-z0-9_*+\-]+)*")

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
}


@dataclass
class Chunk:
    """A small serializable replacement for a LangChain Document."""

    page_content: str
    metadata: dict[str, Any] = field(default_factory=dict)


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
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text, flags=re.DOTALL)
    text = text.replace("∗", "*").replace("﹡", "*").replace("＊", "*")
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[_~`]+", "", text)
    text = re.sub(r"\s*\*\s*", "*", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().casefold()


def display_table_cell(value: Any) -> str:
    """Remove presentation markup while preserving the cell's value."""

    text = html.unescape(str(value or ""))
    text = re.sub(r"<[^>]*>", "", text)
    text = re.sub(r"[*_~`]+", "", text)
    text = text.replace("∗", "*").replace("﹡", "*").replace("＊", "*")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _caption_near(lines: list[str], header_idx: int) -> tuple[str, int | None]:
    for candidate_idx in range(header_idx - 1, max(-1, header_idx - 4), -1):
        candidate = lines[candidate_idx].strip()
        if candidate and TABLE_NUMBER_RE.search(candidate):
            return candidate, candidate_idx
    return "未命名表格", None


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

        caption, caption_idx = _caption_near(lines, header_idx)
        start_idx = caption_idx if caption_idx is not None else header_idx
        consumed.update(range(max(0, start_idx), j))
        metadata = _table_metadata(source, caption, len(tables) + 1, base_metadata)
        metadata["headers"] = normalize_for_match(header_line)
        tables.append(
            Chunk(
                page_content="\n".join([header_line, lines[separator_idx]] + rows),
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

    # Prefer a literal header match.  This avoids mapping a question about a
    # longer header to a shorter, ambiguous alias.
    literal_matches = [
        (idx, header)
        for idx, header in enumerate(normalized_headers)
        if header and header in normalized_question
    ]
    if literal_matches:
        return max(literal_matches, key=lambda item: len(item[1]))

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


def find_table_cell_in_chunks(
    question: str,
    texts: list[str],
    metas: list[dict[str, Any]],
) -> tuple[int, dict[str, str]] | None:
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
