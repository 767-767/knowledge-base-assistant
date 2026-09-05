"""Side-effect-free document parsing, table normalization, and chunking.

The UI/runtime lives in :mod:`app`.  This module deliberately does not import
Gradio, ChromaDB, SentenceTransformers, OpenAI, or dotenv so that parsing and
retrieval-contract tests can run offline and without a model or API key.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
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
    r"\btables?\s*\d+\b|\b(?:the|this|following|above)\s+tables?\b|"
    r"表\s*\d+|(?:该|此|下|上|上述|以下)表(?:格)?|表(?:格)?(?:中|内|里|所示)",
    re.IGNORECASE,
)
TABLE_ROW_VALUE_QUESTION_RE = re.compile(
    r"数量|占比|比例|数值|值|得分|values?|scores?|count|percentage|proportion|how\s+many",
    re.IGNORECASE,
)
TABLE_COMPARISON_QUESTION_RE = re.compile(
    r"相比|比较|各基线|基线模型|哪些指标|最优|最高|最低|优于|逊于|"
    r"compare|comparison|baselines?|which\s+metrics?|best|highest|lowest",
    re.IGNORECASE,
)
DERIVED_VALUE_QUESTION_RE = re.compile(
    r"差(?:多少|值)|相差|高(?:出|于)?多少|低(?:出|于)?多少|多多少|少多少|"
    r"计算|合计|总和|总计|求和|相加|加总|算术平均|求平均|的平均值|"
    r"(?:比例|占比|百分比|百分之几)(?:[^。！？?]{0,24}(?:计算|保留(?:两|二|三|四|五|\d+)位))|"
    r"(?:计算|求)[^。！？?]{0,32}(?:比例|占比|百分比|百分之几)|"
    r"(?:占总数|占全部|占[^。！？?]{0,12}的)[^。！？?]{0,16}(?:比例|占比|百分比|百分之几)|"
    r"(?:比例|占比|百分比|百分之几)[^。！？?]{0,24}保留(?:两|二|三|四|五|\d+)位|"
    r"几成|倍数|多少倍|几倍|保留率[^。！？?]{0,32}保留(?:两|二|三|四|五|\d+)位|"
    r"相对(?:提升|提高|增长|增加|下降|降低)|绝对提升|提升百分比|降低百分比|增幅|降幅|"
    r"\b(?:calculate|compute|calculation|difference(?:\s+between)?|"
    r"how\s+much\s+(?:higher|lower|more|less)|sum|total|arithmetic\s+average|"
    r"mean\s+of|average\s+of|ratio\s+of|percentage\s+of|proportion\s+of|times|relative\s+"
    r"(?:increase|decrease)|increase\s+percentage|decrease\s+percentage)\b",
    re.IGNORECASE,
)
FIGURE_REFERENCE_RE = re.compile(
    r"(?P<extended>extended\s+data\s+)?(?:fig(?:ure)?\.?|图)\s*"
    r"(?P<number>[1-9]\d*)(?:[A-Za-z])?(?![\dA-Za-z])",
    re.IGNORECASE,
)
FIGURE_CAPTION_RE = re.compile(
    r"^\s*(?P<extended>extended\s+data\s+)?(?:fig(?:ure)?\.?|图)\s*"
    r"(?P<number>[1-9]\d*)(?![\dA-Za-z])",
    re.IGNORECASE,
)
FORMULA_QUESTION_RE = re.compile(
    r"公式|方程|表达式|形式化|equation|formula|expression|"
    r"(?:集合|组成).{0,20}\b[A-Z]\b|"
    r"(?:吸收|展开).{0,16}(?:节点|摘要|子节点)|\b(?:absorb|expand)\b|"
    r"(?:定义|如何定义).{0,30}(?:空间|子空间|集合|变量|符号|S_[A-Za-z])|"
    r"(?:token|logit).{0,30}(?:约束|修改|mask|valid)|"
    r"卷积核|kernel\s*(?:size|dimension|大小|尺寸)|"
    r"(?:形式化|数学|显式|明确).{0,20}(?:输出|变量|结果|定义|形式)|"
    r"(?:输出|变量|结果)\s*[A-Za-z]\b.{0,20}(?:定义|表达|形式)|"
    r"(?:output|variable|result)\s+[A-Za-z]\b.{0,24}(?:defined|definition|form|equation)|"
    r"(?:激活|非线性激活|activation|nonlinear).{0,24}(?:函数|function|operator|算子)?|"
    r"离散(?:后的?)?.{0,20}(?:系统|方程).{0,12}(?:形式|表达式|写成)|"
    r"(?:finite[- ]element|FEM).{0,30}(?:system|equation|kernel|form)|"
    r"(?:system|equation).{0,20}(?:form|written|expression)|"
    r"(?:初始化|残差|平滑迭代|更新).{0,32}(?:状态|量|过程|形式|如何|什么|更新|iteration|residual)|"
    r"(?:限制|延拓|restriction|prolongation).{0,48}(?:操作|算子|网格|层级|stride|循环|cycle|如何|改变)|"
    r"(?:循环|cycle).{0,24}(?:类型|区别|操作|名称|V-cycle|Backslash)|"
    r"(?:PDE|椭圆).{0,40}(?:区域|定义|domain|defined|边界条件|boundary)",
    re.IGNORECASE,
)
LIMITATION_QUESTION_RE = re.compile(
    r"限制|局限|不足|缺点|失败模式|挑战|"
    r"\blimitations?\b|\bdrawbacks?\b|\bfailure\s+modes?\b|"
    r"\b(?:what|which|how)\b.{0,36}\b(?:limitation|challenge|failure)\b",
    re.IGNORECASE,
)
LIMITATION_OPERATOR_RE = re.compile(
    r"(?:限制|restriction|prolongation).{0,48}(?:操作|算子|网格|层级|stride|循环|cycle|改变|"
    r"operator|grid|level|how\s+(?:does|do)|change)",
    re.IGNORECASE,
)
UNAMBIGUOUS_LIMITATION_RE = re.compile(
    r"局限|不足|缺点|失败模式|挑战|"
    r"\blimitations?\b|\bdrawbacks?\b|\bfailure\s+modes?\b",
    re.IGNORECASE,
)
FORMULA_EVIDENCE_RE = re.compile(
    r"(?:\\[A-Za-z]+|[×∗*]|[=≤≥]|∈|ℝ|ℤ|∫|∂|"
    r"\b(?:PDE|equation|formula|kernel|operator|convolution|residual|"
    r"finite[- ]element|FEM)\b)",
    re.IGNORECASE,
)
_LATEX_JSON_ESCAPE_REPAIRS = {
    "\n": (
        ("otin", r"\notin"),
        ("abla", r"\nabla"),
        ("eq", r"\neq"),
        ("leq", r"\nleq"),
        ("geq", r"\ngeq"),
        ("exists", r"\nexists"),
        ("subseteq", r"\nsubseteq"),
        ("rightarrow", r"\nrightarrow"),
        ("mid", r"\nmid"),
    ),
    "\t": (
        ("imes", r"\times"),
        ("ext", r"\text"),
        ("heta", r"\theta"),
        ("au", r"\tau"),
        ("op", r"\top"),
        ("ilde", r"\tilde"),
    ),
    "\r": (
        ("ightarrow", r"\rightarrow"),
        ("ho", r"\rho"),
        ("ight", r"\right"),
    ),
    "\b": (
        ("oldsymbol", r"\boldsymbol"),
        ("mathbf", r"\mathbf"),
        ("egin", r"\begin"),
        ("inom", r"\binom"),
        ("eta", r"\beta"),
    ),
    "\f": (
        ("orall", r"\forall"),
        ("rac", r"\frac"),
    ),
}
RAW_FORMULA_LINE_RE = re.compile(
    r"(?=.*[A-Za-zα-ωΑ-ΩΩ])(?=.*[=≤≥∗×∈])",
)
LIMITATION_EVIDENCE_RE = re.compile(
    r"\b(?:limitation|limited|static\s+structures?|dynamical\s+behavio(?:u)?r|"
    r"dynamic(?:al)?\s+(?:state|behaviour|behavior)|solution\s+ensemble|"
    r"conformation(?:al)?\s+coverage|failure(?:\s+modes?)?|cannot|unable|"
    r"open|closed|apo|holo)\b|"
    r"限制|局限|不足|失败|静态结构|动力学行为|溶液|构象|开放|闭合",
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

# The broader ``EVIDENCE_ENTITY_RE`` above is intentionally permissive for
# the generation supplement.  Runtime answer validation needs a narrower
# token pattern: it should recognize names such as ``DeepSeek-R1`` and
# ``ADMETLab`` without treating every sentence-initial word or citation year
# as a required fact.
VALIDATION_ENTITY_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:"
    r"[A-Z][A-Za-z0-9]*(?:[-_*][A-Za-z0-9_*]+)+|"
    r"[A-Z]{2,}[A-Za-z0-9]*|"
    r"[A-Za-z0-9]+\*"
    r")(?![A-Za-z0-9])"
)
VALIDATION_ENTITY_STOPWORDS = {
    "A",
    "An",
    "And",
    "As",
    "At",
    "By",
    "For",
    "From",
    "In",
    "It",
    "Of",
    "On",
    "Or",
    "The",
    "This",
    "To",
    "We",
    "With",
    "What",
    "When",
    "Where",
    "Which",
    "Who",
    "Why",
    "How",
}
VALIDATION_PROXIMITY_TERMS = {
    "dataset",
    "data",
    "sample",
    "samples",
    "question",
    "questions",
    "answer",
    "answers",
    "pair",
    "pairs",
    "count",
    "number",
    "size",
    "pipeline",
    "process",
    "construct",
    "construction",
    "annotation",
    "annotate",
    "label",
    "rationale",
    "reasoning",
    "algorithm",
    "method",
    "score",
    "metric",
    "threshold",
    "training",
    "train",
    "steps",
    "step",
    "iteration",
    "residual",
    "figure",
    "fig",
    "panel",
    "protein",
    "rna",
    "dna",
    "module",
    "block",
    "blocks",
    "layer",
    "layers",
    "complexity",
    "parameter",
    "parameters",
    "time",
    "error",
    "domain",
    "equation",
    "formula",
    "subsection",
    "subsections",
    "section",
    "retrieval",
    "retrieve",
    "ranking",
    "rank",
    "relevance",
    "split",
    "splitting",
    "chunk",
    "chunks",
    "baseline",
    "top",
    "comprises",
    "consists",
    "contains",
    "included",
    "total",
    "within",
    "after",
    "restriction",
    "prolongation",
    "multigrid",
    "cycle",
    "cycles",
    "optimizer",
    "benchmark",
    "benchmarks",
    "structure",
    "structures",
    "composed",
    "released",
    "success",
    "criterion",
}

# These aliases are deliberately semantic rather than tied to one paper.  A
# user may ask for an English table header in Chinese, or abbreviate a header
# such as "Target property F1 score" to simply "F1 score".
TABLE_COLUMN_ALIASES = {
    "all f1": (
        "all f1",
        "overall f1",
        "overall f1 score",
        "整体 f1",
        "整体 f1 score",
    ),
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
    "l2": (
        "l2",
        "l 2",
        "relative l2",
        "relative l 2",
    ),
    "h1": (
        "h1",
        "h 1",
        "relative h1",
        "relative h 1",
    ),
    "ft": (
        "ft",
        "full-text",
        "full text",
        "fulltext",
        "全文",
    ),
    "target set": (
        "target set",
        "target sets",
        "靶点集合",
        "靶点集",
        "目标集合",
    ),
    "tokens": ("tokens", "token", "token 数", "token数"),
    "time": ("time", "时间", "耗时", "推理时间"),
    "script": ("script",),
    "wikipedia": ("wikipedia",),
}


@dataclass
class Chunk:
    """A small serializable replacement for a LangChain Document."""

    page_content: str
    metadata: dict[str, Any] = field(default_factory=dict)


def figure_reference_from_question(question: str) -> tuple[str, int] | None:
    """Return the explicitly requested figure kind and number, if present.

    Normal figures and ``Extended Data Fig. N`` deliberately use different
    kinds. This prevents a query for a paper's main Figure 1 from silently
    selecting Extended Data Figure 1, or vice versa.
    """

    match = FIGURE_REFERENCE_RE.search(question or "")
    if match is None:
        return None
    kind = "extended_data_figure" if match.group("extended") else "figure"
    return kind, int(match.group("number"))


def figure_reference_from_metadata(metadata: dict[str, Any]) -> tuple[str, int] | None:
    """Read a normalized figure reference from canonical or legacy metadata."""

    value = metadata.get("figure_number")
    try:
        number = int(value) if value is not None else None
    except (TypeError, ValueError):
        number = None
    if number is None:
        caption_match = FIGURE_CAPTION_RE.search(str(metadata.get("figure_caption", "")))
        if caption_match is None:
            return None
        number = int(caption_match.group("number"))
        kind = (
            "extended_data_figure"
            if caption_match.group("extended")
            else "figure"
        )
        return kind, number
    kind = str(metadata.get("figure_kind", "figure"))
    if kind not in {"figure", "extended_data_figure"}:
        kind = "figure"
    return kind, number


def matching_figure_indices(
    question: str,
    texts: list[str],
    metas: list[dict[str, Any]],
) -> list[int]:
    """Return canonical figure chunks satisfying an explicit Figure N filter."""

    del texts  # Kept parallel with ``matching_table_indices`` for callers.
    figure_indices = [
        index
        for index, metadata in enumerate(metas)
        if metadata.get("type") == "figure"
    ]
    reference = figure_reference_from_question(question)
    if reference is None:
        return figure_indices
    return [
        index
        for index in figure_indices
        if figure_reference_from_metadata(metas[index]) == reference
    ]


def is_formula_question(question: str) -> bool:
    """Return whether a question explicitly asks for equation-like evidence."""

    return bool(FORMULA_QUESTION_RE.search(str(question or "")))


def missing_pdf_formula_blocks(
    markdown_text: str,
    pdf_text: str,
    *,
    context_lines: int = 1,
) -> list[str]:
    """Recover omitted equation lines as small, separate PDF-text evidence blocks.

    ``pymupdf4llm`` occasionally drops a displayed equation while retaining its
    surrounding prose.  This compares the raw PDF text layer with Markdown and
    returns only missing equation neighborhoods, leaving normal Markdown
    chunks untouched.
    """

    markdown_normalized = normalize_for_match(markdown_text)
    raw_lines = [" ".join(line.split()) for line in str(pdf_text or "").splitlines()]
    raw_lines = [line for line in raw_lines if line]
    spans: list[tuple[int, int]] = []
    for index, line in enumerate(raw_lines):
        normalized_line = normalize_for_match(line)
        if (
            len(line) > 500
            or not RAW_FORMULA_LINE_RE.search(line)
            or normalized_line in markdown_normalized
        ):
            continue
        spans.append(
            (max(0, index - context_lines), min(len(raw_lines), index + context_lines + 1))
        )

    blocks: list[str] = []
    for start, end in spans:
        if blocks and start <= previous_end:
            previous_end = max(previous_end, end)
            blocks[-1] = "\n".join(raw_lines[previous_start:previous_end])
            continue
        previous_start, previous_end = start, end
        blocks.append("\n".join(raw_lines[start:end]))
    return blocks


def is_limitation_question(question: str) -> bool:
    """Return whether a question explicitly asks about limitations or failures."""

    value = str(question or "")
    if not LIMITATION_QUESTION_RE.search(value):
        return False
    # Chinese “限制” is also the standard name of the multigrid restriction
    # operator. Do not attach limitation evidence to an explicit operator/grid
    # question unless an unambiguous failure/shortcoming cue is present.
    if LIMITATION_OPERATOR_RE.search(value) and not UNAMBIGUOUS_LIMITATION_RE.search(value):
        return False
    return True


def _formula_query_terms(question: str) -> set[str]:
    """Add generic bilingual terms used to rank formula-bearing passages."""

    normalized = normalize_for_match(question)
    terms = set(re.findall(r"[a-z][a-z0-9-]+", normalized))
    aliases = {
        "公式": ("equation", "formula", "system"),
        "形式化": ("formal", "formulation", "defined", "definition", "output"),
        "集合": ("set", "context", "components"),
        "组成": ("set", "context", "components"),
        "吸收": ("absorb", "summary", "current", "level"),
        "展开": ("expand", "child", "nodes"),
        "节点": ("node", "nodes"),
        "摘要": ("summary", "summaries"),
        "数学": ("mathematical", "defined", "definition", "output"),
        "显式": ("explicit", "defined", "definition", "output"),
        "定义": ("define", "defined", "definition", "form", "output"),
        "最终输出": ("final", "output", "result"),
        "激活": ("activation", "function", "GELU", "nonlinear"),
        "非线性": ("nonlinear", "activation", "function"),
        "方程": ("equation", "system", "PDE"),
        "PDE": ("PDE", "equation", "elliptic"),
        "卷积核": ("convolution", "kernel"),
        "卷积": ("convolution", "kernel"),
        "算子": ("operator", "kernel"),
        "尺寸": ("dimension", "dimensions", "size"),
        "大小": ("dimension", "dimensions", "size"),
        "离散系统": ("discretized", "system"),
        "有限元": ("finite", "element", "FEM"),
        "线性有限元": ("linear", "finite", "element", "FEM"),
        "椭圆": ("elliptic",),
        "区域": ("domain", "region", "omega"),
        "边界条件": ("boundary", "condition"),
        "残差": ("residual",),
        "初始化": ("initialize", "initialization"),
        "更新": ("update", "iteration", "residual"),
        "平滑迭代": ("smoothing", "iteration", "residual"),
        "限制": ("restriction", "restrict", "coarser", "stride"),
        "延拓": ("prolongation", "prolong", "transpose", "de-convolution"),
        "循环": ("cycle", "v-cycle", "backslash-cycle", "post-smoothing"),
    }
    for cue, cue_aliases in aliases.items():
        if cue.casefold() in normalized:
            terms.update(alias.casefold() for alias in cue_aliases)
    return terms


def _limitation_query_terms(question: str) -> set[str]:
    """Add generic bilingual terms used to rank limitation evidence."""

    normalized = normalize_for_match(question)
    terms = set(re.findall(r"[a-z][a-z0-9-]+", normalized))
    aliases = {
        "限制": (
            "limitation",
            "limited",
            "static",
            "structure",
            "structures",
            "dynamical",
            "dynamic",
            "behaviour",
            "behavior",
            "solution",
            "conformation",
            "conformational",
            "coverage",
        ),
        "局限": ("limitation", "limited", "challenge", "drawback"),
        "不足": ("limitation", "limited", "shortcoming", "lack"),
        "缺点": ("limitation", "drawback", "weakness"),
        "失败模式": ("failure", "mode", "modes", "incorrect", "error"),
        "挑战": ("challenge", "challenging", "limitation"),
        "分子动力学": ("molecular", "dynamics", "dynamical", "dynamic", "solution"),
        "动力学": ("dynamical", "dynamic", "behaviour", "behavior", "solution"),
        "状态": ("state", "states", "conformation", "conformational"),
        "构象": ("conformation", "conformational", "open", "closed", "coverage"),
        "示例": ("example",),
        "apo": ("apo", "open"),
        "holo": ("holo", "closed"),
    }
    for cue, cue_aliases in aliases.items():
        if cue.casefold() in normalized:
            terms.update(alias.casefold() for alias in cue_aliases)
    return terms


def formula_evidence_indices(
    question: str,
    texts: Iterable[str],
    metadatas: Iterable[dict[str, Any]] | None = None,
    *,
    max_results: int = 4,
    allowed_indices: Iterable[int] | None = None,
) -> list[int]:
    """Rank a small set of formula-bearing chunks for an explicit formula query.

    This is an opt-in retrieval aid, not a symbolic solver. It only scores
    literal PDF text blocks containing equation symbols or common equation
    terms and combines those markers with generic bilingual query terms. The
    caller controls the small result budget so this remains an opt-in aid.
    """

    if max_results <= 0 or not is_formula_question(question):
        return []
    text_list = [str(text or "") for text in texts]
    metadata_list = list(metadatas or [])
    query_terms = _formula_query_terms(question)
    query_symbols = {
        token.casefold()
        for token in re.findall(r"(?<![A-Za-z])[A-Za-z](?:_[A-Za-z]+)?(?![A-Za-z])", str(question or ""))
    }
    query_symbols.update(
        symbol.replace("_", "") for symbol in set(query_symbols) if "_" in symbol
    )
    allowed = (
        {int(index) for index in allowed_indices if int(index) >= 0}
        if allowed_indices is not None
        else None
    )
    rows: list[tuple[int, int]] = []
    for index, text in enumerate(text_list):
        if allowed is not None and index not in allowed:
            continue
        metadata = metadata_list[index] if index < len(metadata_list) else {}
        if isinstance(metadata, dict) and metadata.get("type") in {"figure", "image"}:
            continue
        header = str(metadata.get("headers", "")) if isinstance(metadata, dict) else ""
        if re.search(r"references?|bibliography|参考文献", header, re.IGNORECASE):
            continue
        marker_hits = len(FORMULA_EVIDENCE_RE.findall(text))
        if not marker_hits:
            continue
        normalized = normalize_for_match("\n".join((text, header)))
        query_hits = sum(1 for term in query_terms if term in normalized)
        symbol_hits = sum(
            1
            for symbol in query_symbols
            if re.search(rf"(?<![a-z]){re.escape(symbol)}(?![a-z])", normalized)
        )
        lhs_hits = sum(
            1
            for symbol in query_symbols
            if re.search(
                rf"(?:^|\n)\s*(?:\(\d+\)\s*)?{re.escape(symbol)}\s*=",
                normalized,
            )
        )
        # PDF-to-Markdown conversion can emit many spurious ``=``/``*``
        # markers. Require at least two semantic query terms unless a
        # recovered formula contains a symbol named in the question. Typed
        # formula chunks are higher-confidence than prose with stray markers.
        if query_hits < 2 and not (
            metadata.get("type") == "formula"
            and (symbol_hits or (is_formula_question(question) and marker_hits >= 2))
        ):
            continue
        score = min(marker_hits, 8) + query_hits * 6 + symbol_hits * 8 + lhs_hits * 48
        if metadata.get("type") == "formula":
            score += 40
        rows.append((score, index))
    rows.sort(key=lambda row: (-row[0], row[1]))
    return [index for _score, index in rows[:max_results]]


def repair_latex_json_escapes(value: Any) -> str:
    """Restore LaTeX commands consumed as JSON control-character escapes."""

    text = str(value or "")
    for control, repairs in _LATEX_JSON_ESCAPE_REPAIRS.items():
        for tail, command in repairs:
            pattern = re.compile(re.escape(control + tail) + r"(?![A-Za-z])")
            text = pattern.sub(lambda _match: command, text)
    return text


def supplement_formula_with_evidence(
    question: str,
    answer: str,
    texts: Iterable[str],
    metadatas: Iterable[dict[str, Any]] | None = None,
    *,
    max_lines: int = 2,
) -> str:
    """Append a literal formula line when a formula answer omits it.

    Only formula-typed/evidence-marked chunks are considered, and the appended
    line is visibly labeled as a quotation from retrieved evidence.
    """

    answer_text = str(answer or "").strip()
    if max_lines <= 0 or not is_formula_question(question):
        return answer_text
    answer_text = repair_latex_json_escapes(answer_text)
    answer_normalized = normalize_for_match(answer_text)
    query_terms = _formula_query_terms(question)
    query_symbols = {
        token.casefold()
        for token in re.findall(
            r"(?<![A-Za-z])[A-Za-z](?:_[A-Za-z]+)?(?![A-Za-z])",
            str(question or ""),
        )
    }
    metadata_list = list(metadatas or [])
    candidates: list[tuple[int, int, int, int, str]] = []
    for context_index, raw_text in enumerate(texts):
        metadata = metadata_list[context_index] if context_index < len(metadata_list) else {}
        if not isinstance(metadata, dict) or not (
            metadata.get("type") == "formula" or metadata.get("formula_evidence")
        ):
            continue
        for line_index, raw_line in enumerate(str(raw_text or "").splitlines()):
            line = raw_line.strip()
            if not line or len(line) > 360 or not FORMULA_EVIDENCE_RE.search(line):
                continue
            normalized_line = normalize_for_match(line)
            query_hits = sum(
                1 for term in query_terms if len(term) >= 3 and term in normalized_line
            )
            lhs = re.search(
                r"(?P<lhs>[A-Za-zα-ωΑ-Ω][A-Za-z0-9α-ωΑ-Ω′'_*()\[\]]{0,28})\s*(?:=|∈)",
                line,
            )
            lhs_key = normalize_for_match(lhs.group("lhs")) if lhs else ""
            symbol_hit = lhs_key in query_symbols if lhs_key else False
            operator_positions = [match.start() for match in re.finditer(r"[∈=]", line)]
            formula_span = (
                line[
                    lhs.start() if lhs else operator_positions[0] : operator_positions[0] + 180
                ]
                if operator_positions
                else line
            )
            span_hits = sum(
                1
                for term in query_terms
                if len(term) >= 3 and term in normalize_for_match(formula_span)
            )
            if not symbol_hit and (
                span_hits < 1 if "∈" in line else span_hits < 2
            ):
                continue
            candidates.append((query_hits, int(symbol_hit), -context_index, line_index, line))
    candidates.sort(reverse=True)
    selected: list[str] = []
    for _query_hits, _symbol_hit, _context_order, _line_index, line in candidates:
        normalized_line = normalize_for_match(line)
        if normalized_line in answer_normalized:
            continue
        lhs = re.search(
            r"(?P<lhs>[A-Za-zα-ωΑ-Ω][A-Za-z0-9α-ωΑ-Ω′'_*()\[\]]{0,28})\s*(?:=|∈)",
            line,
        )
        if lhs and normalize_for_match(lhs.group("lhs")) in answer_normalized:
            continue
        selected.append(line)
        if len(selected) >= max_lines:
            break
    if not selected:
        return answer_text
    return answer_text + "\n\n【公式原文核对项】\n" + "\n".join(
        f"- {line}" for line in selected
    )


def limitation_evidence_indices(
    question: str,
    texts: Iterable[str],
    metadatas: Iterable[dict[str, Any]] | None = None,
    *,
    max_results: int = 4,
    allowed_indices: Iterable[int] | None = None,
) -> list[int]:
    """Rank literal limitation/failure passages for an explicit limitation query.

    This is a bounded lexical aid, not a semantic classifier. It keeps the
    source scope under caller control and promotes passages that explicitly
    describe a limitation together with the query's state, failure, or example
    terms (for example a static-versus-dynamical contrast and its illustration).
    """

    if max_results <= 0 or not is_limitation_question(question):
        return []
    text_list = [str(text or "") for text in texts]
    metadata_list = list(metadatas or [])
    query_terms = _limitation_query_terms(question)
    allowed = (
        {int(index) for index in allowed_indices if int(index) >= 0}
        if allowed_indices is not None
        else None
    )
    rows: list[tuple[int, int]] = []
    for index, text in enumerate(text_list):
        if allowed is not None and index not in allowed:
            continue
        metadata = metadata_list[index] if index < len(metadata_list) else {}
        if isinstance(metadata, dict) and metadata.get("type") in {"figure", "image", "table"}:
            continue
        header = str(metadata.get("headers", "")) if isinstance(metadata, dict) else ""
        if re.search(r"references?|bibliography|参考文献", header, re.IGNORECASE):
            continue
        evidence_hits = len(LIMITATION_EVIDENCE_RE.findall(text))
        if not evidence_hits:
            continue
        normalized = normalize_for_match("\n".join((text, header)))
        query_hits = sum(1 for term in query_terms if term in normalized)
        if query_hits < 2:
            continue
        score = min(evidence_hits, 8) + query_hits * 6
        rows.append((score, index))
    rows.sort(key=lambda row: (-row[0], row[1]))
    return [index for _score, index in rows[:max_results]]


def extract_spatial_figure_chunks(
    blocks: Iterable[tuple[Any, ...]],
    source: str,
    page: int,
    page_width: float,
    page_height: float,
    *,
    max_region_height: float = 420.0,
    max_block_characters: int = 280,
    max_chunk_characters: int = 6000,
) -> list[Chunk]:
    """Build experimental figure evidence from a born-digital PDF text layer.

    ``PyMuPDF Page.get_text("blocks")`` returns block tuples beginning with
    ``(x0, y0, x1, y1, text, block_no, block_type)``. This helper consumes
    that shape, finds caption blocks, and records short text blocks immediately
    above each caption with normalized coordinates. It does not run OCR and it
    does not inspect image pixels; scanned/image-only figures remain unsupported.

    Blocks are serialized by horizontal centre and then vertical position.
    That keeps a panel label and the number printed below it adjacent while
    retaining explicit coordinates so generation can reject cross-panel pairs.
    """

    if page_width <= 0 or page_height <= 0:
        return []

    def separate_numeric_labels(value: str) -> str:
        """Separate adjacent two-decimal chart labels collapsed by PDF extraction."""

        # Some born-digital charts expose neighbouring labels as one token
        # (for example ``67.4068.80``). Restrict this repair to two-decimal
        # values so ordinary years, identifiers, and prose are left alone.
        return re.sub(r"(\d+\.\d{2})(?=\d+\.\d{2})", r"\1 / ", value)

    parsed: list[dict[str, Any]] = []
    for raw in blocks:
        if len(raw) < 7:
            continue
        try:
            x0, y0, x1, y1 = (float(raw[index]) for index in range(4))
            block_type = int(raw[6])
        except (TypeError, ValueError):
            continue
        if block_type != 0:
            continue
        text = " / ".join(
            line.strip() for line in str(raw[4]).splitlines() if line.strip()
        )
        text = separate_numeric_labels(text)
        if not text:
            continue
        parsed.append(
            {
                "x0": max(0.0, min(page_width, x0)),
                "y0": max(0.0, min(page_height, y0)),
                "x1": max(0.0, min(page_width, x1)),
                "y1": max(0.0, min(page_height, y1)),
                "text": text,
                "caption": FIGURE_CAPTION_RE.match(text),
            }
        )

    captions = [block for block in parsed if block["caption"] is not None]
    captions.sort(key=lambda block: (block["y0"], block["x0"]))
    output: list[Chunk] = []
    previous_captions: list[dict[str, Any]] = []
    for caption in captions:
        match = caption["caption"]
        assert match is not None
        kind = "extended_data_figure" if match.group("extended") else "figure"
        number = int(match.group("number"))
        overlapping_caption_bottom = max(
            (
                previous["y1"]
                for previous in previous_captions
                if min(previous["x1"], caption["x1"])
                > max(previous["x0"], caption["x0"])
            ),
            default=page_height * 0.05,
        )
        top = max(
            page_height * 0.05,
            overlapping_caption_bottom,
            caption["y0"] - min(max_region_height, page_height * 0.55),
        )
        candidates = []
        for block in parsed:
            if block is caption or block["caption"] is not None:
                continue
            if block["y0"] < top or block["y1"] > caption["y0"] + 2.0:
                continue
            if len(block["text"]) > max_block_characters:
                continue
            candidates.append(block)
        previous_captions.append(caption)
        if not candidates:
            continue

        candidates.sort(
            key=lambda block: (
                (block["x0"] + block["x1"]) / 2.0,
                block["y0"],
                block["x0"],
            )
        )
        label = (
            f"Extended Data Figure {number}"
            if kind == "extended_data_figure"
            else f"Figure {number}"
        )
        lines = [
            f"{label} spatial text evidence (PDF page {page}).",
            "This is text-layer evidence, not image recognition. Pair a label and value only "
            "when their horizontal x-ranges overlap; do not move values between non-overlapping "
            "visual groups.",
            "PDF page coordinates use a top-left origin: x increases rightward and y increases "
            "downward, so a larger y value is lower on the page.",
        ]
        for block in candidates:
            x0_pct = 100.0 * block["x0"] / page_width
            x1_pct = 100.0 * block["x1"] / page_width
            y0_pct = 100.0 * block["y0"] / page_height
            y1_pct = 100.0 * block["y1"] / page_height
            line = (
                f"[x={x0_pct:.1f}-{x1_pct:.1f}%; y={y0_pct:.1f}-{y1_pct:.1f}%] "
                f"{block['text']}"
            )
            if sum(len(existing) + 1 for existing in lines) + len(line) > max_chunk_characters:
                break
            lines.append(line)
        lines.append(f"Caption: {caption['text']}")
        output.append(
            Chunk(
                page_content="\n".join(lines),
                metadata={
                    "source": source,
                    "page": page,
                    "type": "figure",
                    "figure_number": number,
                    "figure_kind": kind,
                    "figure_label": label,
                    "figure_caption": caption["text"],
                    "spatial_evidence": True,
                    "spatial_layout": "x_center_then_y",
                },
            )
        )
    return output


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
        header = (
            metadata.get("headers") or metadata.get("section_context")
            if isinstance(metadata, dict)
            else None
        )
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
    process_question = bool(
        re.search(
            r"管道|流程|步骤|构建|pipeline|process|construct|steps",
            question_normalized,
            re.IGNORECASE,
        )
    )
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
            r"token-level statistics|held[- ]?out|training mixture|"
            r"chemicalqa|moleculenet|ultrachat|cpt\s+(?:text\s+)?corpus|"
            r"corpus statistics|tokens?/example|\bcjk\b",
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
        # For numeric questions, named entities from neighboring prose are not
        # reliable omissions. Keep entity completion for process questions,
        # where tool/model names are part of the requested pipeline.
        markers = [*numbers, *(high_signal_entities if process_question else [])]
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
        if (
            not process_question
            and numbers
            and not any(
                normalize_for_match(number) in answer_normalized
                for number in numbers
            )
        ):
            # A numeric answer should not inherit an unrelated neighboring
            # line merely because that line contains generic dataset terms.
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


def _validation_marker_key(value: Any, kind: str) -> str:
    """Normalize a validator marker without changing the displayed evidence."""

    normalized = normalize_for_match(value)
    if kind == "number":
        # Commas and spaces are presentation details (4,855 vs 4855). Keep
        # the decimal point and comparison/unit suffix because those can
        # change the meaning of a scientific value.
        normalized = re.sub(r"(?<=\d)[,\s]+(?=\d)", "", normalized)
    else:
        # PDF-to-Markdown often changes a hyphenated model name into a spaced
        # form. This comparison is only for a review signal; the original
        # marker remains in the diagnostic output.
        normalized = re.sub(r"[-_\s*]+", "", normalized)
    return normalized


def _validation_markers(content: str) -> list[dict[str, str]]:
    """Extract conservative numeric and named-entity markers from one line."""

    cleaned_content = html.unescape(str(content or ""))
    cleaned_content = re.sub(r"<[^>]*>", " ", cleaned_content)
    cleaned_content = unicodedata.normalize("NFKC", cleaned_content)
    cleaned_content = re.sub(r"[*_~`]+", "", cleaned_content)
    markers: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for match in EVIDENCE_NUMBER_RE.finditer(cleaned_content):
        value = match.group(0)
        # Do not split numeric suffixes out of identifiers such as
        # ``text-bison-001`` or ``R1126``. They are model/reference labels,
        # not answer quantities.
        start, end = match.span()
        if (
            (start > 0 and cleaned_content[start - 1] in "-_" and start > 1 and cleaned_content[start - 2].isalnum())
            or (end < len(cleaned_content) and cleaned_content[end].isalnum())
        ):
            continue
        numeric_match = re.match(
            r"[<>≤≥]?\s*\d[\d,]*(?:\.\d+)?(?:\s*%)?", str(value)
        )
        numeric_text = numeric_match.group(0).strip() if numeric_match else str(value)
        if re.fullmatch(r"[<>≤≥]?\s*\d+,\d", numeric_text):
            # A common pymupdf4llm artifact from a split subscript (``1,0``)
            # is not a meaningful English scientific quantity.
            continue
        # Avoid treating a short section ordinal or a coordinate fragment as
        # a fact. Decimals, percentages, thresholds, and multi-digit counts
        # are retained because they are the useful answer-bearing markers.
        digits = re.sub(r"\D", "", numeric_text)
        if not (
            re.search(r"\d+\.\d+", numeric_text)
            or re.search(r"[%<>≤≥]", numeric_text)
            or len(digits) >= 2
        ):
            continue
        key = ("number", _validation_marker_key(numeric_text, "number"))
        if key in seen:
            continue
        seen.add(key)
        markers.append({"kind": "number", "text": numeric_text})
    for value in VALIDATION_ENTITY_TOKEN_RE.findall(cleaned_content):
        if value in VALIDATION_ENTITY_STOPWORDS:
            continue
        stripped = re.sub(r"[^A-Za-z0-9]", "", value)
        if len(stripped) < 3:
            continue
        key = ("entity", _validation_marker_key(value, "entity"))
        if key in seen:
            continue
        seen.add(key)
        markers.append({"kind": "entity", "text": value})
    return markers


def _validation_marker_is_near_intent(
    content: str,
    marker: dict[str, str],
    question_terms: set[str],
    *,
    window: int = 45,
) -> bool:
    """Keep a marker only when nearby prose expresses the asked intent."""

    normalized_content = normalize_for_match(content)
    marker_text = normalize_for_match(marker["text"])
    marker_position = normalized_content.find(marker_text)
    if marker_position < 0:
        # Markdown/HTML decoration can split a marker; retaining it is safer
        # than silently claiming that the evidence has no answer-bearing
        # value. The surrounding line still has to match question terms.
        return True
    intent_terms = {
        term
        for term in question_terms
        if term.casefold() in VALIDATION_PROXIMITY_TERMS
    }
    if not intent_terms:
        return True
    before = normalized_content[max(0, marker_position - window) : marker_position]
    after = normalized_content[marker_position + len(marker_text) : marker_position + len(marker_text) + window]
    return any(term in before or term in after for term in intent_terms)


def _validation_question_terms(question: str) -> set[str]:
    """Expand question terms only with generic bilingual intent aliases."""

    normalized = normalize_for_match(question)
    terms = _evidence_query_tokens(question)
    aliases = {
        "数据集": ("dataset", "data", "sample", "samples"),
        "样本": ("dataset", "sample", "samples"),
        "数量": ("number", "count", "sample", "samples", "size"),
        "管道": ("pipeline", "process", "construct", "construction", "annotation", "label"),
        "构建": ("pipeline", "process", "construct", "construction", "annotation", "label"),
        "标注": ("annotation", "annotate", "label", "rationale"),
        "推理": ("reasoning", "inference", "rationale"),
        "训练": ("training", "train", "fine-tun", "pretrain"),
        "强化学习": ("reinforcement", "rl", "reward", "grpo"),
        "奖励": ("reward", "objective", "pareto"),
        "步骤": ("step", "steps", "first", "second", "third", "process"),
        "公式": ("equation", "formula", "system", "domain"),
        "初始化": ("initial", "initialize", "state"),
        "更新": ("update", "iteration", "residual"),
        "限制": ("restriction", "limitation", "restrict"),
        "延拓": ("prolongation", "interpolation"),
        "循环": ("cycle", "cycles", "v-cycle", "w-cycle"),
        "切分": ("split", "splitting", "subsection", "subsections", "chunk", "chunks"),
        "检索": ("retrieval", "retrieve", "ranking", "rank", "relevance", "top"),
        "基线": ("baseline", "rag", "retrieval"),
        "图": ("figure", "fig", "panel"),
    }
    for cue, cue_aliases in aliases.items():
        if cue in normalized:
            terms.update(cue_aliases)
    return terms


def _validation_marker_present(marker: dict[str, str], answer_normalized: str) -> bool:
    """Check a literal marker while tolerating harmless PDF formatting changes."""

    marker_key = _validation_marker_key(marker["text"], marker["kind"])
    if not marker_key:
        return False
    if marker["kind"] == "number":
        answer_key = re.sub(r"(?<=\d)[,\s]+(?=\d)", "", answer_normalized)
        return marker_key in answer_key
    answer_key = re.sub(r"[-_\s*]+", "", answer_normalized)
    return marker_key in answer_key


def validate_answer_against_evidence(
    question: str,
    answer: str,
    evidence_ledger: Iterable[str],
    *,
    max_candidate_lines: int = 8,
) -> dict[str, Any]:
    """Return a conservative, evidence-only answer review signal.

    This is deliberately *not* a correctness judge. It has no access to
    benchmark gold answers and never invents a missing fact. It only asks
    whether a generated answer appears to omit salient literal markers from
    ledger lines that match the question. ``review`` means a human or an
    explicitly enabled retry may inspect the answer; the default web path
    does not change the answer based on this diagnostic.
    """

    answer_text = str(answer or "").strip()
    ledger_lines = [str(line or "").strip() for line in evidence_ledger if str(line or "").strip()]
    result: dict[str, Any] = {
        "status": "ok",
        "recommended_action": "none",
        "reasons": [],
        "relevant_line_count": 0,
        "candidate_line_count": 0,
        "candidate_lines": [],
        "missing_markers": [],
    }
    if not answer_text:
        result.update(
            status="review",
            recommended_action="retry_or_manual_review",
            reasons=["empty_answer"],
        )
        return result
    if re.search(r"(?:调用出错|请求失败|api\s+error|rate\s*limit)", answer_text, re.IGNORECASE):
        result.update(
            status="review",
            recommended_action="retry_or_manual_review",
            reasons=["generation_error"],
        )
        return result
    if not ledger_lines:
        result.update(status="insufficient_evidence", recommended_action="manual_review")
        return result

    question_terms = _validation_question_terms(question)
    question_normalized = normalize_for_match(question)
    if re.search(r"\btables?\s*\d+\b|表\s*\d+", question_normalized, re.IGNORECASE):
        # Explicit table row/column questions are answered by the structured
        # cell path before generation. Do not run narrative marker checks on
        # a full Markdown row, where every neighboring cell is visible but
        # not part of the requested answer.
        result.update(
            status="not_applicable",
            recommended_action="none",
            reasons=["structured_table_path"],
        )
        return result
    if re.search(r"\[x\s*=|图形坐标文字|spatial\s+text|figure\s+coordinate", " ".join(ledger_lines), re.IGNORECASE):
        # Coordinate evidence requires a panel/group consistency check that
        # cannot be reduced to line-level marker coverage. Keep this validator
        # separate from the spatial-figure guard rather than issuing a false
        # omission warning for neighboring labels and axes.
        result.update(
            status="not_applicable",
            recommended_action="manual_review",
            reasons=["spatial_figure_requires_coordinate_scope"],
        )
        return result
    if re.search(r"方程|公式|\bequation\b|\bformula\b|\bPDE\b|参数量级|复杂度|\bcomplexity\b", question_normalized, re.IGNORECASE):
        # Formula equivalence, superscripts, and operator order need a
        # structure-aware checker; literal numeric markers are insufficient.
        result.update(
            status="not_applicable",
            recommended_action="manual_review",
            reasons=["formula_or_operator_semantics_required"],
        )
        return result
    numeric_question = bool(
        re.search(
            r"多少|数量|占比|比例|得分|数值|值|样本数|参数量级|复杂度|尺寸|多久|时间|"
            r"how\s+many|how\s+long|number|count|score|value|percentage|proportion|"
            r"complexity|parameter|kernel|size|time|steps",
            question_normalized,
            re.IGNORECASE,
        )
    )
    composite_question = bool(
        re.search(
            r"和|以及|并且|分别|同时|管道|流程|步骤|pipeline|process|annotation|construct|"
            r"how\s+.*\s+and\s+|what\s+.*\s+and\s+",
            question_normalized,
            re.IGNORECASE,
        )
    )
    process_question = bool(
        re.search(
            r"管道|流程|步骤|构建|标注|pipeline|process|annotation|construct|steps",
            question_normalized,
            re.IGNORECASE,
        )
    )
    marker_question_terms = set(question_terms)
    if numeric_question:
        if re.search(r"样本|how\s+many|count|number|dataset\s+size|数据集", question_normalized, re.IGNORECASE):
            focus_terms = {
                "dataset",
                "sample",
                "samples",
                "pair",
                "pairs",
                "comprises",
                "consists",
                "contains",
                "count",
                "number",
                "size",
                "total",
            }
        elif re.search(r"多久|how\s+long|时间|steps|training", question_normalized, re.IGNORECASE):
            focus_terms = {"step", "steps", "within", "after", "time", "optimizer"}
        elif re.search(r"参数|复杂度|complexity|parameter|kernel|尺寸", question_normalized, re.IGNORECASE):
            focus_terms = {"parameter", "parameters", "complexity", "kernel", "size", "dimension"}
        elif re.search(r"占比|比例|percentage|proportion", question_normalized, re.IGNORECASE):
            focus_terms = {"percentage", "proportion", "dataset", "score", "value"}
        else:
            focus_terms = {
                "score",
                "value",
                "values",
                "threshold",
                "figure",
                "fig",
                "protein",
                "rna",
                "dna",
                "metric",
                "dataset",
            }
        if re.search(r"图|figure|fig|评估集|complex", question_normalized, re.IGNORECASE):
            focus_terms.update({"figure", "fig", "protein", "rna", "dna", "structure", "structures", "benchmark", "composed", "released", "n"})
        if re.search(r"PoseBusters|样本|structures?|benchmark|success|criterion", question_normalized, re.IGNORECASE):
            focus_terms.update({"benchmark", "benchmarks", "structure", "structures", "composed", "released", "success", "criterion", "PDB"})
        numeric_focus = {
            term for term in question_terms if term.casefold() in focus_terms
        }
        # Generic focus terms may not literally occur in the question; they
        # are still useful aliases for bilingual intent matching.
        numeric_focus.update(focus_terms)
        if numeric_focus:
            marker_question_terms = numeric_focus
    answer_normalized = normalize_for_match(answer_text)
    candidate_rows: list[dict[str, Any]] = []
    for line_index, line in enumerate(ledger_lines):
        content = line.split("】", 1)[-1].strip()
        prefix = line.split("】", 1)[0]
        normalized_content = normalize_for_match(content)
        normalized_prefix = normalize_for_match(prefix)
        # A narrative mention of ``Table 1`` can share words such as
        # dataset/score with a non-table question while contributing only
        # neighboring metrics. Explicit table questions are returned earlier
        # through the structured path, so skip these prose/table rows here.
        if re.search(
            r"^\s*(?:table|表)\s*\d+\b|^\s*(?:as shown in|according to|based on)"
            r"[^.]{0,30}\b(?:table|表)\s*\d+\b",
            content,
            re.IGNORECASE,
        ):
            continue
        query_hits = sum(
            1
            for term in question_terms
            if term and (term in normalized_content or term in normalized_prefix)
        )
        if query_hits <= 0:
            continue
        markers = [
            marker
            for marker in _validation_markers(content)
            if numeric_question or marker["kind"] == "entity"
        ]
        markers = [
            marker
            for marker in markers
            if _validation_marker_is_near_intent(
                content,
                marker,
                question_terms if marker["kind"] == "entity" and composite_question else marker_question_terms,
            )
        ]
        if not markers:
            continue
        # Full Markdown rows are often tables with many unrelated cells. A
        # deterministic cell path already handles explicit table questions;
        # skipping them here prevents a generic narrative answer from being
        # flagged just because it did not repeat every cell in the row.
        if content.count("|") >= 2:
            continue
        numeric_marker_count = sum(marker["kind"] == "number" for marker in markers)
        candidate_rows.append(
            {
                "line_index": line_index,
                "line": line,
                "markers": markers,
                "query_hits": query_hits,
                "score": (
                    query_hits * 5
                    + min(len(markers), 3)
                    + (10 if numeric_question and numeric_marker_count else 0)
                ),
            }
        )
    candidate_rows.sort(key=lambda row: (-row["score"], row["line_index"]))
    # A ledger can contain several sibling chunks (and, for a multi-paper
    # collection, several papers) that share generic words such as
    # ``dataset`` or ``training``. For a numeric question, prefer the best
    # number-bearing line and, only for a composite/process question, one
    # complementary entity-bearing line. This keeps the signal useful for a
    # retry gate without turning every neighboring citation into a missing
    # answer fact.
    if numeric_question:
        numeric_rows = [
            row for row in candidate_rows if any(marker["kind"] == "number" for marker in row["markers"])
        ]
        selected_rows = numeric_rows[:1]
        if process_question:
            entity_rows = [
                row
                for row in candidate_rows
                if not any(marker["kind"] == "number" for marker in row["markers"])
            ]
            if entity_rows:
                selected_rows.append(entity_rows[0])
    else:
        selected_rows = candidate_rows[:1]
    selected_rows = selected_rows[: max(0, int(max_candidate_lines))]
    result["relevant_line_count"] = len(candidate_rows)
    result["candidate_line_count"] = len(selected_rows)
    missing_by_line: list[dict[str, Any]] = []
    for row in selected_rows:
        markers = row["markers"]
        covered = [
            marker
            for marker in markers
            if _validation_marker_present(marker, answer_normalized)
        ]
        missing = [marker for marker in markers if marker not in covered]
        # Numeric questions should explicitly carry the salient number from
        # a matched evidence line. For composite questions, require a human
        # review if a strongly matched line contributes multiple markers and
        # the answer covers only part of them. This is a warning, not proof of
        # an incorrect answer.
        line_needs_review = False
        reason = ""
        if numeric_question and any(marker["kind"] == "number" for marker in markers):
            numeric_markers = [marker for marker in markers if marker["kind"] == "number"]
            if not any(marker in covered for marker in numeric_markers):
                line_needs_review = True
                reason = "missing_relevant_number"
        if (
            not line_needs_review
            and composite_question
            and row["query_hits"] >= 2
            and len(markers) >= 2
            and missing
            and covered
        ):
            line_needs_review = True
            reason = "partial_high_signal_line"
        if (
            not line_needs_review
            and composite_question
            and row["query_hits"] >= 2
            and len(markers) >= 2
            and not covered
        ):
            line_needs_review = True
            reason = "omitted_high_signal_line"
        if (
            not line_needs_review
            and process_question
            and row["query_hits"] >= 3
            and markers
            and all(marker["kind"] == "entity" for marker in markers)
            and not covered
        ):
            line_needs_review = True
            reason = "omitted_process_marker"
        if line_needs_review:
            missing_by_line.append(
                {
                    "line": row["line"],
                    "reason": reason,
                    "missing_markers": missing,
                    "covered_markers": covered,
                }
            )
    result["candidate_lines"] = [
        {
            "line": row["line"],
            "query_hits": row["query_hits"],
            "markers": row["markers"],
        }
        for row in selected_rows
    ]
    result["missing_markers"] = [
        marker
        for row in missing_by_line
        for marker in row["missing_markers"]
    ]
    if missing_by_line:
        result.update(
            status="review",
            recommended_action="manual_review",
            reasons=list(dict.fromkeys(row["reason"] for row in missing_by_line)),
            flagged_lines=missing_by_line,
        )
    else:
        result["flagged_lines"] = []
    return result


def build_evidence_retry_prompt(
    question: str,
    answer: str,
    evidence_ledger: Iterable[str],
    validation: dict[str, Any],
    *,
    max_lines: int = 4,
) -> str:
    """Build a bounded second-pass prompt from the validator's literal lines.

    The prompt is intentionally independent of benchmark gold data. It asks a
    model to preserve supported parts of the first answer and repair only
    omissions that are visible in the retrieved evidence. Callers decide
    whether a second API request is affordable; this helper itself performs
    no network operation and does not imply that the first answer was wrong.
    """

    answer_text = str(answer or "").strip()
    flagged = validation.get("flagged_lines") or []
    evidence_lines = [
        str(item.get("line", "")).strip()
        for item in flagged
        if isinstance(item, dict) and str(item.get("line", "")).strip()
    ]
    if not evidence_lines:
        evidence_lines = [str(line).strip() for line in evidence_ledger if str(line).strip()]
    evidence_lines = list(dict.fromkeys(evidence_lines))[: max(0, int(max_lines))]
    evidence_text = "\n".join(f"- {line}" for line in evidence_lines)
    reason_text = "、".join(str(reason) for reason in validation.get("reasons") or [])
    return (
        "请对下面的科学论文答案做一次严格的证据核对并输出修订后的最终答案。\n"
        "只允许使用‘证据核对项’中逐字出现的信息；保留原答案中已有且有证据支持的内容，"
        "补齐与问题直接相关而原答案遗漏的数字、工具/模型名和流程步骤。不要把参考文献年份、"
        "图轴数字或不相关邻近事实当作答案；无法由证据确定时明确说资料未提供。只输出答案正文，"
        "不要解释自检过程。\n\n"
        f"【问题】\n{question}\n\n"
        f"【原答案】\n{answer_text}\n\n"
        f"【证据核对项】（触发原因：{reason_text or '未分类'}）\n{evidence_text or '- 无可用证据行'}"
    )


def display_table_cell(value: Any) -> str:
    """Remove presentation markup while preserving the cell's value."""

    text = html.unescape(str(value or ""))
    # Keep a separator when PDF table conversion uses ``<br>`` for multiple
    # values in one cell (for example caption-vs-representation metrics).
    text = re.sub(r"<br\s*/?>", " / ", text, flags=re.IGNORECASE)
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
    used_caption_indices: set[int] | None = None,
) -> tuple[str, int | None]:
    """Find a caption immediately before or after a Markdown table.

    ``pymupdf4llm`` commonly emits captions after the table, while LaTeX
    exports often place them before it.  Only blank lines may occur between a
    table and its caption; this prevents a prose cross-reference to another
    table from being attached accidentally.
    """

    used = used_caption_indices or set()
    # Prefer an unused preceding caption (common in LaTeX exports). Once that
    # caption has been assigned, the next table may use its following caption
    # without inheriting the previous table's label.
    for direction, start in ((-1, header_idx - 1), (1, table_end_idx)):
        candidate_idx = start
        blank_lines = 0
        while 0 <= candidate_idx < len(lines) and not lines[candidate_idx].strip():
            blank_lines += 1
            if blank_lines > 3:
                break
            candidate_idx += direction
        if (
            0 <= candidate_idx < len(lines)
            and candidate_idx not in used
            and _is_table_caption(lines[candidate_idx])
        ):
            return lines[candidate_idx].strip(), candidate_idx
    return "未命名表格", None


def _clean_header_cell(value: Any) -> str:
    """Normalize a header cell while retaining readable metric names."""

    text = html.unescape(str(value or ""))
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)
    text = display_table_cell(text)
    # A PDF line wrap can split a single header token (``Narrat iveQA``).
    # Join fragments containing an internal capital, leaving ordinary
    # multi-word headers (for example ``Darcy s``) available to the grouped
    # header repair below.
    parts = text.split()
    joined: list[str] = []
    for part in parts:
        if joined and part and part[0].islower() and part.isalpha() and any(
            character.isupper() for character in part[1:]
        ):
            joined[-1] += part
        else:
            joined.append(part)
    return " ".join(joined)


def _repair_wrapped_header_line(header_line: str) -> str:
    """Repair cross-cell group labels split by a PDF table converter."""

    cells = _split_markdown_row(header_line)
    parts: list[tuple[str, str] | None] = []
    for cell in cells:
        raw_parts = re.split(r"<br\s*/?>", str(cell), flags=re.IGNORECASE)
        if len(raw_parts) < 2:
            parts.append(None)
            continue
        parts.append(
            (
                _clean_header_cell(raw_parts[0]),
                _clean_header_cell(" ".join(raw_parts[1:])),
            )
        )
    changed = False
    for index in range(len(parts) - 1):
        left = parts[index]
        right = parts[index + 1]
        if left is None or right is None:
            continue
        left_group, left_metric = left
        right_group, right_metric = right
        if (
            left_group
            and right_group
            and re.search(r"[A-Za-z]$", left_group)
            and re.match(r"[a-z]", right_group)
        ):
            merged_group = f"{left_group}{right_group}"
            parts[index] = (merged_group, left_metric)
            parts[index + 1] = (merged_group, right_metric)
            changed = True
    if not changed:
        return header_line
    repaired = []
    for cell, split in zip(cells, parts):
        if split is None:
            repaired.append(cell)
            continue
        group, metric = split
        repaired.append(f"{group} {metric}".strip())
    return _join_markdown_cells(repaired)


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
    if hints >= 2 and numeric <= len(cells) // 2:
        return True
    # Some PDF exporters emit a second header row made entirely of repeated
    # dataset labels (for example three groups of PopQA/NQ/TriviaQA), none of
    # which are generic metric words.  Only accept this shape when every cell
    # is textual and at least two labels repeat; the caller still requires the
    # first row to look like a spanning header.
    clean = [_clean_header_cell(cell) for cell in cells]
    normalized = [normalize_for_match(cell) for cell in clean if cell]
    repeated = len(normalized) - len(set(normalized))
    return (
        numeric == 0
        and len(normalized) == len(cells)
        and repeated >= 2
    )


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
            first_fragment, separator, remainder = next_label.partition(" ")
            if separator and len(first_fragment) <= 4:
                # A wrapped word can share a cell with the next header word
                # (``Inter`` + ``nal Kno``). Join only the short lowercase
                # fragment and retain the following word boundary.
                label = f"{label}{first_fragment} {remainder}".strip()
                end += 1
                continue
            # PDF line wrapping can split a word across adjacent cells (for
            # example ``Darcy s`` + ``mooth``).  Join a one-letter trailing
            # fragment directly; normal multi-word group labels retain a
            # separating space (``Darcy`` + ``rough``).
            last_token = label.rsplit(" ", 1)[-1]
            joiner = (
                ""
                if (
                    (len(last_token) == 1 and last_token.isalpha())
                    or len(last_token) <= 4
                    or "-" in last_token
                    or len(next_label) <= 4
                )
                else " "
            )
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
    used_caption_indices: set[int] = set()
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
        header_line = _repair_wrapped_header_line(header_line)
        if header_line:
            header_line = _join_markdown_cells(
                [_clean_header_cell(cell) for cell in _split_markdown_row(header_line)]
            )
        header_line, data_rows = _canonicalize_unit_column(header_line, data_rows)
        caption, caption_idx = _caption_near(
            lines, header_idx, j, used_caption_indices
        )
        if caption_idx is not None:
            used_caption_indices.add(caption_idx)
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
        if base_metadata.get("type") in {"figure", "formula"}:
            if document.page_content.strip():
                final_chunks.append(
                    Chunk(page_content=document.page_content, metadata=base_metadata)
                )
            continue
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


def _row_entity_label(row: str, target: str) -> str | None:
    """Return the display cell matching an entity anywhere in a table row.

    PDF-to-Markdown converters frequently leave merged leading cells blank and
    place the model/setting in a later column.  Restrict matching to non-empty
    non-numeric cells so a requested entity can be found without treating a
    metric value as a row label.
    """

    normalized_target = _row_entity_key(target)
    if not normalized_target:
        return None
    for cell in _split_markdown_row(row):
        display = display_table_cell(cell)
        normalized = _row_entity_key(display)
        if not normalized or _looks_numeric_cell(display):
            continue
        if normalized_target == normalized or normalized_target in normalized:
            return display
        # PDF table extraction may drop punctuation in a model name
        # (``ChatGPT-4o-Mini`` -> ``ChatGPT-4oMINI``).  Treat a punctuation-only
        # difference as the same entity without fuzzy-matching similar models.
        compact_target = re.sub(r"[^a-z0-9]", "", normalized_target)
        compact_value = re.sub(r"[^a-z0-9]", "", normalized)
        if compact_target and compact_target == compact_value:
            return display
        # Allow a small decoration difference such as a superscript star, but
        # do not accept a short component (``WikiTQ``) as the requested
        # composite entity (``WikiTQ+SQA+SciGen``).
        if normalized in normalized_target and len(normalized_target) - len(normalized) <= 2:
            return display
    return None


def _row_entity_key(value: Any) -> str:
    """Normalize spacing around separators used in composite row labels."""

    text = normalize_for_match(value)
    text = re.sub(r"\s*\+\s*", "+", text)
    text = re.sub(r"\s*/\s*", "/", text)
    return text


def _table_row_group_marker(row: str) -> str | None:
    """Return a normalized section marker carried by a table row, if any.

    Scientific tables often encode a spanning group header as a row whose
    text is split across adjacent cells (for example ``Test(C&L)`` becoming
    ``Test(C&L`` and ``)``).  Joining and removing Markdown cell separators
    lets the lookup code preserve that group context without depending on a
    particular PDF converter's exact cell layout.
    """

    display = display_table_cell(row)
    flattened = re.sub(r"[|\s]+", "", display).casefold()
    match = re.search(
        r"(?<![a-z0-9])(?:test|dev|validation|evaluation|setting|configuration|split)\("
        r"([^)]{1,48})\)",
        flattened,
    )
    if match is not None:
        return match.group(0).casefold()
    # PDF table exporters often flatten a spanning label such as
    # ``Oracle setting`` into a row whose cells contain only fragments.
    setting = re.search(
        r"(?:oracle|closedbook|openbook)setting",
        flattened,
    )
    return setting.group(0).casefold() if setting is not None else None


def _is_table_section_marker(row: str) -> bool:
    """Return whether a row is a one-cell or wrapped section label."""

    cells = [display_table_cell(cell) for cell in _split_markdown_row(row)]
    non_empty = [cell for cell in cells if cell]
    return bool(
        len(non_empty) == 1
        or (
            len(non_empty) == 2
            and all(not _looks_numeric_cell(cell) for cell in non_empty)
            and len(non_empty[1]) <= 2
        )
    )


def _table_question_group(question: str) -> str | None:
    """Extract a normalized table section marker explicitly named in a query."""

    text = normalize_for_match(question)
    flattened = re.sub(r"\s+", "", text)
    match = re.search(
        r"(?<![a-z0-9])(?:test|dev|validation|evaluation|setting|configuration|split)\("
        r"([^)]{1,48})\)",
        flattened,
    )
    if match is None:
        return None
    return match.group(0).casefold()


def _table_data_rows_with_groups(table_content: str) -> list[tuple[str, str | None]]:
    """Return data rows paired with the most recent spanning group marker."""

    rows: list[tuple[str, str | None]] = []
    lines = table_content.splitlines()
    separator_idx = next(
        (idx for idx, line in enumerate(lines) if TABLE_SEPARATOR_RE.fullmatch(line.strip())),
        0,
    )
    current_group: str | None = None
    for line in lines[separator_idx + 1 :]:
        if not (line.strip().startswith("|") and line.strip().endswith("|")):
            continue
        marker = _table_row_group_marker(line)
        if marker is not None:
            current_group = marker
            continue
        cells = [display_table_cell(cell) for cell in _split_markdown_row(line)]
        non_empty = [cell for cell in cells if cell]
        if _is_table_section_marker(line):
            # Section labels are often emitted as one cell, or split across
            # two cells when the final character wraps (for example
            # ``Llama-3-Ins-70`` + ``B``). Keep them as a bounded row group.
            label = "".join(non_empty) if len(non_empty) == 2 else non_empty[0]
            current_group = f"section:{_row_entity_key(label)}"
            continue
        rows.append((line, current_group))
    return rows


def _table_question_section(question: str, table_content: str) -> str | None:
    """Return a named one-cell section (usually a model family) from a query."""

    for _row, group in _table_data_rows_with_groups(table_content):
        if not group or not group.startswith("section:"):
            continue
        label = group.removeprefix("section:")
        if len(label) < 8 and not re.search(r"\d", label):
            continue
        if _question_mentions_label(question, label):
            return group
    return None


def _table_row_labels(table_content: str) -> list[str]:
    """Return non-numeric cells that can act as row/entity labels."""

    parsed = parse_markdown_table(table_content)
    if parsed is None:
        return []
    _, rows = parsed
    labels: list[str] = []
    # Keep section marker rows out of the entity list.  PDF exporters may
    # split ``Llama-3-Ins-70B`` into ``Llama-3-Ins-70`` + ``B``; the section
    # parser already recognizes that marker and carries it to its data rows.
    data_rows = _table_data_rows_with_groups(table_content)
    if data_rows:
        rows = [_split_markdown_row(row) for row, _group in data_rows]
    for row in rows:
        row_text = _join_markdown_cells(row)
        if _table_row_group_marker(row_text) is not None:
            continue
        row_cells = _split_markdown_row(row_text)
        if not any(display_table_cell(cell) for cell in row_cells[1:]):
            continue
        for cell in row_cells:
            display = display_table_cell(cell)
            display = re.sub(r"^\s*\d+\.\s*", "", display).strip()
            if display and not _looks_numeric_cell(display) and display not in labels:
                labels.append(display)
    return labels


def _question_mentions_label(question: str, label: str) -> bool:
    question_key = normalize_for_match(question)
    label_key = _row_entity_key(label)
    if not label_key:
        return False
    compact_question = re.sub(r"[^a-z0-9]", "", question_key)
    compact_label = re.sub(r"[^a-z0-9]", "", label_key)
    return bool(
        re.search(
            rf"(?<![a-z0-9]){re.escape(label_key)}(?![a-z0-9])",
            question_key,
        )
        or len(compact_label) >= 4
        and compact_label
        and compact_label in compact_question
    )


def _question_relation_targets(question: str, labels: Iterable[str]) -> list[str]:
    """Return labels used as the object before an ``on/在`` qualifier."""

    question_key = normalize_for_match(question)
    return [
        label
        for label in labels
        if re.search(
            rf"(?<![a-z0-9]){re.escape(_row_entity_key(label))}(?![a-z0-9])\s*(?:在|on)\b",
            question_key,
            re.IGNORECASE,
        )
    ]


def _question_relation_qualifiers(question: str) -> list[str]:
    """Return compact ASCII values appearing after ``on/在``."""

    values: list[str] = []
    for match in re.finditer(
        r"(?:在|on)\s*([A-Za-z0-9_*+\-]+)",
        normalize_for_match(question),
        re.IGNORECASE,
    ):
        value = match.group(1)
        if value.casefold() not in {"oracle", "closed", "open"}:
            values.append(value)
    return list(dict.fromkeys(values))


def _table_question_entities(question: str, table_content: str) -> list[str]:
    """Select row labels while separating ``model on dataset`` qualifiers."""

    requested_group = _table_question_group(question)
    if requested_group and not {
        group for _, group in _table_data_rows_with_groups(table_content) if group is not None
    }:
        return []
    labels = _table_row_labels(table_content)
    mentioned = [label for label in labels if _question_mentions_label(question, label)]
    # Ablation questions often name only the removed component (RT) while the
    # row carries a generic w/o prefix.  Similarly, complete/full selects a
    # row named Ours.
    for label in labels:
        stripped = re.sub(r"^w(?:/o)?\s+", "", label, flags=re.IGNORECASE)
        if stripped != label and _question_mentions_label(question, stripped):
            mentioned.append(label)
        if (
            normalize_for_match(label) in {"ours", "full", "full model"}
            and re.search(r"完整|全部|full|ours", question, re.IGNORECASE)
        ):
            mentioned.append(label)
    if len([label for label in mentioned if re.match(r"^w/o\b", label, re.IGNORECASE)]) > 1:
        mentioned.extend(
            label for label in labels if normalize_for_match(label) == "ours"
        )
    # Spacing around ``+`` and similar separators is frequently unstable in
    # PDF Markdown output; keep one label for one logical entity.
    deduped: list[str] = []
    seen_keys: set[str] = set()
    for label in mentioned:
        key = _row_entity_key(label)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        deduped.append(label)
    mentioned = deduped
    usage_group = _table_usage_group(question, table_content)
    if usage_group and len(mentioned) > 1:
        mentioned = [
            label
            for label in mentioned
            if _row_entity_key(label) != _row_entity_key(usage_group)
        ]
    relation_targets = _question_relation_targets(question, mentioned)
    if relation_targets:
        qualifier_keys = {
            _row_entity_key(value) for value in _question_relation_qualifiers(question)
        }
        candidates = [
            label
            for label in mentioned
            if label in relation_targets or _row_entity_key(label) not in qualifier_keys
        ]
        # In phrasing such as ``ChatGPT ... 使用 THINKNOTE`` the model is
        # mentioned before the method, while the ``在 PopQA`` relation would
        # otherwise make the method look like the row entity.  Prefer the
        # named entity before a generic usage verb when both are present.
        usage = re.search(r"使用|采用|using|with", question, re.IGNORECASE)
        if usage and len(candidates) > 1:
            question_prefix = question[: usage.start()]
            before_usage = [
                label
                for label in candidates
                if _question_mentions_label(question_prefix, label)
            ]
            if before_usage:
                return [max(before_usage, key=len)]
        return candidates
    if len(mentioned) > 1:
        # A question may mention a dataset and a model from the same row. They
        # are not two requested rows; keep the later (usually model) column.
        columns: dict[str, set[int]] = {label: set() for label in mentioned}
        parsed = parse_markdown_table(table_content)
        if parsed is not None:
            _, rows = parsed
            for row in rows:
                row_text = _join_markdown_cells(row)
                if _table_row_group_marker(row_text) is not None:
                    continue
                for index, cell in enumerate(row):
                    display = display_table_cell(cell)
                    for label in mentioned:
                        if _row_entity_label(display, label) is not None:
                            columns[label].add(index)
        distinct_columns = {tuple(sorted(values)) for values in columns.values() if values}
        if len(distinct_columns) > 1:
            return [max(mentioned, key=lambda label: max(columns[label], default=-1))]
    return mentioned


def _table_question_group_for_entity(question: str, entity: str) -> str | None:
    """Return a normalized setting group attached to one queried entity."""

    question_key = normalize_for_match(question)
    match = re.search(
        rf"(?<![a-z0-9]){re.escape(_row_entity_key(entity))}(?![a-z0-9])\s*(?:在|on)\s*"
        r"(?P<group>oracle\s+setting|closed\s+book\s+setting|open\s+book\s+setting|"
        r"(?:test|dev|validation|evaluation|setting|configuration|split)\s*\([^)]{1,48}\))",
        question_key,
        re.IGNORECASE,
    )
    if match is None:
        return None
    return re.sub(r"[\s|]+", "", match.group("group")).casefold()


def _dataset_label_matches(value: str, target: str) -> bool:
    """Match a PDF-exported dataset label, including split-cell typos."""

    left = re.sub(r"[^a-z0-9]", "", normalize_for_match(value))
    right = re.sub(r"[^a-z0-9]", "", normalize_for_match(target))
    if not left or not right:
        return False
    if left == right:
        return True
    threshold = 0.9 if "+" in str(value) or "+" in str(target) else 0.72
    return min(len(left), len(right)) >= 4 and SequenceMatcher(None, left, right).ratio() >= threshold


def _table_question_dataset_targets(question: str, table_content: str) -> list[str]:
    """Return dataset qualifiers that can disambiguate repeated table rows."""

    parsed = parse_markdown_table(table_content)
    if parsed is None:
        return []
    headers, rows = parsed
    dataset_idx = next(
        (
            index
            for index, header in enumerate(headers)
            if re.search(r"dataset|data\s+set|setting", normalize_for_match(header), re.IGNORECASE)
        ),
        None,
    )
    if dataset_idx is None:
        return []
    relation_targets = _question_relation_qualifiers(question)
    if relation_targets:
        return relation_targets
    candidates = extract_query_entities(question)
    values = [row[dataset_idx] for row in rows if dataset_idx < len(row)]
    return [
        candidate
        for candidate in candidates
        if any(_dataset_label_matches(value, candidate) for value in values)
    ]


def _table_data_rows_with_context(
    question: str,
    table_content: str,
) -> list[tuple[str, str | None, str | None]]:
    """Pair rows with setting and requested dataset context."""

    parsed = parse_markdown_table(table_content)
    if parsed is None:
        return []
    headers, _ = parsed
    dataset_idx = next(
        (
            index
            for index, header in enumerate(headers)
            if re.search(r"dataset|data\s+set|setting", normalize_for_match(header), re.IGNORECASE)
        ),
        None,
    )
    targets = _table_question_dataset_targets(question, table_content)
    current_dataset: str | None = None
    result: list[tuple[str, str | None, str | None]] = []
    for row, group in _table_data_rows_with_groups(table_content):
        if dataset_idx is not None and targets:
            cells = _split_markdown_row(row)
            if dataset_idx < len(cells):
                value = display_table_cell(cells[dataset_idx])
                for target in targets:
                    if _dataset_label_matches(value, target):
                        current_dataset = target
                        break
        result.append((row, group, current_dataset))
    return result


def _table_usage_group(question: str, table_content: str) -> str | None:
    """Return a first-column method group named after a usage verb."""

    usage = re.search(r"使用|采用|using|with", question, re.IGNORECASE)
    if usage is None:
        return None
    grouped_rows = _table_data_rows_with_groups(table_content)
    method_labels = {
        display_table_cell(_split_markdown_row(row)[0])
        for row, _group in grouped_rows
        if _split_markdown_row(row)
        and display_table_cell(_split_markdown_row(row)[0])
        and not _looks_numeric_cell(display_table_cell(_split_markdown_row(row)[0]))
    }
    method_labels.update(
        group.removeprefix("section:")
        for _row, group in grouped_rows
        if group and group.startswith("section:")
    )
    suffix = question[usage.end() :]
    matches: list[tuple[int, str]] = []
    suffix_key = normalize_for_match(suffix)
    for label in sorted(method_labels):
        candidates = [label]
        # Ablation/group rows are often rendered as ``w SciDC`` while the
        # question names the method simply as ``SciDC``.
        stripped = re.sub(r"^w(?:/o)?\s+", "", label, flags=re.IGNORECASE)
        if stripped != label:
            candidates.append(stripped)
        if any(_question_mentions_label(suffix, candidate) for candidate in candidates):
            positions = [
                suffix_key.find(normalize_for_match(candidate))
                for candidate in candidates
                if normalize_for_match(candidate) in suffix_key
            ]
            matches.append((min(positions, default=len(suffix_key)), label))
    return min(matches, key=lambda item: (item[0], -len(item[1])))[1] if matches else None


def _table_data_rows_with_leading_groups(
    question: str,
    table_content: str,
) -> list[tuple[str, str | None, str | None, str | None]]:
    """Carry a non-empty first-column label through blank continuation rows."""

    usage_group = _table_usage_group(question, table_content)
    current: str | None = None
    result = []
    for row, group, dataset in _table_data_rows_with_context(question, table_content):
        cells = _split_markdown_row(row)
        first = display_table_cell(cells[0]) if cells else ""
        if group and group.startswith("section:"):
            current = group.removeprefix("section:")
        elif first and not _looks_numeric_cell(first):
            current = first
        result.append((row, group, dataset, current if usage_group else None))
    return result


def _best_section_group(
    entities: Iterable[str],
    rows: Iterable[tuple[str, str | None, str | None]],
) -> str | None:
    """Choose a section that contains the largest complete set of entities."""

    targets = list(entities)
    coverage: dict[str, set[str]] = {}
    for row, group, _ in rows:
        if not group or not group.startswith("section:"):
            continue
        for entity in targets:
            if _row_entity_label(row, entity) is not None:
                coverage.setdefault(group, set()).add(_row_entity_key(entity))
    if not coverage:
        return None
    best_group, best_entities = max(coverage.items(), key=lambda item: len(item[1]))
    return best_group if len(best_entities) > 1 else None


def select_row_entity(question: str, table_content: str) -> str:
    """Choose the question entity that appears in the applicable table rows."""

    contextual_entities = _table_question_entities(question, table_content)
    if contextual_entities:
        return contextual_entities[0]
    grouped_rows = _table_data_rows_with_groups(table_content)
    requested_group = _table_question_group(question)
    known_groups = {group for _, group in grouped_rows if group is not None}
    if requested_group and not known_groups:
        # A query that names a section must not fall through to an unrelated
        # table chunk with the same Table N from another paper.
        grouped_rows = []
    elif requested_group and known_groups:
        grouped_rows = [
            (row, group) for row, group in grouped_rows if group == requested_group
        ]
    candidates = sorted(extract_query_entities(question), key=len, reverse=True)
    for candidate in candidates:
        if len(_row_entity_key(candidate)) < 2:
            continue
        if any(_row_entity_label(row, candidate) is not None for row, _ in grouped_rows):
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
    header_line = _repair_wrapped_header_line(lines[separator_idx - 1])
    header = [_clean_header_cell(cell) for cell in _split_markdown_row(header_line)]
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
    table_numbers = table_numbers_from_question(question)
    if not table_numbers:
        return table_indices
    return [
        idx
        for idx in table_indices
        if table_number_from_metadata(metas[idx]) in table_numbers
    ]


def _match_table_column(question: str, headers: list[str]) -> tuple[int, str] | None:
    """Find the header requested by a question, including common aliases."""

    normalized_question = normalize_for_match(question)
    normalized_headers = [normalize_for_match(header) for header in headers]

    # Reuse the multi-column resolver when it yields one unambiguous metric.
    # This is important for repeated qualified headers such as Darcy smooth
    # L2 versus Darcy rough L2; a bare ``l2`` alias must not silently select
    # the first column.
    requested_columns = _requested_table_columns(question, headers)
    if len(requested_columns) == 1:
        index = requested_columns[0]
        return index, normalized_headers[index]

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
                if canonical in {"l2", "h1"} and header != canonical:
                    continue
                for alias in aliases:
                    normalized_alias = normalize_for_match(alias)
                    if normalized_alias and normalized_alias in normalized_question:
                        alias_matches.append((idx, header, len(normalized_alias)))
    if alias_matches:
        idx, header, _ = max(alias_matches, key=lambda item: item[2])
        return idx, header
    return None


def _requested_table_columns(question: str, headers: list[str]) -> list[int]:
    """Return every table column explicitly named by a multi-value question."""

    normalized_question = normalize_for_match(question)
    compact_question = re.sub(r"\s+", "", normalized_question)
    metric_re = re.compile(
        r"(?<![a-z0-9])(?:l2|h1|f1(?:-score)?|mse|meteor|rouge-?\d+|berts?)(?![a-z0-9])",
        re.IGNORECASE,
    )
    qualifier_re = re.compile(
        r"rough|smooth|multiscale|helmholtz|full[- ]?text|rag|caption|representation|"
        r"has[- ]?answer|miss[- ]?answer|internal\s+knowledge|表格|题注",
        re.IGNORECASE,
    )
    question_metrics = set(metric_re.findall(normalized_question))
    question_qualifiers = set(qualifier_re.findall(normalized_question.casefold()))
    direct_requested: list[int] = []
    fallback_requested: list[int] = []
    for idx, raw_header in enumerate(headers):
        header = normalize_for_match(raw_header)
        clean_header = normalize_for_match(_clean_header_cell(raw_header))
        candidates = {value for value in (header, clean_header) if value}
        # A question may name a dataset column while the PDF header adds a
        # metric suffix, e.g. ``PopQA (acc)``.  Match the distinctive leading
        # label without relaxing metric/setting disambiguation below.
        header_stem = re.split(r"\s*[([<]", clean_header, maxsplit=1)[0].strip()
        if header_stem and re.search(
            rf"(?<![a-z0-9]){re.escape(header_stem)}(?![a-z0-9])",
            normalized_question,
            re.IGNORECASE,
        ):
            direct_requested.append(idx)
            continue
        header_terms = set(re.findall(r"[a-z][a-z0-9@+\-]*", clean_header.casefold()))
        question_terms = set(re.findall(r"[a-z][a-z0-9@+\-]*", normalized_question.casefold()))
        if len(header_terms) >= 2 and header_terms <= question_terms:
            direct_requested.append(idx)
            continue
        # Reuse the semantic aliases used by the single-column matcher.
        for canonical, aliases in TABLE_COLUMN_ALIASES.items():
            if header == canonical or header in canonical or canonical in header:
                # ``l2``/``h1`` are metric tokens, not sufficient evidence to
                # choose one of several qualified columns such as Darcy
                # smooth/rough/multiscale.  Their context-aware fallback
                # below handles those columns; keep aliases for exact or
                # genuinely semantic headers such as ``FT (full-text)``.
                if canonical not in {"l2", "h1"} or header == canonical:
                    candidates.update(normalize_for_match(alias) for alias in aliases)
        compact_candidates = {re.sub(r"\s+", "", candidate) for candidate in candidates}
        if any(
            candidate and (
                candidate in normalized_question or candidate in compact_question
            )
            for candidate in candidates | compact_candidates
        ):
            direct_requested.append(idx)
            continue

        # Headers such as ``Darcy rough L2`` contain a dataset qualifier and
        # a metric, while the query may mention only the metric plus the
        # qualifier.  Select the metric column only when the qualifier is
        # shared; this avoids returning all three L2 columns in a Darcy table.
        header_metrics = set(metric_re.findall(clean_header))
        header_qualifiers = set(qualifier_re.findall(clean_header.casefold()))
        if question_metrics & header_metrics and (
            not header_qualifiers or question_qualifiers & header_qualifiers
        ):
            fallback_requested.append(idx)
    repeated_question_qualifiers = {
        qualifier
        for qualifier in question_qualifiers
        if sum(
            qualifier.casefold() in normalize_for_match(header).casefold()
            for header in headers
        ) > 1
    }
    if direct_requested and repeated_question_qualifiers:
        qualified_requested = [
            index
            for index in direct_requested
            if repeated_question_qualifiers
            & set(qualifier_re.findall(normalize_for_match(headers[index])))
        ]
        if qualified_requested:
            direct_requested = qualified_requested
    if not direct_requested and question_qualifiers:
        question_tokens = set(re.findall(r"[a-z][a-z0-9-]*", normalized_question.casefold()))
        qualified_requested = []
        for index, raw_header in enumerate(headers):
            header = normalize_for_match(raw_header)
            header_qualifiers = set(qualifier_re.findall(header.casefold()))
            if not question_qualifiers & header_qualifiers:
                continue
            tail = header.casefold()
            for qualifier in header_qualifiers:
                tail = tail.replace(qualifier.casefold(), " ")
            tail_tokens = set(re.findall(r"[a-z][a-z0-9-]*", tail))
            if tail_tokens & question_tokens:
                qualified_requested.append(index)
        if qualified_requested:
            direct_requested = qualified_requested
    # Prefer explicitly named semantic aliases (for example ``overall F1`` →
    # ``all F1``) over the broad metric-token fallback, which would otherwise
    # return every repeated F1 column in a multi-setting table.
    return direct_requested or fallback_requested


def _select_table_value_for_question(value: Any, question: str) -> str:
    """Select a multi-line table value when the question names its variant."""

    raw = str(value or "")
    variants = [part for part in re.split(r"<br\s*/?>|\n", raw, flags=re.IGNORECASE) if part.strip()]
    if len(variants) > 1:
        if re.search(r"表格表示|table\s+representations?|representations?", question or "", re.IGNORECASE):
            return display_table_cell(variants[-1])
        if re.search(r"题注|table\s+captions?|captions?", question or "", re.IGNORECASE):
            return display_table_cell(variants[0])
        # A line break inside an entity/list cell is usually a wrapped value,
        # not two alternatives.  Keep numeric alternatives distinguishable
        # above, but join ordinary text such as ``ADRB1,<br>ADRB2`` naturally.
        if not all(re.match(r"\s*[<>≤≥-]?\d", part) for part in variants):
            return display_table_cell(re.sub(r"<br\s*/?>", " ", raw, flags=re.IGNORECASE))
    return display_table_cell(raw)


def _question_requests_multiple_rows(question: str) -> bool:
    """Whether a query asks for more than one row/entity value."""

    return bool(
        re.search(
            r"分别|各自|两者|both|respectively|each|\band\b|\bor\b",
            question or "",
            re.IGNORECASE,
        )
    )


def _row_entity_exactish(row: str, target: str) -> str | None:
    """Match an entity as a token, avoiding ``MgNO-high-in`` false matches."""

    target_key = _row_entity_key(target)
    if not target_key:
        return None
    for cell in _split_markdown_row(row):
        display = display_table_cell(cell)
        normalized = _row_entity_key(display)
        if not normalized or _looks_numeric_cell(display):
            continue
        if normalized == target_key:
            return display
        boundary = rf"(?:^|[\s,(*]){re.escape(target_key)}(?:$|[\s,*)])"
        if re.search(boundary, normalized):
            return display
    return None


def _row_matches_requested_variant(row: str, target: str, question: str) -> str | None:
    """Match a repeated row only when its distinguishing variant is requested."""

    display = _row_entity_exactish(row, target)
    if display is None:
        return None
    target_key = _row_entity_key(target)
    row_key = _row_entity_key(display)
    if row_key == target_key:
        return display

    # Compare the descriptor surrounding the shared entity.  The aliases are
    # intentionally small and generic: they cover common baseline/layer
    # ablations without hard-coding a paper or a particular row name.
    descriptor = row_key.replace(target_key, "", 1).strip(" ,()")
    question_key = _row_entity_key(question)
    question_without_table = re.sub(r"table\s*\d+|表\s*\d+", "", question_key)
    if "baseline" in descriptor and (
        "baseline" in question_without_table or "基线" in question
    ):
        return display
    layer_number = re.search(r"\b(\d+)\s*(?:layers?|levels?)\b", descriptor)
    if layer_number:
        number = layer_number.group(1)
        chinese_digits = {
            "0": "零",
            "1": "一",
            "2": "二",
            "3": "三",
            "4": "四",
            "5": "五",
            "6": "六",
            "7": "七",
            "8": "八",
            "9": "九",
        }
        if number in question_without_table or chinese_digits.get(number, "") in question:
            return display
    if any(token in question_without_table for token in ("without", "w/o", "无", "去除")):
        if any(token in descriptor for token in ("without", "w/o", "boundary")):
            return display
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
    headers, _ = parsed
    column_match = _match_table_column(question, headers)
    if column_match is None:
        return None
    column_idx, column_header = column_match
    entity = select_row_entity(question, table_content)
    if not entity:
        return None
    grouped_rows = _table_data_rows_with_leading_groups(question, table_content)
    requested_group = _table_question_group(question)
    known_groups = {group for _, group, _, _leading in grouped_rows if group is not None}
    requested_datasets = _table_question_dataset_targets(question, table_content)
    entity_group = _table_question_group_for_entity(question, entity)
    usage_group = _table_usage_group(question, table_content)
    for raw_row, group, dataset, leading_group in grouped_rows:
        if requested_group and known_groups and group != requested_group:
            continue
        if entity_group and group != entity_group:
            continue
        if requested_datasets and dataset not in requested_datasets:
            continue
        if usage_group and _row_entity_key(leading_group) != _row_entity_key(usage_group):
            continue
        row = _split_markdown_row(raw_row)
        if not row:
            continue
        matched_label = _row_entity_label(_join_markdown_cells(row), entity)
        if matched_label is None:
            continue
        if column_idx >= len(row):
            return None
        number = table_number_from_metadata(metadata or {})
        return {
            "table_number": str(number) if number is not None else "",
            "row": matched_label,
            "column": display_table_cell(headers[column_idx]),
            "value": _select_table_value_for_question(row[column_idx], question),
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

    parsed = parse_markdown_table(table_content)
    if parsed is None:
        return None
    headers, _ = parsed
    entities = _table_question_entities(question, table_content)
    if not entities:
        entity = select_row_entity(question, table_content)
        entities = [entity] if entity else []
    if not entities:
        return None
    requested_columns = _requested_table_columns(question, headers)
    entity_occurrences = max(
        len(re.findall(re.escape(_row_entity_key(entity)), _row_entity_key(question)))
        for entity in entities
    )
    multiple_rows_requested = len(entities) > 1 or (
        _question_requests_multiple_rows(question)
        and (entity_occurrences >= 2 or not requested_columns)
    )
    if len(requested_columns) == 1 and not multiple_rows_requested:
        # A single explicitly named metric is a deterministic cell lookup;
        # returning the whole row would change the public answer contract.
        return None
    if (
        not TABLE_ROW_VALUE_QUESTION_RE.search(question or "")
        and len(requested_columns) < 2
        and not multiple_rows_requested
    ):
        return None
    grouped_rows = _table_data_rows_with_leading_groups(question, table_content)
    requested_group = _table_question_group(question)
    known_groups = {group for _, group, _, _leading in grouped_rows if group is not None}
    requested_datasets = _table_question_dataset_targets(question, table_content)
    entity_groups = {
        _table_question_group_for_entity(question, entity)
        for entity in entities
    }
    section_group = _table_question_section(question, table_content)
    if section_group is None and not requested_group and not any(entity_groups):
        section_group = _best_section_group(
            entities, [(row, group, dataset) for row, group, dataset, _leading in grouped_rows]
        )
    matched_results: list[dict[str, Any]] = []
    usage_group = _table_usage_group(question, table_content)
    active_outer_entity: str | None = None
    for raw_row, group, dataset, leading_group in grouped_rows:
        row = _split_markdown_row(raw_row)
        if not row:
            continue
        row_text = _join_markdown_cells(row)
        first_label = display_table_cell(row[0])
        if (
            first_label
            and not _looks_numeric_cell(first_label)
            and not re.match(r"^(?:w(?:/o)?\b|∆)", first_label, re.IGNORECASE)
        ):
            active_outer_entity = first_label
        for entity in entities:
            matched_label = _row_entity_label(row_text, entity)
            usage_row = usage_group and _row_entity_label(row_text, usage_group)
            if matched_label is None:
                if usage_row and active_outer_entity and _row_entity_label(
                    active_outer_entity, entity
                ):
                    matched_label = usage_row
                else:
                    continue
            entity_group = _table_question_group_for_entity(question, entity)
            if requested_group and known_groups and group != requested_group:
                continue
            if section_group and group != section_group:
                continue
            if entity_group and group != entity_group:
                continue
            if requested_datasets and dataset not in requested_datasets:
                continue
            if usage_group and not (
                _row_entity_key(leading_group) == _row_entity_key(usage_group)
                or usage_row
            ):
                continue
            if multiple_rows_requested and len(entities) == 1:
                # Collect all rows that contain the requested entity as a token.
                # This handles questions such as “baseline MgNO and six-layer
                # MgNO” while excluding similarly prefixed rows like MgNO-fno.
                matched_label = _row_matches_requested_variant(
                    row_text, entity, question
                )
                if matched_label is None:
                    continue
            values: list[dict[str, str]] = []
            column_values = requested_columns or list(range(1, len(row)))
            for column_idx in column_values:
                if column_idx >= len(headers):
                    break
                value = row[column_idx] if column_idx < len(row) else ""
                values.append(
                    {
                        "column": display_table_cell(headers[column_idx]),
                        "value": _select_table_value_for_question(value, question),
                    }
                )
            if not values:
                continue
            if multiple_rows_requested:
                matched_results.append({"row": matched_label, "values": values})
                continue
            number = table_number_from_metadata(metadata or {})
            return {
                "table_number": str(number) if number is not None else "",
                "row": matched_label,
                "values": values,
            }
    if multiple_rows_requested and matched_results:
        number = table_number_from_metadata(metadata or {})
        return {
            "table_number": str(number) if number is not None else "",
            "rows": matched_results,
        }
    return None


def find_table_cell_in_chunks(
    question: str,
    texts: list[str],
    metas: list[dict[str, Any]],
) -> tuple[int, dict[str, Any]] | None:
    """Find a requested table cell without crossing an explicit table boundary."""

    if is_derived_value_question(question):
        return None

    # Without an explicit table number, the same row/column may legitimately
    # occur in several tables.  Leave that ambiguous case to the normal
    # retrieval/generation path instead of silently choosing the first table.
    if table_number_from_question(question) is None:
        return None
    for idx in matching_table_indices(question, texts, metas):
        row_values = extract_table_row_values(question, texts[idx], metas[idx])
        if row_values is not None:
            return idx, row_values
        cell = extract_table_cell(question, texts[idx], metas[idx])
        if cell is not None:
            return idx, cell
    return None


def filter_table_rows_by_entity(content: str, entity: str) -> str | None:
    """Keep the header/separator and rows whose entity appears in any cell.

    Spanning group-marker rows are retained as well.  They carry the section
    context needed to distinguish repeated entities such as the same model
    evaluated under two test settings.
    """

    lines = content.splitlines()
    separator_idx = next(
        (idx for idx, line in enumerate(lines) if TABLE_SEPARATOR_RE.fullmatch(line.strip())),
        None,
    )
    if separator_idx is None:
        header, data = lines[:1], lines[1:]
    else:
        header, data = lines[: separator_idx + 1], lines[separator_idx + 1 :]
    matched_rows = []
    pending_group: str | None = None
    emitted_group: str | None = None
    for row in data:
        if not (row.strip().startswith("|") and row.strip().endswith("|")):
            continue
        if _table_row_group_marker(row) is not None or _is_table_section_marker(row):
            pending_group = row
            continue
        if _row_entity_label(row, entity) is not None:
            if pending_group is not None and pending_group != emitted_group:
                matched_rows.append(pending_group)
                emitted_group = pending_group
            matched_rows.append(row)
    if not matched_rows:
        return None
    return "\n".join(header + matched_rows)


def is_table_question(question: str) -> bool:
    return bool(TABLE_QUESTION_RE.search(question or ""))


def is_derived_value_question(question: str) -> bool:
    """Whether a question explicitly asks for a value derived from evidence."""

    return bool(DERIVED_VALUE_QUESTION_RE.search(question or ""))


def is_comparative_table_question(question: str) -> bool:
    """Whether a table question needs multiple rows for comparison."""

    return bool(TABLE_COMPARISON_QUESTION_RE.search(question or ""))


def table_number_from_question(question: str) -> str | None:
    match = TABLE_NUMBER_RE.search(question or "")
    return match.group(1) if match else None


def table_numbers_from_question(question: str) -> tuple[int, ...]:
    """Return distinct explicit table numbers in question order."""

    return tuple(
        dict.fromkeys(int(match.group(1)) for match in TABLE_NUMBER_RE.finditer(question or ""))
    )


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
            if (
                not is_comparative_table_question(question)
                and not is_derived_value_question(question)
                and _table_usage_group(question, working_texts[table_idx[0]]) is None
                and len(_table_question_entities(question, working_texts[table_idx[0]])) <= 1
                and not _table_question_dataset_targets(question, working_texts[table_idx[0]])
            ):
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
