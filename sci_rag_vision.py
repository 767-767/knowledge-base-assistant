"""Small, side-effect-free helpers for opt-in PDF figure questions."""

from __future__ import annotations

import base64
import hashlib
import re
from pathlib import Path

import pymupdf


CAPTION_RE = re.compile(
    r"^\s*(?P<extended>Extended\s+Data\s+)?(?:Figure|Fig\.)\s*(?P<number>\d+)\b",
    re.IGNORECASE,
)
VISION_SYSTEM = "Answer only from the supplied figure. Be concise and state the visual evidence."
VISION_INSTRUCTION = "Inspect the supplied figure and answer the question directly. Do not guess beyond what is visible."


def _caption_blocks(blocks: list[tuple[object, ...]]) -> list[dict[str, object]]:
    parsed = []
    for block in blocks:
        if len(block) < 7 or int(block[6]) != 0:
            continue
        x0, y0, x1, y1 = (float(block[i]) for i in range(4))
        text = " ".join(str(block[4]).split())
        parsed.append({"x0": x0, "y0": y0, "x1": x1, "y1": y1, "text": text, "match": CAPTION_RE.match(text)})
    return parsed


def figure_clip(
    blocks: list[tuple[object, ...]],
    page_width: float,
    page_height: float,
    reference: tuple[str, int],
) -> tuple[float, float, float, float, str]:
    """Return a full-width figure region ending immediately above its caption."""

    captions = [block for block in _caption_blocks(blocks) if block["match"]]
    target_kind, target_number = reference
    target = None
    for block in captions:
        match = block["match"]
        assert match is not None
        kind = "extended_data_figure" if match.group("extended") else "figure"
        if kind == target_kind and int(match.group("number")) == target_number:
            target = block
            break
    if target is None:
        raise ValueError(f"figure caption not found: {target_kind} {target_number}")

    previous_bottom = page_height * 0.05
    for block in captions:
        if block is target or block["y1"] > target["y0"]:
            continue
        if min(block["x1"], target["x1"]) > max(block["x0"], target["x0"]):
            previous_bottom = max(previous_bottom, float(block["y1"]))
    top = max(page_height * 0.05, previous_bottom, float(target["y0"]) - min(420.0, page_height * 0.55))
    bottom = float(target["y0"])
    if bottom <= top:
        raise ValueError(f"empty figure region: {target_kind} {target_number}")
    return (0.0, top, float(page_width), bottom, str(target["text"]))


def detail_clip(clip: tuple[float, float, float, float, str]) -> tuple[float, float, float, float]:
    """Return one generic lower-center detail crop for a figure region."""

    x0, y0, x1, y1, _ = clip
    width = x1 - x0
    height = y1 - y0
    return (x0 + width * 0.10, y0 + height * 0.35, x1 - width * 0.10, y1)


def _render_clip(page: pymupdf.Page, clip: tuple[float, float, float, float]) -> dict[str, object]:
    pixmap = page.get_pixmap(matrix=pymupdf.Matrix(2, 2), clip=pymupdf.Rect(*clip), alpha=False)
    png = pixmap.tobytes("png")
    return {
        "data_url": "data:image/png;base64," + base64.b64encode(png).decode("ascii"),
        "clip": [round(value, 2) for value in clip],
        "png_sha256": hashlib.sha256(png).hexdigest(),
        "width": pixmap.width,
        "height": pixmap.height,
    }


def figure_page(document: pymupdf.Document, reference: tuple[str, int]) -> int:
    candidates: list[tuple[int, int]] = []
    for page_number, page in enumerate(document, start=1):
        blocks = page.get_text("blocks", sort=True)
        captions = [block for block in _caption_blocks(blocks) if block["match"]]
        for block in captions:
            match = block["match"]
            assert match is not None
            kind = "extended_data_figure" if match.group("extended") else "figure"
            if kind == reference[0] and int(match.group("number")) == reference[1]:
                image_above = any(
                    len(candidate) >= 7
                    and int(candidate[6]) == 1
                    and float(candidate[3]) <= float(block["y0"])
                    and min(float(candidate[2]), float(block["x1"]))
                    > max(float(candidate[0]), float(block["x0"]))
                    for candidate in blocks
                )
                marker = match.group(0).casefold()
                score = (2 if image_above else 0) + (1 if "figure" in marker else 0)
                candidates.append((score, page_number))
    if candidates:
        return max(candidates, key=lambda item: item[0])[1]
    raise ValueError(f"figure caption not found: {reference[0]} {reference[1]}")


def render_figure(
    pdf_path: str | Path,
    reference: tuple[str, int],
    page_number: int | None = None,
    include_detail: bool = True,
) -> dict[str, object]:
    document = pymupdf.open(pdf_path)
    try:
        page_number = page_number or figure_page(document, reference)
        page = document[page_number - 1]
        clip = figure_clip(page.get_text("blocks", sort=True), page.rect.width, page.rect.height, reference)
        image = _render_clip(page, clip[:4])
        image.update({"caption": clip[4], "page": page_number})
        if include_detail:
            image["detail"] = _render_clip(page, detail_clip(clip))
        return image
    finally:
        document.close()


def vision_messages(question: str, data_url: str, detail_data_url: str | None = None) -> list[dict[str, object]]:
    content: list[dict[str, object]] = [
        {"type": "text", "text": f"{VISION_INSTRUCTION}\n\nQuestion: {question}"},
        {"type": "text", "text": "Image 1 is the full figure crop."},
        {"type": "image_url", "image_url": {"url": data_url, "detail": "original"}},
    ]
    if detail_data_url is not None:
        content.extend(
            [
                {"type": "text", "text": "Image 2 is a generic lower-center detail crop; use it to inspect small visual details."},
                {"type": "image_url", "image_url": {"url": detail_data_url, "detail": "original"}},
            ]
        )
    return [{"role": "system", "content": VISION_SYSTEM}, {"role": "user", "content": content}]


def complete_vision(client: object, model: str, messages: list[dict[str, object]]) -> str:
    """Return the first non-empty vision response, allowing one empty-output retry."""

    for _ in range(2):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
            max_tokens=512,
        )
        answer = response.choices[0].message.content or ""
        if answer.strip():
            return answer
    raise RuntimeError("empty vision response")
