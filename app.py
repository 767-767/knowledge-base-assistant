"""个人知识库助手应用入口。

Importing this module is intentionally side-effect free.  Models, the OpenAI
client, ChromaDB, and Gradio are created only by :func:`create_runtime` or
when the UI is launched from ``main``.  The parsing and table logic lives in
``sci_rag_core.py`` so it can be tested offline.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import html
import json
import os
from pathlib import Path
import random
import re
import shutil
from typing import Any, Callable
from urllib.parse import urlsplit

from sci_rag_core import (
    Chunk,
    build_evidence_ledger,
    extract_spatial_figure_chunks,
    file_sha256,
    formula_evidence_indices,
    find_table_cell_in_chunks,
    figure_reference_from_question,
    is_formula_question,
    is_limitation_question,
    is_table_question,
    limitation_evidence_indices,
    missing_pdf_formula_blocks,
    matching_figure_indices,
    matching_table_indices,
    normalize_for_match,
    rerank_table_first,
    split_to_chunks,
    supplement_answer_with_evidence,
    supplement_formula_with_evidence,
    table_number_from_question,
    validate_answer_against_evidence,
)
from sci_rag_reranking import CrossEncoderReranker, reranker_document_text
from sci_rag_retrieval import (
    BM25Index,
    DocumentRoute,
    DocumentRouter,
    RankedItem,
    ensure_source_coverage,
    query_variants,
    reciprocal_rank_fusion,
    tokenize,
)
from sci_rag_vision import complete_vision, render_figure, vision_messages


APP_DISPLAY_NAME = "个人知识库助手"


MODEL_SERVICE_PRESETS = {
    "Ollama（本地）": ("http://localhost:11434/v1", "qwen3:4b-instruct"),
    "DeepSeek": ("https://api.deepseek.com", "deepseek-flash"),
    "Gemini": (
        "https://generativelanguage.googleapis.com/v1beta/openai/",
        "gemini-2.5-flash-lite",
    ),
    "自定义 OpenAI 兼容服务": ("", ""),
}

DOCUMENT_BATCH_SIZE = 64


APP_CSS = """
:root {
    --kb-primary: #2563eb;
    --kb-primary-hover: #1d4ed8;
    --kb-nav-active: #2563eb;
    --kb-accent: #168bff;
    --kb-accent-hover: #0876e8;
    --kb-accent-soft: #dbeafe;
    --kb-text: #10204a;
    --kb-muted: #62749a;
    --kb-canvas: #eef6ff;
    --kb-surface: #ffffff;
    --kb-surface-muted: #edf5ff;
    --kb-border: #c9dcff;
    --kb-sidebar: #f7faff;
    --kb-sidebar-raised: #ffffff;
    --kb-sidebar-text: #10204a;
    --kb-sidebar-muted: #53678f;
    --kb-danger: #dc2626;
    --kb-shadow: 0 18px 48px rgba(37, 99, 235, 0.08);
}

.dark {
    --kb-primary: #2563eb;
    --kb-primary-hover: #1d4ed8;
    --kb-nav-active: #2563eb;
    --kb-accent: #168bff;
    --kb-accent-hover: #0876e8;
    --kb-accent-soft: #dbeafe;
    --kb-text: #10204a;
    --kb-muted: #62749a;
    --kb-canvas: #eef6ff;
    --kb-surface: #ffffff;
    --kb-surface-muted: #edf5ff;
    --kb-border: #c9dcff;
    --kb-sidebar: #f7faff;
    --kb-sidebar-raised: #ffffff;
    --kb-sidebar-text: #10204a;
    --kb-sidebar-muted: #53678f;
    --kb-danger: #dc2626;
    --kb-shadow: 0 18px 48px rgba(37, 99, 235, 0.08);
}

.gradio-container {
    width: 100% !important;
    max-width: none !important;
    min-height: 100dvh;
    padding: 0 !important;
    background:
        radial-gradient(circle at 18% 16%, rgba(22, 139, 255, 0.11), transparent 30%),
        linear-gradient(135deg, #f8fbff 0%, var(--kb-canvas) 48%, #f7fbff 100%) !important;
    color: var(--kb-text);
    font-family: "PingFang SC", "Microsoft YaHei", ui-sans-serif, -apple-system,
        BlinkMacSystemFont, "Segoe UI", sans-serif;
}

.gradio-container .main {
    width: 100%;
    max-width: none !important;
    margin: 0 !important;
    padding: 0 !important;
    gap: 0 !important;
}

#kb-header {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    z-index: 70;
    margin: 0;
    padding: 0 !important;
    border: 0 !important;
    border-radius: 0 !important;
    background: transparent !important;
    overflow: visible !important;
}

.kb-header {
    display: flex;
    min-height: 72px;
    align-items: center;
    justify-content: space-between;
    gap: 20px;
    margin-left: 264px;
    padding: 12px clamp(24px, 3vw, 46px);
    border-bottom: 1px solid var(--kb-border);
    background: rgba(255, 255, 255, 0.76);
}

.kb-brand {
    position: fixed;
    top: 0;
    left: 0;
    z-index: 61;
    display: flex;
    align-items: center;
    width: 264px;
    min-width: 264px;
    min-height: 96px;
    gap: 14px;
    padding: 20px 24px;
    background: var(--kb-sidebar);
}

.kb-mark {
    display: grid;
    width: 34px;
    height: 38px;
    flex: 0 0 34px;
    place-items: center;
    border: 1px solid rgba(255, 255, 255, 0.72);
    border-radius: 11px;
    color: #ffffff;
    background: linear-gradient(135deg, #2563eb 0%, #168bff 100%);
    box-shadow: 0 10px 24px rgba(37, 99, 235, 0.24);
    font-size: 17px;
    font-weight: 700;
}

.dark .kb-mark {
    color: #ffffff !important;
    background: linear-gradient(135deg, #2563eb 0%, #168bff 100%) !important;
}

.kb-title {
    margin: 0;
    color: var(--kb-sidebar-text);
    font-size: 19px;
    font-weight: 700;
    letter-spacing: 0.06em;
    line-height: 1.25;
}

.kb-subtitle {
    margin: 2px 0 0;
    overflow: hidden;
    color: var(--kb-sidebar-muted);
    font-size: 10px;
    letter-spacing: 0.06em;
    line-height: 1.35;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.kb-crumb {
    color: var(--kb-muted);
    font-size: 11px;
    letter-spacing: 0.08em;
}

.kb-crumb strong {
    color: var(--kb-text);
    font-weight: 620;
}

#kb-workspace {
    display: block !important;
    width: 100%;
    min-height: 100dvh;
    margin: 0;
}

#kb-workspace > .tab-wrapper {
    position: fixed;
    top: 96px;
    bottom: 0;
    left: 0;
    z-index: 55;
    display: flex !important;
    align-self: start;
    align-items: stretch;
    flex-direction: column;
    gap: 10px;
    width: 264px;
    height: calc(100dvh - 96px) !important;
    margin: 0;
    padding: 20px 16px 96px !important;
    border: 0 !important;
    border-right: 1px solid var(--kb-border) !important;
    border-radius: 0 !important;
    background: var(--kb-sidebar) !important;
    overflow-y: auto;
}

#kb-workspace > .tab-wrapper > .tab-container[role="tablist"] {
    display: flex;
    flex: 0 0 auto;
    flex-direction: column;
    gap: 5px;
    height: auto !important;
    overflow: visible !important;
}

#kb-workspace > .tab-wrapper > .tab-container.visually-hidden button {
    width: 1px !important;
    min-width: 0 !important;
    padding-inline: 0 !important;
}

#kb-workspace > .tab-wrapper [role="tab"],
#kb-workspace > .tab-wrapper .overflow-menu > button {
    justify-content: flex-start;
    width: 100%;
    min-height: 46px;
    padding: 11px 15px !important;
    border: 0 !important;
    border-radius: 12px !important;
    color: var(--kb-sidebar-muted) !important;
    -webkit-text-fill-color: var(--kb-sidebar-muted) !important;
    font-size: 13px !important;
    font-weight: 560 !important;
    text-align: left;
}

#kb-workspace > .tab-wrapper [role="tab"]:hover,
#kb-workspace > .tab-wrapper .overflow-menu > button:hover {
    color: var(--kb-primary) !important;
    -webkit-text-fill-color: var(--kb-primary) !important;
    background: rgba(37, 99, 235, 0.08) !important;
}

#kb-workspace > .tab-wrapper [role="tab"].selected {
    border-bottom: 0 !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    background: linear-gradient(135deg, #2563eb 0%, #168bff 100%) !important;
    font-weight: 680 !important;
    box-shadow: none !important;
}

#kb-workspace > .tab-wrapper [role="tab"].selected::before,
#kb-workspace > .tab-wrapper [role="tab"].selected::after {
    display: none !important;
    content: none !important;
}

#kb-workspace > .tabitem {
    min-width: 0;
    width: calc(100% - 264px);
    max-width: none;
    margin: 0 0 0 264px;
    padding: 108px clamp(24px, 3vw, 48px) 48px !important;
}

.kb-page-head {
    margin: 0 0 26px;
    padding-bottom: 20px;
    border-bottom: 1px solid rgba(201, 220, 255, 0.8);
}

.kb-page-head h1 {
    margin: 0;
    color: var(--kb-text);
    font-size: clamp(30px, 2.4vw, 42px);
    font-weight: 700;
    letter-spacing: 0.01em;
    line-height: 1.2;
}

.kb-page-head p {
    max-width: 72ch;
    margin: 10px 0 0;
    color: var(--kb-muted);
    font-size: 13px;
    line-height: 1.65;
}

.kb-panel {
    min-width: 0;
    padding: 22px !important;
    border: 1px solid var(--kb-border) !important;
    border-radius: 16px !important;
    background: var(--kb-surface) !important;
    box-shadow: var(--kb-shadow) !important;
}

.kb-panel-title h3 {
    margin: 0 0 5px;
    color: var(--kb-text);
    font-size: 18px;
    font-weight: 700;
    letter-spacing: 0.025em;
}

.kb-panel-title p {
    margin: 0 0 14px;
    color: var(--kb-muted);
    font-size: 12px;
    line-height: 1.55;
}

.kb-library-grid {
    align-items: flex-start;
}

.kb-settings-grid {
    align-items: stretch;
    gap: 18px;
}

.kb-chat-layout {
    display: grid !important;
    grid-template-columns: minmax(0, 1.85fr) minmax(320px, 0.95fr);
    max-width: 1480px;
    margin: 0 auto;
    align-items: stretch;
    gap: 16px;
}

.kb-chat-layout > .column {
    width: 100% !important;
    min-width: 0 !important;
    max-width: none !important;
    flex: none !important;
}

.kb-library-panel {
    align-self: flex-start;
    min-height: 0;
}

#kb-delete-confirm {
    padding: 11px 13px !important;
    border: 1px solid var(--kb-border) !important;
    border-radius: 11px !important;
    background: var(--kb-surface-muted) !important;
}

#kb-delete-confirm:has(input[type="checkbox"]:checked) {
    border-color: var(--kb-primary) !important;
    background: var(--kb-accent-soft) !important;
    box-shadow: inset 4px 0 0 var(--kb-primary);
}

#kb-delete-confirm input[type="checkbox"] {
    width: 18px;
    height: 18px;
    accent-color: var(--kb-primary);
}

#kb-delete-confirm input[type="checkbox"]:checked {
    border-color: var(--kb-primary) !important;
    background: var(--kb-primary) !important;
}

#kb-delete-confirm:has(input[type="checkbox"]:checked) label {
    color: var(--kb-primary) !important;
    font-weight: 650;
}

#kb-delete-confirm:has(input[type="checkbox"]:checked) label::after {
    content: "已确认";
    margin-left: 8px;
    padding: 2px 7px;
    border-radius: 999px;
    color: #ffffff;
    background: var(--kb-primary);
    font-size: 11px;
    white-space: nowrap;
}

.kb-chat-panel {
    grid-column: 1;
    grid-row: 1;
    overflow: hidden;
}

.kb-context-panel {
    grid-column: 2;
    grid-row: 1;
    align-self: stretch;
    background: var(--kb-surface) !important;
}

.kb-context-panel .kb-panel-title {
    margin: -22px -22px 18px;
    padding: 17px 20px 15px;
    border-bottom: 1px solid var(--kb-border);
    background: rgba(255, 255, 255, 0.62);
    border-radius: 16px 16px 0 0;
}

.kb-context-panel .kb-panel-title h3 {
    margin: 0;
    color: var(--kb-text);
    font-size: 17px;
}

.kb-context-panel .kb-panel-title p {
    margin: 5px 0 0;
    color: var(--kb-muted);
    font-size: 11px;
}

.kb-context-note {
    margin: 0 0 12px;
    padding: 10px 12px;
    border: 1px solid var(--kb-border);
    border-radius: 12px;
    color: var(--kb-muted);
    background: rgba(219, 234, 254, 0.62);
    font-size: 12px;
    line-height: 1.6;
}

.kb-evidence {
    max-height: clamp(420px, 57vh, 720px);
    overflow: auto;
    padding-right: 4px;
}

.kb-evidence h3 {
    margin: 22px 0 8px;
    font-size: 16px;
}

.kb-evidence blockquote {
    margin: 8px 0 0;
    padding: 11px 14px;
    border: 1px solid var(--kb-border);
    border-left: 3px solid var(--kb-accent);
    border-radius: 12px;
    color: var(--kb-text);
    background: rgba(237, 245, 255, 0.84);
    font-size: 12px;
    line-height: 1.6;
}

.kb-action-bar {
    display: grid !important;
    grid-template-columns: minmax(0, 1fr) auto;
    align-items: center;
    gap: 18px;
    margin-bottom: 14px;
    padding: 17px 20px !important;
}

.kb-action-bar > .column {
    width: auto !important;
    min-width: 0 !important;
    flex: none !important;
}

.kb-action-copy h3 {
    margin: 0 0 3px;
    color: var(--kb-text);
    font-size: 15px;
}

.kb-action-copy p {
    margin: 0;
    color: var(--kb-muted);
    font-size: 12px;
    line-height: 1.5;
}

.kb-output {
    min-height: clamp(300px, 48vh, 660px);
    padding: 28px !important;
    border: 1px solid var(--kb-border) !important;
    border-radius: 16px !important;
    background: var(--kb-surface) !important;
    box-shadow: var(--kb-shadow) !important;
}

.kb-output h1,
.kb-output h2,
.kb-output h3 {
    color: var(--kb-text);
}

.kb-quiz-stack {
    gap: 14px;
}

.kb-quiz-question {
    padding: 14px !important;
    border: 1px solid var(--kb-border) !important;
    border-radius: 12px !important;
    background: var(--kb-surface-muted) !important;
}

.kb-quiz-result {
    margin-top: 14px;
}

#kb-upload-button,
#kb-settings-button,
#kb-mindmap-button,
#kb-quiz-button {
    min-height: 40px;
    font-weight: 650;
}

#kb-mindmap-button,
#kb-quiz-button {
    min-width: 150px;
}

#kb-exit-button {
    position: fixed;
    z-index: 65;
    bottom: 20px;
    left: 16px;
    width: 232px;
    max-width: 232px;
    min-height: 38px;
    margin: 0;
    border-color: var(--kb-border) !important;
    color: var(--kb-sidebar-text) !important;
    background: rgba(255, 255, 255, 0.72) !important;
}

.kb-panel .block,
.kb-output.block {
    box-shadow: none;
}

.kb-panel > .hide-container {
    flex: 0 0 auto !important;
}

.kb-context-panel > .kb-evidence {
    flex: 1 1 auto !important;
}

.kb-chat-panel .block,
.kb-chat-panel .chatbot,
.kb-chat-panel [data-testid="chatbot"] {
    border: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
}

.kb-chat-panel .message.bot,
.kb-chat-panel [data-testid="bot"] {
    color: var(--kb-text) !important;
    background: transparent !important;
    font-size: 15px;
    line-height: 1.85;
}

.kb-chat-panel .message.user,
.kb-chat-panel [data-testid="user"] {
    color: var(--kb-text) !important;
    border: 1px solid var(--kb-border) !important;
    border-radius: 14px !important;
    background: rgba(219, 234, 254, 0.76) !important;
}

.kb-chat-panel textarea,
.kb-chat-panel input {
    background: rgba(255, 255, 255, 0.92) !important;
}

button,
input,
textarea,
select {
    transition: border-color 140ms ease, background-color 140ms ease,
        color 140ms ease !important;
}

.gradio-container button.primary {
    border-color: var(--kb-primary) !important;
    color: #ffffff !important;
    background: linear-gradient(135deg, #2563eb 0%, #168bff 100%) !important;
    box-shadow: 0 10px 22px rgba(37, 99, 235, 0.22);
}

.gradio-container button {
    border-radius: 11px !important;
    font-weight: 650;
}

.gradio-container button.primary:hover {
    border-color: var(--kb-primary-hover) !important;
    background: linear-gradient(135deg, #1d4ed8 0%, #0876e8 100%) !important;
}

button:focus-visible,
input:focus-visible,
textarea:focus-visible,
select:focus-visible {
    outline: 3px solid var(--kb-accent-soft) !important;
    outline-offset: 2px;
}

.gradio-container textarea,
.gradio-container input,
.gradio-container select {
    border-color: var(--kb-border) !important;
    border-radius: 11px !important;
    background: rgba(255, 255, 255, 0.9) !important;
    color: var(--kb-text) !important;
}

.block:has(input[role="combobox"][aria-label$="范围"]) .secondary-wrap {
    position: relative;
    width: 100%;
}

.block:has(input[role="combobox"][aria-label$="范围"]) .secondary-wrap input[role="combobox"] {
    width: 100% !important;
    padding-right: 36px !important;
    cursor: pointer;
}

.block:has(input[role="combobox"][aria-label$="范围"]) .secondary-wrap .icon-wrap {
    position: absolute;
    top: 50%;
    right: 8px;
    transform: translateY(-50%);
    pointer-events: none;
}

.block:has(input[role="combobox"][aria-label$="范围"]) .wrap-inner:has(.token) .secondary-wrap {
    flex: 0 0 56px;
    min-width: 56px;
}

.block:has(input[role="combobox"][aria-label$="范围"]) .wrap-inner:has(.token) .secondary-wrap input[role="combobox"] {
    height: 100%;
    padding: 0 !important;
    opacity: 0;
}

.block:has(input[role="combobox"][aria-label$="范围"]) .wrap-inner:has(.token) .secondary-wrap .remove-all {
    position: absolute;
    top: 50%;
    right: 28px;
    z-index: 1;
    transform: translateY(-50%);
}

.gradio-container footer {
    display: none !important;
}

@media (max-width: 1180px) {
    .kb-chat-layout {
        grid-template-columns: minmax(0, 1fr);
        gap: 16px;
    }

    .kb-chat-panel,
    .kb-context-panel {
        grid-column: 1;
    }

    .kb-chat-panel {
        grid-row: 1;
        border-right: 1px solid var(--kb-border) !important;
    }

    .kb-context-panel {
        grid-row: 2;
    }

    .kb-evidence {
        max-height: 340px;
    }
}

@media (max-width: 1020px) {
    #kb-header {
        position: sticky;
    }

    .kb-header {
        min-height: 62px;
        margin-left: 0;
        padding: 9px 18px;
    }

    .kb-brand {
        position: static;
        width: auto;
        min-width: 0;
        min-height: 0;
        padding: 0;
        background: transparent;
    }

    .kb-title {
        color: var(--kb-text);
        font-size: 17px;
    }

    .kb-subtitle {
        color: var(--kb-muted);
    }

    .kb-mark {
        color: var(--kb-accent);
        border-color: var(--kb-accent);
    }

    .kb-crumb {
        display: none;
    }

    #kb-workspace {
        display: block !important;
        min-height: 0;
    }

    #kb-workspace > .tab-wrapper {
        position: sticky;
        top: 62px;
        display: flex !important;
        width: 100%;
        height: auto !important;
        padding: 8px 18px !important;
        border-right: 0 !important;
        border-bottom: 1px solid var(--kb-border) !important;
        background: var(--kb-sidebar) !important;
        overflow-x: auto;
    }

    #kb-workspace > .tab-wrapper > .tab-container[role="tablist"] {
        flex-direction: row;
        gap: 4px;
    }

    #kb-workspace > .tab-wrapper [role="tab"] {
        flex: 0 0 auto;
        width: auto;
        white-space: nowrap;
    }

    #kb-workspace > .tab-wrapper [role="tab"],
    #kb-workspace > .tab-wrapper .overflow-menu > button {
        color: var(--kb-sidebar-muted) !important;
        -webkit-text-fill-color: var(--kb-sidebar-muted) !important;
    }

    #kb-workspace > .tab-wrapper [role="tab"]:hover,
    #kb-workspace > .tab-wrapper .overflow-menu > button:hover {
        color: var(--kb-primary) !important;
        -webkit-text-fill-color: var(--kb-primary) !important;
        background: rgba(37, 99, 235, 0.08) !important;
    }

    #kb-workspace > .tab-wrapper [role="tab"].selected {
        border-bottom: 0 !important;
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        background: linear-gradient(135deg, #2563eb 0%, #168bff 100%) !important;
        box-shadow: none !important;
    }

    #kb-workspace > .tabitem {
        width: 100%;
        max-width: none;
        margin: 0;
        padding: 28px 24px 44px !important;
    }

    #kb-exit-button {
        position: static;
        width: auto;
        max-width: 168px;
        margin: 0 auto 24px;
    }

    .kb-library-grid,
    .kb-settings-grid {
        display: grid !important;
        grid-template-columns: minmax(0, 1fr) !important;
    }

    .kb-library-grid > .column,
    .kb-settings-grid > .column {
        width: 100% !important;
        min-width: 0 !important;
        max-width: none !important;
        flex: none !important;
    }

}

@media (max-width: 680px) {
    .kb-header {
        min-height: 58px;
        gap: 12px;
        padding: 9px 14px;
    }

    .kb-mark {
        width: 27px;
        height: 31px;
        flex-basis: 27px;
        font-size: 14px;
    }

    .kb-title {
        font-size: 15px;
        color: var(--kb-text);
    }

    .kb-subtitle {
        display: none;
    }

    #kb-workspace > .tab-wrapper {
        top: 58px;
        padding: 6px 10px !important;
    }

    #kb-workspace > .tab-wrapper [role="tab"],
    #kb-workspace > .tab-wrapper .overflow-menu > button {
        min-height: 38px;
        padding: 7px 9px !important;
        font-size: 12px !important;
    }

    .kb-page-head {
        margin-bottom: 18px;
    }

    .kb-page-head h1 {
        font-size: 28px;
    }

    .kb-panel,
    .kb-output {
        padding: 15px !important;
    }

    .kb-context-panel .kb-panel-title {
        margin: -15px -15px 14px;
        padding: 14px 15px 13px;
    }

    #kb-workspace > .tabitem {
        padding: 24px 13px 36px !important;
    }

    .kb-library-panel {
        min-height: 0;
    }

    .kb-action-bar {
        align-items: stretch;
        grid-template-columns: minmax(0, 1fr);
    }
}

@media (prefers-reduced-motion: reduce) {
    button,
    input,
    textarea,
    select {
        transition: none !important;
    }
}
"""


def app_theme(gr: Any) -> Any:
    return gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="blue",
        neutral_hue="slate",
        spacing_size="md",
        radius_size="lg",
        text_size="md",
    ).set(
        body_background_fill="#eef6ff",
        body_background_fill_dark="#eef6ff",
        body_text_color="#10204a",
        body_text_color_dark="#10204a",
        body_text_color_subdued="#62749a",
        body_text_color_subdued_dark="#62749a",
        background_fill_primary="#ffffff",
        background_fill_primary_dark="#ffffff",
        background_fill_secondary="#edf5ff",
        background_fill_secondary_dark="#edf5ff",
        border_color_primary="#c9dcff",
        border_color_primary_dark="#c9dcff",
        input_background_fill="#ffffff",
        input_background_fill_dark="#ffffff",
        input_border_color="#c9dcff",
        input_border_color_dark="#c9dcff",
        input_border_color_focus="#168bff",
        input_border_color_focus_dark="#168bff",
        input_placeholder_color="#7585a6",
        input_placeholder_color_dark="#7585a6",
        button_primary_background_fill="#2563eb",
        button_primary_background_fill_hover="#1d4ed8",
        button_primary_background_fill_dark="#2563eb",
        button_primary_background_fill_hover_dark="#1d4ed8",
        button_primary_text_color="#ffffff",
        button_primary_text_color_dark="#ffffff",
        button_transform_active="none",
        block_radius="14px",
        block_label_background_fill="transparent",
        block_label_background_fill_dark="transparent",
        block_label_border_width="0px",
        block_label_border_width_dark="0px",
        block_label_padding="4px 0",
        block_label_text_color="#2563eb",
        block_label_text_color_dark="#2563eb",
        input_radius="11px",
        button_large_radius="11px",
        button_medium_radius="11px",
        button_small_radius="10px",
        block_shadow="none",
        block_shadow_dark="none",
    )


@dataclass(frozen=True)
class RuntimeConfig:
    """Runtime settings, all overridable through environment variables."""

    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    db_path: str = "./chroma_db"
    llm_base_url: str = "https://api.deepseek.com"
    llm_model: str = "deepseek-flash"
    retrieval_k: int = 12
    context_k: int = 4
    llm_context_tokens: int = 8192
    llm_max_tokens: int = 2048
    retrieval_mode: str = "dense"
    document_routing: bool = False
    query_decomposition: bool = False
    parent_window: bool = False
    spatial_figure_evidence: bool = False
    formula_evidence: bool = False
    formula_evidence_auto: bool = True
    answer_validation: bool = False
    hybrid_candidate_k: int = 50
    hybrid_rrf_k: int = 60
    reranker_model: str | None = None
    reranker_revision: str | None = None
    reranker_batch_size: int = 8
    reranker_max_length: int = 512
    reranker_device: str = "cpu"
    reranker_rrf_k: int = 60
    vision_enabled: bool = False
    vision_model: str = "deepseek-v4-flash-vision-exp"

    @classmethod
    def from_env(cls) -> "RuntimeConfig":
        def positive_int(name: str, default: int) -> int:
            try:
                value = int(os.getenv(name, str(default)))
            except (TypeError, ValueError):
                return default
            return value if value > 0 else default

        def optional_text(name: str, default: str | None = None) -> str | None:
            value = os.getenv(name, default or "").strip()
            return value or None

        def enabled(name: str, default: str = "0") -> bool:
            return os.getenv(name, default).strip().casefold() in {"1", "true", "yes", "on"}

        retrieval_mode = os.getenv("SCI_RAG_RETRIEVAL_MODE", cls.retrieval_mode).strip().casefold()
        if retrieval_mode not in {"dense", "hybrid"}:
            retrieval_mode = cls.retrieval_mode
        document_routing = enabled("SCI_RAG_DOCUMENT_ROUTING")
        query_decomposition = enabled("SCI_RAG_QUERY_DECOMPOSITION")
        parent_window = enabled("SCI_RAG_PARENT_WINDOW")
        spatial_figure_evidence = enabled("SCI_RAG_SPATIAL_FIGURE_EVIDENCE")
        formula_evidence = enabled("SCI_RAG_FORMULA_EVIDENCE")
        formula_evidence_auto = enabled("SCI_RAG_FORMULA_EVIDENCE_AUTO", "1")
        answer_validation = enabled("SCI_RAG_ANSWER_VALIDATION")
        vision_enabled = enabled("SCI_RAG_VISION_ENABLED")
        return cls(
            embedding_model=os.getenv("SCI_RAG_EMBEDDING_MODEL", cls.embedding_model),
            db_path=os.getenv("SCI_RAG_DB_PATH", cls.db_path),
            llm_base_url=(
                os.getenv("LLM_BASE_URL")
                or cls.llm_base_url
            ),
            llm_model=(
                os.getenv("LLM_MODEL")
                or cls.llm_model
            ),
            retrieval_k=positive_int("SCI_RAG_RETRIEVAL_K", cls.retrieval_k),
            context_k=positive_int("SCI_RAG_CONTEXT_K", cls.context_k),
            llm_context_tokens=positive_int("LLM_CONTEXT_TOKENS", cls.llm_context_tokens),
            llm_max_tokens=positive_int("LLM_MAX_TOKENS", cls.llm_max_tokens),
            retrieval_mode=retrieval_mode,
            document_routing=document_routing,
            query_decomposition=query_decomposition,
            parent_window=parent_window,
            spatial_figure_evidence=spatial_figure_evidence,
            formula_evidence=formula_evidence,
            formula_evidence_auto=formula_evidence_auto,
            answer_validation=answer_validation,
            hybrid_candidate_k=positive_int(
                "SCI_RAG_HYBRID_CANDIDATE_K", cls.hybrid_candidate_k
            ),
            hybrid_rrf_k=positive_int("SCI_RAG_HYBRID_RRF_K", cls.hybrid_rrf_k),
            reranker_model=optional_text("SCI_RAG_RERANKER_MODEL"),
            reranker_revision=optional_text("SCI_RAG_RERANKER_REVISION"),
            reranker_batch_size=positive_int(
                "SCI_RAG_RERANKER_BATCH_SIZE", cls.reranker_batch_size
            ),
            reranker_max_length=positive_int(
                "SCI_RAG_RERANKER_MAX_LENGTH", cls.reranker_max_length
            ),
            reranker_device=os.getenv(
                "SCI_RAG_RERANKER_DEVICE", cls.reranker_device
            ).strip()
            or cls.reranker_device,
            reranker_rrf_k=positive_int(
                "SCI_RAG_RERANKER_RRF_K", cls.reranker_rrf_k
            ),
            vision_enabled=vision_enabled,
            vision_model=os.getenv("SCI_RAG_VISION_MODEL", cls.vision_model).strip()
            or cls.vision_model,
        )


@dataclass
class LexicalSnapshot:
    """Cached collection text used by optional lexical and source routing."""

    collection_count: int
    ids: list[str]
    texts: list[str]
    metadatas: list[dict[str, Any]]
    index: BM25Index
    lexical_indices: list[int]
    router: DocumentRouter | None = None


class Runtime:
    """Explicitly initialized model/API/database resources."""

    def __init__(
        self,
        config: RuntimeConfig,
        client: Any,
        embedding_model: Any,
        collection: Any,
        reranker: Any | None = None,
    ):
        if reranker is not None and config.retrieval_mode != "hybrid":
            raise ValueError("cross-encoder reranker 只能与 hybrid 检索一起启用")
        self.config = config
        self.client = client
        self.embedding_model = embedding_model
        self.collection = collection
        self.reranker = reranker
        self._lexical_snapshot: LexicalSnapshot | None = None

    def invalidate_lexical_index(self) -> None:
        self._lexical_snapshot = None


def formula_evidence_enabled(question: str, config: RuntimeConfig) -> bool:
    """Return whether the formula evidence path should run for one question."""

    return bool(
        config.formula_evidence
        or (config.formula_evidence_auto and is_formula_question(question))
    )


def create_runtime(config: RuntimeConfig | None = None) -> Runtime:
    """Initialize external resources exactly once per caller-owned runtime."""

    from dotenv import load_dotenv

    load_dotenv()
    config = config or RuntimeConfig.from_env()
    if config.reranker_model and config.retrieval_mode != "hybrid":
        raise ValueError("SCI_RAG_RERANKER_MODEL 需要 SCI_RAG_RETRIEVAL_MODE=hybrid")
    api_key = os.getenv("LLM_API_KEY")

    from openai import OpenAI
    from sentence_transformers import SentenceTransformer
    import chromadb

    client = OpenAI(api_key=api_key, base_url=config.llm_base_url) if api_key else None
    embedding_model = SentenceTransformer(config.embedding_model)
    reranker = None
    if config.reranker_model:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        reranker = CrossEncoderReranker(
            config.reranker_model,
            revision=config.reranker_revision,
            batch_size=config.reranker_batch_size,
            max_length=config.reranker_max_length,
            device=config.reranker_device,
            local_files_only=True,
        )
    chroma_client = chromadb.PersistentClient(path=config.db_path)
    collection = chroma_client.get_or_create_collection(
        name="knowledge_base",
        metadata={"hnsw:space": "cosine"},
    )
    return Runtime(config, client, embedding_model, collection, reranker=reranker)


def model_service_defaults(service: str) -> tuple[str, str]:
    """Return the editable URL and model for one UI preset."""

    return MODEL_SERVICE_PRESETS.get(str(service), ("", ""))


def configure_model_service(
    base_url: str,
    model: str,
    api_key: str = "",
    *,
    runtime: Runtime,
) -> str:
    """Configure an OpenAI-compatible generation service for this process."""

    base_url = str(base_url or "").strip()
    model = str(model or "").strip()
    api_key = str(api_key or "").strip()
    parsed_url = urlsplit(base_url)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        return "请输入有效的模型服务 Base URL。"
    if not model:
        return "请输入模型名称。"
    if not api_key and parsed_url.hostname not in {"localhost", "127.0.0.1", "::1"}:
        return "云端模型服务需要 API Key；Ollama 本地服务可以留空。"
    from openai import OpenAI

    client = OpenAI(api_key=api_key or "local", base_url=base_url)
    try:
        models = client.models.list()
    except Exception as exc:
        if getattr(exc, "status_code", None) == 401:
            return "❌ 连接测试失败，设置未应用：API Key 无效或没有访问权限，请确认复制完整。"
        if getattr(exc, "status_code", None) == 402:
            return "❌ 连接测试失败，设置未应用：模型账户余额不足，请先充值或检查额度。"
        return f"❌ 连接测试失败，设置未应用：{html.escape(str(exc))}"
    available_models = {
        str(getattr(item, "id", "")).strip() for item in (models.data or [])
    }
    if available_models and model not in available_models:
        available = "、".join(sorted(available_models))
        return f"❌ 服务中没有模型 {html.escape(model)}；可用模型：{html.escape(available)}"
    runtime.client = client
    runtime.config = replace(runtime.config, llm_base_url=base_url, llm_model=model)
    return f"✅ 连接测试成功，当前会话已使用模型 {html.escape(model)}。"


def _docx_to_markdown(file_path: str) -> str:
    """Read DOCX with the declared ``python-docx`` dependency."""

    from docx import Document as WordDocument

    document = WordDocument(file_path)
    parts = [paragraph.text for paragraph in document.paragraphs if paragraph.text.strip()]
    for table in document.tables:
        rows = [[cell.text.replace("\n", " ").strip() for cell in row.cells] for row in table.rows]
        if not rows:
            continue
        table_lines = ["|" + "|".join(rows[0]) + "|", "|" + "|".join("---" for _ in rows[0]) + "|"]
        table_lines.extend("|" + "|".join(row) + "|" for row in rows[1:])
        parts.append("\n".join(table_lines))
    return "\n\n".join(parts)


def _pdf_text_layer_for_formula_recovery(page: Any) -> str:
    """Keep equation text in visual order when PyMuPDF's plain text flattens it."""

    lines: list[tuple[tuple[float, float, float, float], str]] = []
    for block in page.get_text("dict", sort=True).get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            text = "".join(str(span.get("text", "")) for span in line.get("spans", []))
            if text.strip():
                lines.append((tuple(float(value) for value in line["bbox"]), text))
    lines.sort(key=lambda item: (item[0][1], item[0][0]))

    merged: list[list[Any]] = []
    for bbox, text in lines:
        center = (bbox[1] + bbox[3]) / 2
        if merged:
            previous_bbox, previous_text, previous_center = merged[-1]
            gap = bbox[0] - previous_bbox[2]
            if abs(center - previous_center) <= 4 and 0 <= gap <= 8:
                merged[-1] = [
                    (previous_bbox[0], min(previous_bbox[1], bbox[1]), bbox[2], max(previous_bbox[3], bbox[3])),
                    previous_text + text,
                    (previous_center + center) / 2,
                ]
                continue
        merged.append([bbox, text, center])

    return "\n".join(
        text.replace("\x10", "(").replace("\x11", ")")
        for _bbox, text, _center in merged
    )


def _pdf_markdown_in_column_order(page_chunk: dict[str, Any], page_width: float) -> str:
    """Keep each separated column together before Markdown heading splitting."""

    text = page_chunk["text"]
    boxes = page_chunk["page_boxes"]
    # Page-footer boxes are publication metadata, not sentence continuations.
    without_footers = text
    for box in sorted(
        (box for box in boxes if box["class"] == "page-footer"),
        key=lambda box: box["pos"][0],
        reverse=True,
    ):
        start, end = box["pos"]
        without_footers = without_footers[:start] + without_footers[end:]
    body = [box for box in boxes if box["class"] not in ("page-header", "page-footer")]
    middle = page_width / 2
    left = [box for box in body if box["bbox"][2] < middle]
    right = [box for box in body if box["bbox"][0] > middle]
    # ponytail: only strict two-column pages; spanning layouts keep native order.
    # Extend this only after a measured failure on a spanning layout.
    if len(left) + len(right) != len(body) or any(
        sum(box["class"] == "text" for box in column) < 2 for column in (left, right)
    ):
        return without_footers
    if max(min(box["bbox"][1] for box in column) for column in (left, right)) >= min(
        max(box["bbox"][3] for box in column) for column in (left, right)
    ):
        return without_footers
    ordered = (
        [box for box in boxes if box["class"] == "page-header"]
        + sorted(left, key=lambda box: (box["bbox"][1], box["bbox"][0]))
        + sorted(right, key=lambda box: (box["bbox"][1], box["bbox"][0]))
    )
    return (
        text[:boxes[0]["pos"][0]]
        + "".join(text[box["pos"][0]:box["pos"][1]] for box in ordered)
        + text[boxes[-1]["pos"][1]:]
    )


def load_and_split_document(
    file_path: str,
    include_spatial_figures: bool = False,
) -> list[Chunk]:
    """Load PDF/TXT/DOCX and return canonical, page-aware chunks.

    Spatial figure evidence is experimental and opt-in. It reads coordinates
    already present in a born-digital PDF text layer; it neither persists nor
    interprets image pixels.
    """

    source = os.path.basename(file_path)
    suffix = Path(file_path).suffix.lower()
    documents: list[Chunk] = []

    if suffix == ".pdf":
        import pymupdf
        import pymupdf4llm

        document = pymupdf.open(file_path)
        try:
            page_chunks = pymupdf4llm.to_markdown(
                document,
                filename=source,
                page_chunks=True,
                table_output="markdown",
                write_images=False,
                embed_images=False,
            )
            for page_number, page_chunk in enumerate(page_chunks, start=1):
                text = _pdf_markdown_in_column_order(
                    page_chunk, document[page_number - 1].rect.width
                )
                text = re.sub(r"(?<=\d) _\._ (?=\d)", ".", text)
                if text.strip():
                    documents.append(
                        Chunk(
                            page_content=text,
                            metadata={"source": source, "page": page_number},
                        )
                    )
                    for formula_text in missing_pdf_formula_blocks(
                        text,
                        _pdf_text_layer_for_formula_recovery(
                            document[page_number - 1]
                        ),
                    ):
                        documents.append(
                            Chunk(
                                page_content=formula_text,
                                metadata={
                                    "source": source,
                                    "page": page_number,
                                    "type": "formula",
                                },
                            )
                        )
            if include_spatial_figures:
                for page_number, page in enumerate(document, start=1):
                    documents.extend(
                        extract_spatial_figure_chunks(
                            page.get_text("blocks", sort=True),
                            source,
                            page_number,
                            float(page.rect.width),
                            float(page.rect.height),
                        )
                    )
        finally:
            document.close()
    elif suffix == ".txt":
        documents.append(
            Chunk(Path(file_path).read_text(encoding="utf-8"), {"source": source, "page": 1})
        )
    elif suffix == ".docx":
        documents.append(Chunk(_docx_to_markdown(file_path), {"source": source, "page": 1}))
    else:
        raise ValueError("不支持的文件类型（仅支持 .pdf / .txt / .docx）")

    return split_to_chunks(documents, source)


def _metadata_for_chroma(metadata: dict[str, Any]) -> dict[str, Any]:
    """Remove unsupported ``None`` values before writing Chroma metadata."""

    return {key: value for key, value in metadata.items() if value is not None}


def _source_name_for_upload(file_path: str, document_hash: str, runtime: Runtime) -> str:
    """Keep different files with the same basename independently selectable."""

    source = os.path.basename(file_path)
    existing = runtime.collection.get(
        where={"source": {"$eq": source}},
        include=["metadatas"],
    )
    if not _flat_result_values(existing, "ids"):
        return source
    existing_hashes = {
        str(metadata.get("document_sha256", ""))
        for metadata in _flat_result_values(existing, "metadatas")
        if isinstance(metadata, dict)
    }
    if document_hash in existing_hashes:
        return source
    path = Path(source)
    return f"{path.stem} ({document_hash[:12]}){path.suffix}"


def add_document_to_db(
    file_path: str,
    runtime: Runtime,
    progress: Callable[..., Any] | None = None,
) -> str:
    if progress is not None:
        progress(0.05, desc="正在读取文档")
    document_hash = file_sha256(file_path)
    source = _source_name_for_upload(file_path, document_hash, runtime)
    previous = runtime.collection.get(
        where={"source": {"$eq": source}}, include=["metadatas"]
    )
    previous_ids = {
        str(doc_id)
        for doc_id, metadata in zip(
            _flat_result_values(previous, "ids"),
            _flat_result_values(previous, "metadatas"),
        )
        if metadata.get("document_sha256") == document_hash
    }
    if progress is not None:
        progress(0.1, desc="正在解析文档")
    chunks = load_and_split_document(
        file_path,
        include_spatial_figures=runtime.config.spatial_figure_evidence,
    )
    if not chunks:
        raise ValueError("文档中没有可导入的文本")
    if runtime.config.vision_enabled and Path(file_path).suffix.lower() == ".pdf":
        source_dir = Path(runtime.config.db_path) / "source_pdfs"
        source_dir.mkdir(parents=True, exist_ok=True)
        source_pdf = source_dir / f"{document_hash}.pdf"
        if not source_pdf.exists():
            shutil.copyfile(file_path, source_pdf)
    records: list[tuple[str, str, dict[str, Any]]] = []
    normal_chunk_index = 0
    for index, chunk in enumerate(chunks):
        text = chunk.page_content
        metadata = dict(chunk.metadata)
        is_formula = metadata.get("type") == "formula"
        stable_index: int | str = f"formula:{index}" if is_formula else normal_chunk_index
        metadata.update(
            {
                "source": source,
                "document_sha256": document_hash,
            }
        )
        if not is_formula:
            # Formula evidence is intentionally outside the ordinary document
            # sequence, so it cannot change parent-window neighbors or IDs of
            # pre-existing prose/table chunks.
            metadata["chunk_index"] = normal_chunk_index
            normal_chunk_index += 1
        stable_id = hashlib.sha256(
            f"{document_hash}:{stable_index}:{metadata.get('type', 'text')}:{text}".encode("utf-8")
        ).hexdigest()
        metadata["chunk_id"] = stable_id
        records.append((stable_id, text, _metadata_for_chroma(metadata)))

    for start in range(0, len(records), DOCUMENT_BATCH_SIZE):
        batch = records[start : start + DOCUMENT_BATCH_SIZE]
        texts = [record[1] for record in batch]
        embeddings = runtime.embedding_model.encode(texts).tolist()
        runtime.collection.upsert(
            ids=[record[0] for record in batch],
            embeddings=embeddings,
            documents=texts,
            metadatas=[record[2] for record in batch],
        )
        if progress is not None:
            progress(
                0.1 + 0.85 * min(start + len(batch), len(records)) / max(len(records), 1),
                desc=f"正在写入知识库（{min(start + len(batch), len(records))}/{len(records)}）",
            )
    # Retain the old parse until every replacement batch has been written.
    # Empty/failed imports must not erase the existing document.
    stale_ids = previous_ids - {record[0] for record in records}
    if records and stale_ids:
        runtime.collection.delete(ids=sorted(stale_ids))
    runtime.invalidate_lexical_index()
    if progress is not None:
        progress(1, desc="文档已添加")
    return f"成功添加 {source}，共 {len(chunks)} 个文本块；知识库现有 {runtime.collection.count()} 个。"


def upload_file(
    file: str | os.PathLike[str] | None,
    runtime: Runtime,
    progress: Callable[..., Any] | None = None,
) -> str:
    if file is None:
        return "请选择一个文件"
    file_path = os.fspath(file)
    try:
        return add_document_to_db(file_path, runtime=runtime, progress=progress)
    except Exception as exc:
        return f"添加失败：{exc}"


def _vision_pdf_for_question(
    message: str,
    runtime: Runtime,
    source_filter: str | set[str] | None = None,
) -> tuple[Path, str] | None:
    """Resolve one persisted PDF for a routed, explicit figure question."""

    if (
        not runtime.config.vision_enabled
        or not runtime.config.document_routing
        or figure_reference_from_question(message) is None
        or is_table_question(message)
    ):
        return None
    snapshot = _get_lexical_snapshot(runtime)
    route = snapshot.router.route(message) if snapshot.router else None
    allowed_sources = _normalise_source_filter(source_filter)
    source = str(getattr(route, "document_id", "") or "").strip()
    if source and allowed_sources and source not in allowed_sources:
        source = ""
    if not source and len(allowed_sources) == 1:
        source = next(iter(allowed_sources))
    if not source or not source.casefold().endswith(".pdf"):
        return None
    hashes = {
        str(metadata.get("document_sha256", "")).strip()
        for metadata in snapshot.metadatas
        if str(metadata.get("source", "")).strip() == source
        and str(metadata.get("document_sha256", "")).strip()
    }
    if len(hashes) != 1:
        return None
    source_pdf = Path(runtime.config.db_path) / "source_pdfs" / f"{next(iter(hashes))}.pdf"
    return (source_pdf, source)


def _vision_answer(
    message: str,
    runtime: Runtime,
    source_filter: str | set[str] | None = None,
) -> tuple[str | None, str | None, dict[str, Any] | None]:
    """Try one full+detail figure request without changing text retrieval."""

    if runtime.client is None:
        return None, None, None
    selected = _vision_pdf_for_question(message, runtime, source_filter=source_filter)
    if selected is None:
        return None, None, None
    source_pdf, source = selected
    if not source_pdf.is_file():
        return None, "视觉源 PDF 不存在，已回退文本检索。", None
    reference = figure_reference_from_question(message)
    assert reference is not None
    try:
        image = render_figure(source_pdf, reference, include_detail=True)
        detail = image.get("detail")
        answer = complete_vision(
            runtime.client,
            runtime.config.vision_model,
            vision_messages(
                message,
                str(image["data_url"]),
                str(detail["data_url"]) if isinstance(detail, dict) else None,
            ),
        )
    except Exception as exc:
        return None, f"视觉问答不可用，已回退文本检索（{type(exc).__name__}）。", None
    label = (
        f"Extended Data Figure {reference[1]}"
        if reference[0] == "extended_data_figure"
        else f"Figure {reference[1]}"
    )
    page = image.get("page")
    page_text = f"，第 {page} 页" if page else ""
    answer += f"\n\n📌 **参考来源：**\n- {source}{page_text}（{label}）"
    metadata = {"source": source, "page": page, "type": "vision", "figure_label": label}
    return answer, None, metadata


def _merge_results(
    dense: dict[str, Any],
    tables: dict[str, Any] | None,
    additional_results: list[dict[str, Any]] | None = None,
) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    texts: list[str] = []
    ids: list[str] = []
    metas: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(result: dict[str, Any]) -> None:
        result_ids = (result.get("ids") or [[]])[0]
        result_docs = (result.get("documents") or [[]])[0]
        result_metas = (result.get("metadatas") or [[]])[0]
        for index, doc_id in enumerate(result_ids):
            if doc_id in seen:
                continue
            seen.add(doc_id)
            ids.append(doc_id)
            texts.append(result_docs[index] if index < len(result_docs) else "")
            metas.append(result_metas[index] if index < len(result_metas) else {})

    for result in [*(additional_results or []), dense]:
        add(result)
    if tables:
        # Collection.get() returns flat lists, unlike query()'s nested lists.
        flat_ids = tables.get("ids") or []
        flat_docs = tables.get("documents") or []
        flat_metas = tables.get("metadatas") or []
        add({
            "ids": [flat_ids],
            "documents": [flat_docs],
            "metadatas": [flat_metas],
        })
    return texts, ids, metas


def _flat_result_values(result: dict[str, Any], key: str) -> list[Any]:
    """Read the flat get payload or the single-query nested payload from Chroma."""

    values = result.get(key) or []
    if values and isinstance(values[0], list):
        values = values[0]
    return list(values)


def document_inventory(runtime: Runtime) -> list[tuple[str, int]]:
    """Return uploaded source names and their chunk counts."""

    result = runtime.collection.get(include=["metadatas"])
    counts: dict[str, int] = {}
    for metadata in _flat_result_values(result, "metadatas"):
        source = str(metadata.get("source", "")).strip() if isinstance(metadata, dict) else ""
        if source:
            counts[source] = counts.get(source, 0) + 1
    return sorted(counts.items(), key=lambda item: item[0].casefold())


def delete_document(
    source: str | list[str] | None,
    confirmed: bool = False,
    *,
    runtime: Runtime,
) -> str:
    """Delete selected sources and persisted vision PDFs after confirmation."""

    sources = sorted(_normalise_source_filter(source), key=str.casefold)
    if not sources:
        return "请选择要删除的文档。"
    if not confirmed:
        return "请先确认删除。"

    ids: list[str] = []
    digests: set[str] = set()
    deleted_sources: list[str] = []
    for selected_source in sources:
        result = runtime.collection.get(
            where={"source": {"$eq": selected_source}},
            include=["metadatas"],
        )
        source_ids = [str(value) for value in _flat_result_values(result, "ids")]
        if not source_ids:
            continue
        ids.extend(source_ids)
        deleted_sources.append(selected_source)
        digests.update(
            str(metadata.get("document_sha256", "")).strip().casefold()
            for metadata in _flat_result_values(result, "metadatas")
            if isinstance(metadata, dict)
            and re.fullmatch(
                r"[0-9a-fA-F]{64}",
                str(metadata.get("document_sha256", "")).strip(),
            )
        )
    if not ids:
        return "未找到所选文档。"
    runtime.collection.delete(ids=ids)
    runtime.invalidate_lexical_index()
    for digest in digests:
        remaining = runtime.collection.get(
            where={"document_sha256": {"$eq": digest}},
            include=["metadatas"],
        )
        if not _flat_result_values(remaining, "ids"):
            (Path(runtime.config.db_path) / "source_pdfs" / f"{digest}.pdf").unlink(
                missing_ok=True
            )
    names = "、".join(deleted_sources)
    return f"✅ 已删除 {len(deleted_sources)} 份文档（{len(ids)} 个文本块）：{names}。"


def _normalise_source_filter(value: Any) -> set[str]:
    values = value if isinstance(value, (list, tuple, set, frozenset)) else [value]
    return {
        str(item).strip()
        for item in values
        if item is not None and str(item).strip()
    }


def _all_selected_sources_requested(question: str, sources: set[str]) -> bool:
    """Recognize an explicit comparison of exactly two selected documents."""

    if len(sources) != 2:
        return False
    text = str(question or "")
    chinese_scope = bool(
        re.search(r"(?:这|所选)?两\s*(?:篇|份|个)?\s*(?:论文|文档|资料)", text)
        and re.search(r"各自|分别|比较|对比|异同|区别|取舍", text)
    )
    english_scope = bool(
        re.search(r"\b(?:both|two)\s+(?:selected\s+)?(?:papers?|documents?)\b", text, re.I)
        and re.search(r"\b(?:each|respectively|compare|comparison|difference|trade-?off)\b", text, re.I)
    )
    return chinese_scope or english_scope


def _source_where_clause(sources: set[str]) -> dict[str, Any] | None:
    if not sources:
        return None
    if len(sources) == 1:
        return {"source": {"$eq": next(iter(sources))}}
    return {"source": {"$in": sorted(sources)}}


def _lexical_search_text(text: str, metadata: dict[str, Any]) -> str:
    return "\n".join(
        [
            text,
            str(metadata.get("table_caption", "")),
            str(metadata.get("headers", "")),
            str(metadata.get("source", "")),
        ]
    )


_TABLE_UNIT_HINT_RE = re.compile(
    r"(?:[×x]\s*10|%|Å|kcal(?:\s*/\s*mol)?|s\s*/\s*iter|\bunits?\b)",
    re.IGNORECASE,
)


def _table_caption_unit_note(metadata: dict[str, Any]) -> str:
    """Return a concise table-note suffix when the caption declares units.

    Deterministic table-cell lookup otherwise returns only the row/column value.
    Captions are the authoritative place where some scientific tables declare a
    shared scale (for example ``×10^-2``), so preserve that note without
    inventing or converting a value.  Captions without an explicit unit hint
    are intentionally omitted to keep existing concise answers unchanged.
    """

    caption = str(metadata.get("table_caption", "") or "")
    if not caption:
        return ""
    plain = html.unescape(re.sub(r"<[^>]*>", " ", caption))
    plain = re.sub(r"\s+", " ", plain).strip()
    if not plain or not _TABLE_UNIT_HINT_RE.search(plain):
        return ""
    body = re.sub(
        r"^(?:extended\s+data\s+)?table\s*[A-Za-z]?\d+[A-Za-z]?\s*[:.]?\s*",
        "",
        plain,
        flags=re.IGNORECASE,
    ).strip()
    return f"（表注：{body or plain}）"


def _table_caption_metric_note(metadata: dict[str, Any], question: str) -> str:
    """Preserve a caption-declared metric when the question names it."""

    caption = str(metadata.get("table_caption", "") or "")
    plain = html.unescape(re.sub(r"<[^>]*>", " ", caption))
    match = re.search(
        r"(?:measured|evaluated|based)\s+(?:by|using|on)\s+([^.;]+)",
        plain,
        re.IGNORECASE,
    )
    metric = re.sub(r"\s+", " ", match.group(1)).strip() if match else ""
    if not metric or normalize_for_match(metric) not in normalize_for_match(question):
        return ""
    return f"（表格指标：{metric}）"


def _attach_table_caption(text: str, metadata: dict[str, Any]) -> str:
    """Keep a table's caption beside its body in the generation context."""

    caption = str(metadata.get("table_caption", "") or "").strip()
    if metadata.get("type") != "table" or not caption or caption in text:
        return text
    return f"{caption}\n\n{text}"


def _fuse_dense_results(
    results: list[dict[str, Any]],
    limit: int,
    rrf_k: int,
) -> dict[str, Any]:
    """Fuse dense rankings produced for deterministic query variants."""

    if not results:
        return {"ids": [[]], "documents": [[]], "metadatas": [[]]}
    id_lists = [
        [str(value) for value in _flat_result_values(result, "ids")]
        for result in results
    ]
    fused = reciprocal_rank_fusion(id_lists, rrf_k=rrf_k, limit=limit)
    payloads: dict[str, tuple[str, dict[str, Any]]] = {}
    for result in results:
        ids = [str(value) for value in _flat_result_values(result, "ids")]
        texts = _flat_result_values(result, "documents")
        metas = _flat_result_values(result, "metadatas")
        for index, doc_id in enumerate(ids):
            if doc_id in payloads:
                continue
            metadata = metas[index] if index < len(metas) and isinstance(metas[index], dict) else {}
            text = str(texts[index]) if index < len(texts) else ""
            payloads[doc_id] = (text, dict(metadata))
    ids: list[str] = []
    texts: list[str] = []
    metadatas: list[dict[str, Any]] = []
    for item in fused:
        doc_id = str(item.key)
        payload = payloads.get(doc_id)
        if payload is None:
            continue
        ids.append(doc_id)
        texts.append(payload[0])
        metadatas.append(payload[1])
    return {"ids": [ids], "documents": [texts], "metadatas": [metadatas]}


def _get_lexical_snapshot(runtime: Runtime) -> LexicalSnapshot:
    """Build or reuse lexical indexes for the current Chroma collection."""

    collection_count = runtime.collection.count()
    cached = runtime._lexical_snapshot
    if cached is not None and cached.collection_count == collection_count:
        return cached

    result = runtime.collection.get(include=["documents", "metadatas"])
    raw_ids = _flat_result_values(result, "ids")
    raw_texts = _flat_result_values(result, "documents")
    raw_metas = _flat_result_values(result, "metadatas")
    ids = [str(value) for value in raw_ids]
    texts = [str(raw_texts[index]) if index < len(raw_texts) else "" for index in range(len(ids))]
    metadatas = [
        dict(raw_metas[index]) if index < len(raw_metas) and isinstance(raw_metas[index], dict) else {}
        for index in range(len(ids))
    ]
    lexical_indices = [
        index
        for index, metadata in enumerate(metadatas)
        if (
            metadata.get("type") != "formula"
            and (
                not runtime.config.spatial_figure_evidence
                or metadata.get("type") != "figure"
            )
        )
    ]
    search_documents = [
        _lexical_search_text(texts[index], metadatas[index])
        for index in lexical_indices
    ]
    source_profiles: dict[str, list[str]] = {}
    for text, metadata in zip(texts, metadatas):
        if (
            metadata.get("type") == "formula"
            or (
                runtime.config.spatial_figure_evidence
                and metadata.get("type") == "figure"
            )
        ):
            continue
        source = str(metadata.get("source", "")).strip()
        if source:
            source_profiles.setdefault(source, []).append(
                _lexical_search_text(text, metadata)
            )
    router = None
    if len(source_profiles) >= 1:
        router = DocumentRouter(
            source_profiles.keys(),
            ("\n".join(parts) for parts in source_profiles.values()),
        )
    snapshot = LexicalSnapshot(
        collection_count=collection_count,
        ids=ids,
        texts=texts,
        metadatas=metadatas,
        index=BM25Index(search_documents),
        lexical_indices=lexical_indices,
        router=router,
    )
    runtime._lexical_snapshot = snapshot
    return snapshot


def _hybrid_fused_result(
    question: str,
    dense: dict[str, Any],
    runtime: Runtime,
    candidate_k: int,
    source_filter: str | set[str] | None = None,
    lexical_queries: list[str] | None = None,
) -> dict[str, Any]:
    """Fuse Chroma dense results with a cached lexical ranking using RRF."""

    dense_ids = [str(value) for value in _flat_result_values(dense, "ids")]
    dense_texts = _flat_result_values(dense, "documents")
    dense_metas = _flat_result_values(dense, "metadatas")
    dense_by_id = {
        doc_id: (
            str(dense_texts[index]) if index < len(dense_texts) else "",
            dict(dense_metas[index])
            if index < len(dense_metas) and isinstance(dense_metas[index], dict)
            else {},
        )
        for index, doc_id in enumerate(dense_ids)
    }

    snapshot = _get_lexical_snapshot(runtime)
    allowed_sources = _normalise_source_filter(source_filter)
    lexical_indices = [
        index
        for index, metadata in enumerate(snapshot.metadatas)
        if (
            metadata.get("type") != "formula"
            and (
                not allowed_sources
                or str(metadata.get("source", "")).strip() in allowed_sources
            )
            and (
                not runtime.config.spatial_figure_evidence
                or metadata.get("type") != "figure"
            )
        )
    ]
    lexical_positions = {
        index: position for position, index in enumerate(snapshot.lexical_indices)
    }
    lexical_positions_for_query = [
        lexical_positions[index]
        for index in lexical_indices
        if index in lexical_positions
    ]
    lexical_rankings: list[list[str]] = []
    for lexical_query in lexical_queries or [question]:
        if not snapshot.index.has_lexical_signal(lexical_query):
            continue
        lexical_rankings.append(
            [
                snapshot.ids[snapshot.lexical_indices[int(item.key)]]
                for item in snapshot.index.retrieve(
                    lexical_query, candidate_k, indices=lexical_positions_for_query
                )
            ]
        )
    lexical_ids = (
        [
            str(item.key)
            for item in reciprocal_rank_fusion(
                lexical_rankings,
                rrf_k=runtime.config.hybrid_rrf_k,
                limit=candidate_k,
            )
        ]
        if lexical_rankings
        else []
    )
    snapshot_by_id = {
        doc_id: (snapshot.texts[index], snapshot.metadatas[index])
        for index, doc_id in enumerate(snapshot.ids)
    }
    fused = reciprocal_rank_fusion(
        [dense_ids, lexical_ids],
        rrf_k=runtime.config.hybrid_rrf_k,
        limit=candidate_k,
    )

    ids: list[str] = []
    texts: list[str] = []
    metadatas: list[dict[str, Any]] = []
    for item in fused:
        doc_id = str(item.key)
        payload = dense_by_id.get(doc_id) or snapshot_by_id.get(doc_id)
        if payload is None:
            continue
        text, metadata = payload
        ids.append(doc_id)
        texts.append(text)
        metadatas.append(metadata)
    return {"ids": [ids], "documents": [texts], "metadatas": [metadatas]}


def _cross_encoder_reranked_result(
    question: str,
    result: dict[str, Any],
    runtime: Runtime,
) -> dict[str, Any]:
    """Rerank Hybrid candidates, then conservatively fuse the original order."""

    if runtime.reranker is None:
        return result
    ids = [str(value) for value in _flat_result_values(result, "ids")]
    raw_texts = _flat_result_values(result, "documents")
    raw_metas = _flat_result_values(result, "metadatas")
    texts = [str(raw_texts[index]) if index < len(raw_texts) else "" for index in range(len(ids))]
    metadatas = [
        dict(raw_metas[index])
        if index < len(raw_metas) and isinstance(raw_metas[index], dict)
        else {}
        for index in range(len(ids))
    ]
    candidates = [RankedItem(index, 0.0) for index in range(len(ids))]
    passages = [
        reranker_document_text(text, metadata)
        for text, metadata in zip(texts, metadatas)
    ]
    reranked = runtime.reranker.rerank(question, candidates, passages)
    fused = reciprocal_rank_fusion(
        [reranked, candidates],
        rrf_k=runtime.config.reranker_rrf_k,
        limit=len(candidates),
    )
    order = [int(item.key) for item in fused]
    return {
        "ids": [[ids[index] for index in order]],
        "documents": [[texts[index] for index in order]],
        "metadatas": [[metadatas[index] for index in order]],
    }


_COMPOSITE_FACT_CUE_RE = re.compile(
    r"多少|哪些|如何|管道|步骤|阶段|效率|代价|开销|以及|并且|同时|与|和|"
    r"(?:[一二三四五六七八九十0-9]+种|多个|两种|若干).{0,20}(?:什么|哪些|分别)|"
    r"\b(?:what|which|how|and|pipeline|dataset)\b",
    re.IGNORECASE,
)
_ENGLISH_LISTED_FACT_QUESTION_RE = re.compile(
    r"\b(?:list\s+(?:each|the)|(?:what|which)\s+(?:one|two|three|four|five|six|seven|eight|nine|\d+)\s+\w+)\b",
    re.IGNORECASE,
)
_LISTED_FACT_QUESTION_RE = re.compile(
    r"(?:[一二三四五六七八九十0-9]+种|多个|两种|若干).{0,20}(?:什么|哪些|分别)|"
    r"列出.{0,30}(?:每个|各)|"
    + _ENGLISH_LISTED_FACT_QUESTION_RE.pattern,
    re.IGNORECASE,
)
_REFERENCE_HEADER_RE = re.compile(
    r"(?:references?|bibliography|参考文献)", re.IGNORECASE
)
_PICTURE_TEXT_MARKER_RE = re.compile(
    r"<!--\s*(?:start|end) of picture text\s*-->|<img\b",
    re.IGNORECASE,
)
_FOOTNOTE_MARKER_RE = re.compile(r"<sup>\s*(\d{1,2})\s*</sup>")
_FOOTNOTE_LINE_RE = re.compile(r"^\s*>\s*(\d{1,2})(?!\d)\s*\S")
_SPATIAL_COORDINATE_RE = re.compile(
    r"\[x\s*=\s*(?P<x0>[-+]?\d+(?:\.\d+)?)\s*-\s*(?P<x1>[-+]?\d+(?:\.\d+)?)%?;\s*"
    r"y\s*=\s*(?P<y0>[-+]?\d+(?:\.\d+)?)\s*-\s*(?P<y1>[-+]?\d+(?:\.\d+)?)%?\]",
    re.IGNORECASE,
)
_SECTION_QUERY_ALIASES = {
    "数据集": ("dataset", "data"),
    "规模": ("dataset", "size", "entries", "samples", "statistics"),
    "分布": ("distribution", "benchmark", "questions", "text", "table", "image", "video"),
    "条目": ("entries",),
    "平均": ("average", "mean", "statistics", "analysis"),
    "变化": ("change", "difference", "delta"),
    "绝对": ("absolute", "difference"),
    "答案表": ("answer", "table"),
    "行": ("rows", "row"),
    "列": ("columns", "column"),
    "阈值": ("threshold", "overlap", "unigram"),
    "重叠": ("overlap", "unigram"),
    "一致率": ("agreement", "agreement rate"),
    "人工标注": ("annotation", "annotators", "reviewed", "instances", "agreement"),
    "标注": ("annotation", "annotator", "label"),
    "问题标注": ("question annotation", "annotation", "annotator"),
    "代理模型": ("agent annotator", "agent model", "model"),
    "表格集合": ("table set", "multi-table set", "tables"),
    "构造": ("construction", "collection", "construct"),
    "实体筛选": ("entity", "BM25", "alpha", "top"),
    "相似句": ("similar", "sentence", "top", "m"),
    "结构邻接": ("structural", "adjacency", "three", "sentences"),
    "设置": ("implementation", "details", "setting", "alpha", "top"),
    "构成": ("construction", "composition", "collection", "dataset"),
    "来源": ("source", "dataset"),
    "线索": ("cue", "cues", "metadata", "titles", "headers"),
    "人类引导": ("human-guided", "human-in-the-loop", "demonstration"),
    "关键方法": ("methods", "method", "approach"),
    "数量": ("number", "count", "statistics", "distribution", "instances"),
    "占比": ("percentage", "proportion", "distribution", "statistics"),
    "分类": ("category", "categories", "types", "distribution"),
    "消融": ("ablation", "ablated", "without"),
    "分为": ("taxonomy", "categories", "types"),
    "幻觉": ("hallucination", "hallucinations"),
    "整合": ("integration", "integrate", "integrates"),
    "论文": ("paper", "papers", "relevant papers", "abstracts"),
    "摘要": ("abstract", "abstracts", "PubMed"),
    "领域": ("domain", "domains", "field", "fields", "discipline", "disciplines"),
    "上限": ("up to", "capped", "maximum", "max"),
    "评分尺度": ("rating scale", "Likert", "score"),
    "输出格式": ("response format", "format", "JSON", "structured", "rationale"),
    "质检": ("quality", "quality control", "experts", "annotators", "annotation", "agreement", "kappa", "inferences", "code interpreter"),
    "质量控制": ("quality", "quality control", "experts", "annotators", "annotation", "agreement", "kappa"),
    "多少": ("number", "count"),
    "修订": ("revised", "refined", "edit"),
    "筛选": ("filter", "filtering", "filtered"),
    "盲测": ("blind test",),
    "错误答案": ("incorrect", "ground-truth"),
    "filtering": ("filtered",),
    "改写": ("rephrasing",),
    "技能": ("skills",),
    "问答对": ("question-answer pairs", "pairs"),
    "数据子集": ("data subsets", "subsets"),
    "下游任务": ("downstream tasks", "QA", "T2T"),
    "实例": ("instances",),
    "生成": ("generate", "generation"),
    "候选实例": ("candidate", "instances"),
    "显式推理": ("explicit-reasoning", "reasoning"),
    "推理": ("reasoning",),
    "模型": ("model", "models"),
    "架构": ("architecture", "encoder-decoder", "decoder-only"),
    "主干": ("trunk", "backbone", "architecture"),
    "结构模块": ("structure module", "diffusion module", "module"),
    "替换": ("replace", "replacing", "replacement", "substitute"),
    "问题生成": ("question", "generation", "generate"),
    "切分": ("split", "splitting", "caption", "subcaption"),
    "文本块": ("chunk", "chunked", "tokens"),
    "检索": ("retrieve", "retrieval", "BM25", "ranker", "rank", "ranks", "relevance", "top-k"),
    "嵌入": ("embedding", "embeddings", "vector"),
    "索引": ("index", "indexing", "OpenSearch"),
    "子章节": ("subsections", "section"),
    "排序": ("rank", "ranking", "ranker", "top"),
    "重排": ("rerank", "reranking", "reranker"),
    "候选句": ("candidate sentences",),
    "种子": ("seed", "seeds"),
    "扩展": ("expansion", "expand"),
    "三元组": ("triple", "triples"),
    "迭代": ("iteration", "iterations", "iterative"),
    "结束": ("stop", "terminate"),
    "证据评估": ("evidence", "evaluation", "supported", "refuted"),
    # Prefer a self-balanced/multi-granular RL section over a generic
    # training-settings heading when both share the same broad token.
    "强化学习": ("reinforcement", "rl", "training", "self-balanced", "multi-granular"),
    "训练": ("training", "train"),
    "随机种子": ("random", "seed", "seeds", "reproducibility", "variability"),
    "激活函数": ("activation", "GELU"),
    "正确答案": ("correct", "answer", "example"),
    "示例": ("example", "correct", "answer"),
    "评估指标": ("evaluation", "metrics", "accuracy", "recall", "hit"),
    "评价指标": ("evaluation", "metrics", "accuracy", "recall", "hit"),
    "评测": ("evaluation", "metrics"),
    "参考答案": ("reference answer", "reference answers"),
    "相似": ("similarity",),
    "忠实": ("faithfulness", "supported", "unsupported", "contradictory"),
    "加速策略": (
        "acceleration", "accelerate", "speedup", "latency", "draft",
        "speculative", "retrieval", "interaction",
    ),
    "效率": ("efficiency", "tokens", "regeneration", "time", "cost", "overhead"),
    "代价": ("cost", "overhead", "time", "tokens", "regeneration"),
    "开销": ("cost", "overhead", "time", "tokens", "regeneration"),
    "取舍": ("trade-off", "tradeoff", "overhead", "accuracy", "efficiency"),
    "部署": ("deployment", "deploy", "practitioners", "implications"),
    "高风险": ("high-stakes",),
    "建议": ("recommend", "recommendation", "practitioners", "implications"),
    "全文": ("full-text", "full text"),
    "多文档": ("multiple documents",),
    "图像": ("images", "figures", "visual", "multimodal"),
    "像素": ("images", "pixels"),
    "实验边界": ("limitations", "experiments", "include", "evaluate"),
    "完整文本": ("full-text", "full text"),
    "第一人称": ("first-person", "third-person", "rewrite", "decontextualize"),
    "偏差": ("bias",),
    "实验配置": ("experimental", "setup", "configuration", "configurations"),
    "量化": ("analysis", "experiment", "Recall@2", "sampled"),
    "配置": ("configuration", "configurations", "setup"),
    "基线": ("baseline", "configured"),
    "准确性奖励": ("accuracy rewards",),
    "格式奖励": ("format reward", "template"),
    "奖励": ("reward",),
    "管道": ("pipeline",),
    "流程": ("pipeline", "process", "steps"),
    "工具": ("tool", "toolkit", "parse", "parsing"),
    "过滤": ("filter", "blind test", "agreement"),
    "人工复核": ("manual", "verified", "subset", "samples", "instances"),
    "样本": ("samples", "instances", "subset"),
    "选项": ("answer choices", "options", "choices"),
}
_SOURCE_LOCAL_EVIDENCE_CUES = (
    "设置", "实体筛选", "相似句", "结构邻接", "多文档", "图像", "实验边界", "评测", "忠实", "分为", "整合",
    "阈值", "重叠", "一致率", "人工标注", "候选实例", "架构", "切分", "检索", "量化", "规模", "分布", "嵌入", "索引",
    "排序", "重排", "候选句", "种子", "扩展", "显式推理", "强化学习", "随机种子", "激活函数", "正确答案", "示例",
    "主干", "结构模块", "替换",
    "评估指标", "评价指标", "加速策略", "全文", "完整文本", "第一人称", "偏差", "实验配置", "配置", "基线", "奖励",
    "管道", "流程", "步骤", "阶段", "效率", "代价", "开销", "变化", "绝对", "工具", "过滤", "人工复核", "样本",
    "标注", "问题标注", "代理模型", "表格集合", "人类引导", "数量", "条目", "摘要", "领域", "上限", "评分尺度", "输出格式", "构成", "占比", "分类",
    "修订", "数据子集", "选项", "质检", "质量控制", "改写", "技能", "答案表", "盲测",
    "threshold", "overlap", "annotation", "architecture",
    "chunk", "retrieval", "ranking", "seed", "activation", "correct answer", "evaluation", "metrics", "acceleration strategies",
    "first-person", "pipeline", "steps", "configuration", "reward", "replace", "replacing", "rephrasing", "skills", "filtering",
)
_EXPLICIT_NUMBER_RE = re.compile(
    r"(?<![\w])(?:\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)(?![\w])"
)
MAX_SECTION_EXPANSION_CHUNKS = 6
MAX_PARENT_WINDOW_ANCHORS = 2


def _section_continuation_indices(
    anchor_index: int,
    all_metas: list[Any],
    all_texts: list[Any],
    source: str,
    limit: int,
    allow_unnumbered_heading: bool = False,
) -> list[int]:
    """Return text chunks contiguous with a section anchor.

    PDF-to-Markdown extraction may repeat a section header on adjacent chunks
    or omit it from continuations. Walk source-local ``chunk_index`` values
    until the next explicit heading, skip table/figure blocks, and never cross
    an index gap.
    """

    if limit <= 0:
        return []
    try:
        anchor_chunk_index = int(all_metas[anchor_index].get("chunk_index"))
    except (IndexError, AttributeError, TypeError, ValueError):
        return []
    source_by_chunk: dict[int, int] = {}
    for index, raw_meta in enumerate(all_metas):
        metadata = raw_meta if isinstance(raw_meta, dict) else {}
        if str(metadata.get("source", "")) != source:
            continue
        try:
            chunk_index = int(metadata.get("chunk_index"))
        except (TypeError, ValueError):
            continue
        source_by_chunk[chunk_index] = index

    continuation: list[int] = []
    # Only walk forward. Walking backward from the first chunk could import
    # prose belonging to the preceding section.
    current = anchor_chunk_index
    while len(continuation) < limit:
        current += 1
        index = source_by_chunk.get(current)
        if index is None:
            break
        metadata = all_metas[index] if isinstance(all_metas[index], dict) else {}
        if metadata.get("type", "text") != "text":
            continue
        headers = str(metadata.get("headers") or "").strip()
        anchor_headers = str(
            all_metas[anchor_index].get("headers")
            if isinstance(all_metas[anchor_index], dict)
            else ""
        ).strip()
        if headers and headers != anchor_headers:
            parent = anchor_headers.split(" > ")[0]
            child = headers.split(" > ")[-1]
            if not (
                allow_unnumbered_heading
                and headers.startswith(parent + " > ")
                and not re.match(r"H\d+:\s*\**\d", child)
            ):
                break
        if _is_picture_text_chunk(all_texts[index] if index < len(all_texts) else ""):
            continue
        if index not in continuation:
            continuation.append(index)
    return continuation[:limit]


def _is_picture_text_chunk(text: Any) -> bool:
    """Identify pymupdf4llm's OCR/image-text blocks for window isolation."""

    return bool(_PICTURE_TEXT_MARKER_RE.search(str(text or "")))


def _annotate_spatial_context(text: Any) -> str:
    """Add a deterministic page-level quadrant to coordinate text evidence."""

    lines: list[str] = []
    for line in str(text or "").splitlines():
        match = _SPATIAL_COORDINATE_RE.search(line)
        if not match or "page-level region" in line.casefold():
            lines.append(line)
            continue
        x_center = (float(match.group("x0")) + float(match.group("x1"))) / 2
        y_center = (float(match.group("y0")) + float(match.group("y1"))) / 2
        horizontal = "right" if x_center >= 50 else "left"
        vertical = "bottom" if y_center >= 50 else "top"
        lines.append(f"{line} [page-level region: {vertical}-{horizontal}]")
    return "\n".join(lines)


def _is_composite_fact_question(question: str) -> bool:
    """Detect questions likely to require evidence from multiple chunks."""

    text = str(question or "")
    matches = _COMPOSITE_FACT_CUE_RE.findall(text)
    return (
        len(matches) >= 2
        or bool(_LISTED_FACT_QUESTION_RE.search(text))
        or bool(re.search(r"(?:与|和|以及).{0,60}(?:分别|各自)", text))
        or bool(re.search(r"阶段|管道|步骤", text) and re.search(r"什么|如何|哪些", text))
    )


def _section_query_terms(question: str) -> set[str]:
    tokens = tokenize(question)
    terms = {token for token in tokens if len(token) >= 3}
    for token in tokens:
        if re.fullmatch(r"[a-z]+(?:-[a-z]+)+", token):
            terms.update(part for part in token.split("-") if len(part) >= 3)
    for prefix, suffix in re.findall(
        r"(?<![A-Za-z0-9])([A-Z]{2,})([a-z]{2,})(?![A-Za-z0-9])", str(question or "")
    ):
        terms.update((prefix.casefold(), suffix.casefold()))
    normalized = str(question or "").casefold()
    for phrase, aliases in _SECTION_QUERY_ALIASES.items():
        if phrase in normalized:
            terms.update(aliases)
    return terms


def _source_local_evidence_requested(question: str) -> bool:
    """Gate source-local fallback to explicit evidence-seeking questions."""

    normalized = str(question or "").casefold()
    return bool(_ENGLISH_LISTED_FACT_QUESTION_RE.search(normalized) or re.search(r"\bratio\b", normalized)) or any(
        str(cue).casefold() in normalized for cue in _SOURCE_LOCAL_EVIDENCE_CUES
    )


def _explicit_number_tokens(value: Any) -> set[str]:
    """Return normalized, non-trivial numbers for a routed evidence fallback."""

    tokens = set()
    for token in _EXPLICIT_NUMBER_RE.findall(str(value or "")):
        normalized = token.replace(",", "")
        try:
            significant = float(normalized)
        except ValueError:
            continue
        nearby_rank = bool(
            re.search(
                rf"(?:top|bottom|k)\s*[-_ ]?{re.escape(token)}\b",
                str(value or ""),
                re.IGNORECASE,
            )
        )
        if "," in token or "." in token or significant >= 10 or nearby_rank:
            tokens.add(normalized)
    return tokens


def _numeric_route_evidence_result(
    question: str,
    runtime: Runtime,
    route: Any | None,
) -> dict[str, Any] | None:
    """Recover explicit numeric facts that fall outside a dense top-k."""

    source = str(getattr(route, "document_id", "") or "").strip()
    query_numbers = _explicit_number_tokens(question)
    if not source or not query_numbers:
        return None
    snapshot = _get_lexical_snapshot(runtime)
    candidates: list[tuple[int, int, int]] = []
    query_terms = set(tokenize(question))
    for index, (text, metadata) in enumerate(zip(snapshot.texts, snapshot.metadatas)):
        if str(metadata.get("source", "")).strip() != source:
            continue
        if metadata.get("type") in {"formula", "figure"}:
            continue
        matched = query_numbers & _explicit_number_tokens(text)
        if not matched:
            continue
        overlap = len(query_terms & set(tokenize(_lexical_search_text(text, metadata))))
        candidates.append((-len(matched), -overlap, index))
    if not candidates:
        return None
    indices = [row[2] for row in sorted(candidates)[:4]]
    return {
        "ids": [[snapshot.ids[index] for index in indices]],
        "documents": [[snapshot.texts[index] for index in indices]],
        "metadatas": [[
            {**snapshot.metadatas[index], "route_evidence": True}
            for index in indices
        ]],
    }


def _lexical_route_evidence_result(
    question: str,
    runtime: Runtime,
    route: Any | None,
) -> dict[str, Any] | None:
    """Rank a lexical fallback using only the selected source's statistics."""

    source = str(getattr(route, "document_id", "") or "").strip()
    if not source:
        return None
    snapshot = _get_lexical_snapshot(runtime)
    source_indices = [
        index for index in snapshot.lexical_indices
        if str(snapshot.metadatas[index].get("source", "")).strip() == source
    ]
    positions = {
        position: index
        for position, index in enumerate(source_indices)
        if is_table_question(question)
        or snapshot.metadatas[index].get("type", "text") == "text"
    }
    if not positions:
        return None
    expanded_question = " ".join([question, *_section_query_terms(question)])
    local_index = BM25Index([
        _lexical_search_text(snapshot.texts[index], snapshot.metadatas[index])
        for index in source_indices
    ])
    ranking = local_index.retrieve(expanded_question, 6, indices=positions)
    indices = [positions[int(item.key)] for item in ranking]
    if not indices:
        return None
    clauses = [part.strip() for part in re.split(r"[?？]+", question) if len(part.strip()) >= 3]
    if len(clauses) > 1:
        clause_rankings = [
            [item for item in local_index.retrieve(
                " ".join([clause, *_section_query_terms(clause)]), 2, indices=positions
            ) if item.score > 0]
            for clause in clauses[:6]
        ]
        clause_indices = [
            positions[int(items[rank].key)]
            for rank in range(2)
            for items in clause_rankings
            if rank < len(items)
        ]
        indices = list(dict.fromkeys([*clause_indices, *indices]))[:6]
    listed = bool(_LISTED_FACT_QUESTION_RE.search(question))
    if re.search(r"分为|分类|taxonomy|categories", question, re.I) or listed:
        query_terms = _section_query_terms(question)
        english_listed = bool(_ENGLISH_LISTED_FACT_QUESTION_RE.search(question))
        anchor = indices[0] if english_listed else max(
            indices,
            key=lambda index: (
                _header_match_score(
                    str(snapshot.metadatas[index].get("headers", "")), query_terms
                )
                if re.match(r"H[1-6]:", str(snapshot.metadatas[index].get("headers") or ""))
                else 0
            ),
        )
        if english_listed or _header_match_score(str(snapshot.metadatas[anchor].get("headers", "")), query_terms) >= 3:
            # ponytail: a list split across more than three chunks needs a wider window.
            continuation = _section_continuation_indices(
                anchor, snapshot.metadatas, snapshot.texts, source, 2,
                allow_unnumbered_heading=True,
            )
            indices = list(dict.fromkeys([anchor, *continuation, *indices]))[:6]
        for candidate in indices:
            text = str(snapshot.texts[candidate]).rstrip()
            if not text.endswith(":"):
                continue
            continuation = _section_continuation_indices(
                candidate, snapshot.metadatas, snapshot.texts, source, 1
            )
            if not continuation:
                continue
            following = continuation[0]
            metadata = snapshot.metadatas[candidate]
            next_metadata = snapshot.metadatas[following]
            if (
                metadata.get("type", "text") == "text"
                and isinstance(metadata.get("page"), int)
                and next_metadata.get("page") == metadata["page"] + 1
                and not next_metadata.get("headers")
                and _is_cross_page_continuation(text, snapshot.texts[following].lstrip(" *_`"))
            ):
                indices = list(dict.fromkeys([candidate, following, *indices]))[:6]
                break
    elif re.search(r"为什么|为何|\bwhy\b", question, re.I):
        # A causal explanation often starts in a method section and finishes
        # after a page-boundary table; keep the first lexical section's prose.
        for anchor in indices:
            if not re.match(r"H[2-6]:", str(snapshot.metadatas[anchor].get("headers") or "")):
                continue
            continuation = _section_continuation_indices(
                anchor, snapshot.metadatas, snapshot.texts, source, 2,
                allow_unnumbered_heading=True,
            )
            if continuation:
                indices = list(dict.fromkeys([anchor, *continuation, *indices]))[:6]
            break

    # A proven page-split sentence takes precedence over a referenced caption.
    anchor = indices[0]
    references = re.findall(r"\bFigure\s+(\d+)\b", snapshot.texts[anchor], re.I)
    continuation = _section_continuation_indices(
        anchor, snapshot.metadatas, snapshot.texts, source, 1,
    )
    if continuation:
        following = continuation[0]
        page = snapshot.metadatas[anchor].get("page")
        if (
            snapshot.metadatas[anchor].get("type", "text") == "text"
            and isinstance(page, int)
            and snapshot.metadatas[following].get("page") == page + 1
            and re.match(r"[a-z0-9]", snapshot.texts[following].lstrip(" *_`"))
            and _is_cross_page_continuation(snapshot.texts[anchor], snapshot.texts[following])
        ):
            indices = list(dict.fromkeys([anchor, following, *indices]))[:6]
            references = []
    # Follow one explicit figure reference only when there is no sentence split.
    for index in positions.values():
        caption = re.match(r"\s*Figure\s+(\d+)\s*[:.]", snapshot.texts[index], re.I)
        if caption and caption.group(1) in references and index != indices[0]:
            indices = [indices[0], index, *[item for item in indices[1:] if item != index]][:6]
            break
    return {
        "ids": [[snapshot.ids[index] for index in indices]],
        "documents": [[snapshot.texts[index] for index in indices]],
        "metadatas": [[
            {**snapshot.metadatas[index], "route_evidence": True}
            for index in indices
        ]],
    }


def _missing_identifier_result(
    question: str,
    leading_texts: list[str],
    runtime: Runtime,
    source: str,
) -> dict[str, Any] | None:
    """Keep one exact named-item passage when section expansion hides it."""

    identifiers = set(re.findall(r"\b[A-Za-z][A-Za-z0-9-]*\d[A-Za-z0-9-]*\b", question))
    leading = "\n".join(leading_texts).casefold()
    missing = {term.casefold() for term in identifiers if term.casefold() not in leading}
    if not missing:
        return None
    result = _lexical_route_evidence_result(
        question, runtime, DocumentRoute(source, ())
    )
    if result is None:
        return None
    for doc_id, text, metadata in zip(
        _flat_result_values(result, "ids"),
        _flat_result_values(result, "documents"),
        _flat_result_values(result, "metadatas"),
    ):
        if any(term in str(text).casefold() for term in missing):
            return {
                "ids": [[doc_id]],
                "documents": [[text]],
                "metadatas": [[metadata]],
            }
    return None


def _figure_page_text_result(
    question: str,
    figure_result: dict[str, Any] | None,
    runtime: Runtime,
    source_filter: str | set[str] | None = None,
) -> dict[str, Any] | None:
    """Add same-page prose for an explicit figure query.

    Spatial figure blocks contain labels and coordinates, while the sentence
    that explains a figure's example can remain in an adjacent text chunk.
    Keep a small source/page-local lexical fallback opt-in with the spatial
    evidence path; ordinary text retrieval is unchanged.
    """

    if not figure_result:
        return None
    if not re.search(
        r"正确答案|示例|\b(?:correct\s+answer|example)\b",
        str(question or ""),
        re.IGNORECASE,
    ):
        return None
    figure_metas = _flat_result_values(figure_result, "metadatas")
    figure_pages = {
        (str(meta.get("source", "")).strip(), str(meta.get("page", "")).strip())
        for meta in figure_metas
        if isinstance(meta, dict) and meta.get("source") and meta.get("page") is not None
    }
    if not figure_pages:
        return None
    allowed_sources = _normalise_source_filter(source_filter)
    if allowed_sources:
        figure_pages = {
            pair for pair in figure_pages if pair[0] in allowed_sources
        }
    if not figure_pages:
        return None
    snapshot = _get_lexical_snapshot(runtime)
    candidate_indices = [
        index
        for index, metadata in enumerate(snapshot.metadatas)
        if metadata.get("type", "text") in {"text", "formula"}
        and (
            str(metadata.get("source", "")).strip(),
            str(metadata.get("page", "")).strip(),
        ) in figure_pages
        and not _is_picture_text_chunk(snapshot.texts[index])
    ]
    if not candidate_indices:
        return None
    expanded_question = " ".join([question, *_section_query_terms(question)])
    local_index = BM25Index(
        [
            _lexical_search_text(snapshot.texts[index], snapshot.metadatas[index])
            for index in candidate_indices
        ]
    )
    indices = [
        candidate_indices[int(item.key)]
        for item in local_index.retrieve(expanded_question, 2)
    ]
    if not indices:
        return None
    return {
        "ids": [[snapshot.ids[index] for index in indices]],
        "documents": [[snapshot.texts[index] for index in indices]],
        "metadatas": [[
            {**snapshot.metadatas[index], "figure_page_evidence": True}
            for index in indices
        ]],
    }


def _header_match_score(header: str, query_terms: set[str]) -> int:
    """Score the deepest heading more heavily than inherited parent headings."""

    parts = [part.strip() for part in str(header).split(">") if part.strip()]
    if not parts:
        return 0
    deepest = set(tokenize(parts[-1]))
    score = len(deepest & query_terms) * 3
    if len(parts) > 1:
        score += len(set(tokenize(parts[-2])) & query_terms)
    return score


def _section_expansion_result(
    question: str,
    base_result: dict[str, Any],
    runtime: Runtime,
    route: Any | None = None,
    source_filter: str | set[str] | None = None,
) -> dict[str, Any] | None:
    """Add same-section chunks for composite questions without cross-paper mixing.

    PDF-to-Markdown chunkers keep the heading path in ``metadata['headers']``.
    A multi-fact question can retrieve a section's overview while missing the
    immediately following chunk that contains a threshold or tool name. This
    bounded expansion reads the existing collection, stays within the source
    of the highest-ranked candidate, selects the strongest matching header, and
    adds at most a small neighborhood of chunks before the final context cap.
    It never invents text or facts or mixes papers merely because their headings
    use the same generic terms.
    """

    if not _is_composite_fact_question(question):
        return None
    base_ids = [str(value) for value in _flat_result_values(base_result, "ids")]
    base_metas = _flat_result_values(base_result, "metadatas")
    anchor_ids = base_ids[: runtime.config.context_k]
    anchor_metas = base_metas[: runtime.config.context_k]
    anchor_sources = {
        str(meta.get("source", ""))
        for meta in anchor_metas
        if isinstance(meta, dict) and meta.get("source")
    }
    if len(anchor_sources) > 1:
        return None
    base_source = ""
    for meta in anchor_metas:
        if isinstance(meta, dict) and meta.get("source"):
            base_source = str(meta["source"])
            break
    if not base_source:
        return None
    all_result = runtime.collection.get(include=["documents", "metadatas"])
    all_texts = _flat_result_values(all_result, "documents")
    all_metas = _flat_result_values(all_result, "metadatas")
    corpus_sources = {
        str(meta.get("source", ""))
        for meta in all_metas
        if isinstance(meta, dict) and meta.get("source")
    }
    allowed_sources = _normalise_source_filter(source_filter)
    route_source = str(getattr(route, "document_id", "") or "").strip()
    if route_source and allowed_sources and route_source not in allowed_sources:
        route_source = ""
    if allowed_sources:
        if len(allowed_sources) == 1:
            selected_source = next(iter(allowed_sources))
        elif route_source:
            selected_source = route_source
        else:
            return None
    else:
        if len(corpus_sources) > 1 and not route_source:
            return None
        selected_source = route_source or base_source
    if selected_source not in corpus_sources:
        return None
    query_terms = _section_query_terms(question)
    if route is not None:
        query_terms.difference_update(
            str(token).casefold()
            for token in getattr(route, "distinctive_tokens", ())
        )
    if not query_terms:
        return None

    header_rows: list[tuple[int, int, str, str]] = []
    for index, raw_meta in enumerate(all_metas):
        metadata = raw_meta if isinstance(raw_meta, dict) else {}
        source = str(metadata.get("source", ""))
        header = str(metadata.get("headers", ""))
        if source != selected_source or not re.match(r"H[1-6]:", header):
            continue
        score = _header_match_score(header, query_terms)
        if score:
            header_rows.append((score, index, source, header))
    if not header_rows:
        return None

    anchor_headers = {
        (str(meta.get("source", "")), str(meta.get("headers", "")))
        for meta in anchor_metas
        if isinstance(meta, dict) and meta.get("headers")
    }
    scored_headers = [
        row for row in header_rows if (row[2], row[3]) in anchor_headers
    ] or header_rows
    best_score = max(row[0] for row in scored_headers)
    selected_headers = {
        (row[2], row[3]) for row in scored_headers if row[0] == best_score
    }
    all_ids = _flat_result_values(all_result, "ids")
    all_id_to_index = {str(doc_id): index for index, doc_id in enumerate(all_ids)}
    anchors = [
        all_id_to_index[doc_id]
        for doc_id in anchor_ids
        if doc_id in all_id_to_index
        and (
            str(
                all_metas[all_id_to_index[doc_id]].get("source", "")
                if isinstance(all_metas[all_id_to_index[doc_id]], dict)
                else ""
            ),
            str(
                all_metas[all_id_to_index[doc_id]].get("headers", "")
                if isinstance(all_metas[all_id_to_index[doc_id]], dict)
                else ""
            ),
        ) in selected_headers
    ]
    if not anchors:
        return None
    selected_indices = [
        index
        for index, raw_meta in enumerate(all_metas)
        if (
            str(raw_meta.get("source", "")) if isinstance(raw_meta, dict) else "",
            str(raw_meta.get("headers", "")) if isinstance(raw_meta, dict) else "",
        ) in selected_headers
        and index < len(all_texts)
        and index < len(all_ids)
    ]
    selected_indices.sort(
        key=lambda index: (
            min((abs(index - anchor) for anchor in anchors), default=index),
            index,
        )
    )
    # A section header is often present only on its first chunk.  Add
    # contiguous, headerless text continuations before the final context cap;
    # table/figure blocks and the next explicit heading remain boundaries.
    continuation_indices: list[int] = []
    continuation_headers: dict[int, str] = {}
    for anchor in anchors:
        anchor_metadata = (
            all_metas[anchor] if isinstance(all_metas[anchor], dict) else {}
        )
        section_header = str(anchor_metadata.get("headers") or "").strip()
        for index in _section_continuation_indices(
            anchor,
            all_metas,
            all_texts,
            selected_source,
            MAX_SECTION_EXPANSION_CHUNKS,
        ):
            continuation_indices.append(index)
            if section_header:
                continuation_headers.setdefault(index, section_header)
    selected_indices = [
        *anchors,
        *continuation_indices,
        *[
            index
            for index in selected_indices
            if index not in anchors and index not in continuation_indices
        ],
    ]
    selected_ids: list[str] = []
    selected_docs: list[str] = []
    selected_metas: list[dict[str, Any]] = []
    cursor = 0
    while cursor < len(selected_indices) and len(selected_ids) < MAX_SECTION_EXPANSION_CHUNKS:
        index = selected_indices[cursor]
        metadata = all_metas[index] if isinstance(all_metas[index], dict) else {}
        if index in continuation_headers and not str(metadata.get("headers") or "").strip():
            # Keep the source metadata's real ``headers`` untouched while
            # exposing the inherited section used for deterministic evidence
            # auditing and citations.
            metadata = {
                **metadata,
                "section_context": continuation_headers[index],
            }
        text = str(all_texts[index])
        if cursor + 1 < len(selected_indices):
            next_index = selected_indices[cursor + 1]
            next_meta = all_metas[next_index] if isinstance(all_metas[next_index], dict) else {}
            next_text = str(all_texts[next_index])
            # ponytail: only compact short, obvious page splits; widen this
            # if longer splits are measured crowding out needed evidence.
            if (
                len(text) < 160
                and metadata.get("type", "text") == next_meta.get("type", "text") == "text"
                and metadata.get("source") == next_meta.get("source")
                and metadata.get("page") is not None
                and next_meta.get("page") is not None
                and metadata["page"] != next_meta["page"]
                and isinstance(metadata.get("chunk_index"), int)
                and next_meta.get("chunk_index") == metadata["chunk_index"] + 1
                and not next_meta.get("headers")
                and _is_cross_page_continuation(
                    re.sub(r"\s*\d{1,4}\s*$", "", text),
                    next_text.lstrip(" *_`"),
                )
            ):
                text += "\n\n" + next_text
                metadata = {
                    **metadata,
                    "window_chunk_ids": [str(all_ids[index]), str(all_ids[next_index])],
                    "window_chunk_indices": [metadata["chunk_index"], next_meta["chunk_index"]],
                    "window_pages": [metadata["page"], next_meta["page"]],
                }
                cursor += 1
        doc_id = str(all_ids[index])
        selected_ids.append(doc_id)
        selected_docs.append(text)
        selected_metas.append(dict(metadata))
        cursor += 1

    if not selected_ids:
        return None
    return {
        "ids": [selected_ids],
        "documents": [selected_docs],
        "metadatas": [selected_metas],
    }


def _is_cross_page_continuation(previous: str, following: str) -> bool:
    """Recognize a sentence split at a PDF page boundary."""

    previous = str(previous or "")
    # Ignore a standalone page number and footnotes cited in this same chunk
    # for boundary detection only; the supplied evidence remains untouched.
    previous = re.sub(r"\n\s*\d{1,4}\s*$", "", previous)
    for number in set(_FOOTNOTE_MARKER_RE.findall(previous)):
        previous = re.sub(rf"(?m)^\s*(?:>\s*)?{number}(?!\d)\S[^\n]*$", "", previous)
    previous = re.sub(r"\s+", " ", previous).strip()
    following = re.sub(r"\s+", " ", str(following or "")).strip()
    if not previous or not following:
        return False
    if not re.search(
        r"(?:\b(?:and|or|of|with|to|for|a|an|the)\s*|[,;:，、和及与的在]\s*)$",
        previous,
        re.IGNORECASE,
    ):
        return False
    return bool(re.match(r"(?:[0-9A-Za-z]|[，。；：、])", following))


def _parent_window_contexts(
    texts: list[str],
    ids: list[str],
    metas: list[dict[str, Any]],
    runtime: Runtime,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Attach same-page neighbors to top text anchors without adding slots.

    Uploaded chunks carry a source-local ``chunk_index``.  The optional parent
    window uses that stable sequence rather than Collection.get() ordering,
    skips tables/references and already-selected contexts, and records every
    contributing chunk ID in returned metadata.  The returned text is therefore
    exactly what generation receives while citations show both pages for a
    cross-page sentence.
    """

    if not runtime.config.parent_window or not texts:
        return list(texts), [dict(meta) for meta in metas]
    snapshot = _get_lexical_snapshot(runtime)
    sequence: dict[tuple[str, int], tuple[str, str, dict[str, Any]]] = {}
    for doc_id, text, raw_meta in zip(
        snapshot.ids, snapshot.texts, snapshot.metadatas
    ):
        metadata = raw_meta if isinstance(raw_meta, dict) else {}
        source = str(metadata.get("source", ""))
        try:
            chunk_index = int(metadata.get("chunk_index"))
        except (TypeError, ValueError):
            continue
        if source:
            sequence[(source, chunk_index)] = (doc_id, text, metadata)

    selected_ids = {str(doc_id) for doc_id in ids}
    effective_texts = list(texts)
    effective_metas = [dict(meta) for meta in metas]
    for position in range(min(MAX_PARENT_WINDOW_ANCHORS, len(effective_texts))):
        metadata = effective_metas[position]
        if metadata.get("type", "text") != "text":
            continue
        # Figure/OCR blocks often contain several unrelated axes and sample
        # counts.  Expanding them with adjacent prose increases numeric
        # ambiguity, so retain the ranked block but do not create a window.
        if _is_picture_text_chunk(effective_texts[position]):
            continue
        source = str(metadata.get("source", ""))
        page = metadata.get("page")
        header = str(metadata.get("headers", ""))
        try:
            anchor_chunk_index = int(metadata.get("chunk_index"))
        except (TypeError, ValueError):
            continue
        if not source or page is None or _REFERENCE_HEADER_RE.search(header):
            continue
        anchor_record = sequence.get((source, anchor_chunk_index))
        if anchor_record is None:
            continue
        included = [(anchor_chunk_index, *anchor_record)]
        for neighbor_index in (anchor_chunk_index - 1, anchor_chunk_index + 1):
            record = sequence.get((source, neighbor_index))
            if record is None:
                continue
            neighbor_id, neighbor_text, neighbor_meta = record
            if str(neighbor_id) in selected_ids:
                continue
            if neighbor_meta.get("type", "text") != "text":
                continue
            if _is_picture_text_chunk(neighbor_text):
                continue
            if neighbor_meta.get("page") != page:
                previous, following = (
                    (neighbor_text, effective_texts[position])
                    if neighbor_index < anchor_chunk_index
                    else (effective_texts[position], neighbor_text)
                )
                if not _is_cross_page_continuation(previous, following):
                    continue
            if _REFERENCE_HEADER_RE.search(str(neighbor_meta.get("headers", ""))):
                continue
            included.append(
                (neighbor_index, str(neighbor_id), str(neighbor_text), neighbor_meta)
            )
        included.sort(key=lambda row: row[0])
        if len(included) <= 1:
            continue
        effective_texts[position] = "\n\n".join(row[2] for row in included)
        effective_metas[position]["window_chunk_indices"] = [
            row[0] for row in included
        ]
        effective_metas[position]["window_chunk_ids"] = [row[1] for row in included]
        pages = list(dict.fromkeys(
            row[3].get("page") for row in included if row[3].get("page") is not None
        ))
        if len(pages) > 1:
            effective_metas[position]["window_pages"] = pages
        effective_metas[position]["window_added_character_count"] = sum(
            len(row[2]) for row in included if row[1] != str(ids[position])
        )
    return effective_texts, effective_metas


def _attach_matching_footnotes(
    texts: list[str],
    metas: list[dict[str, Any]],
    runtime: Runtime,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Attach only uniquely matched, same-source/page footnote lines."""

    requested = {
        (str(meta.get("source", "")), meta.get("page"), number)
        for text, meta in zip(texts, metas)
        if meta.get("type", "text") == "text" and meta.get("source") and meta.get("page") is not None
        for number in _FOOTNOTE_MARKER_RE.findall(text)
    }
    if not requested:
        return texts, metas
    snapshot = _get_lexical_snapshot(runtime)
    matches: dict[tuple[str, Any, str], set[tuple[str, str]]] = {}
    for doc_id, text, meta in zip(snapshot.ids, snapshot.texts, snapshot.metadatas):
        if meta.get("type", "text") != "text":
            continue
        source, page = str(meta.get("source", "")), meta.get("page")
        if not any(key[:2] == (source, page) for key in requested):
            continue
        for line in text.splitlines():
            match = _FOOTNOTE_LINE_RE.match(line)
            if match and (source, page, match.group(1)) in requested:
                matches.setdefault((source, page, match.group(1)), set()).add((doc_id, line.strip()))

    result_texts = list(texts)
    result_metas = [dict(meta) for meta in metas]
    for index, (text, meta) in enumerate(zip(texts, metas)):
        key_base = (str(meta.get("source", "")), meta.get("page"))
        for number in dict.fromkeys(_FOOTNOTE_MARKER_RE.findall(text)):
            candidates = matches.get((*key_base, number), set())
            lines = {line for _doc_id, line in candidates}
            if len(lines) != 1 or next(iter(lines)) in text:
                continue
            doc_id, line = min(candidates)
            result_texts[index] += f"\n\n[原文脚注 {number}] {line}"
            footnote_ids = result_metas[index].setdefault("footnote_chunk_ids", [])
            if doc_id not in footnote_ids:
                footnote_ids.append(doc_id)
    return result_texts, result_metas


def _estimated_tokens(text: str) -> int:
    """Approximate input cost; actual tokenization depends on the provider."""
    # ponytail: no portable tokenizer exists for arbitrary compatible APIs;
    # retain headroom and expose the window instead of downloading tokenizers.
    return sum(
        (len(part) + 2) // 3 if part.isascii() and part.isalpha()
        else (len(part) + 1) // 2 if part.isascii() and part.isdigit()
        else 0 if part == " "  # Usually encoded with the following word.
        else (len(part) + 3) // 4 if part.isspace()
        else len(part.encode("utf-8")) if any(ord(char) > 0xFFFF for char in part)
        else len(part)
        for part in re.findall(r"[A-Za-z]+|[0-9]+|\s+|.", text, re.DOTALL)
    )


def _pack_contexts(
    runtime: Runtime,
    system: str,
    instruction: str,
    texts: list[str],
    *,
    labels: list[str] | None = None,
    suffix: str = "",
) -> tuple[list[int], str]:
    """Keep whole evidence chunks, reserving output and chat-template space."""
    limit = runtime.config.llm_context_tokens - runtime.config.llm_max_tokens - 128
    selected: list[int] = []
    parts: list[str] = []
    for index, text in enumerate(texts):
        label = labels[index] if labels else ""
        part = f"【片段 {len(selected) + 1}】{label}\n{text}"
        candidate = instruction + "\n\n" + "\n\n---\n\n".join([*parts, part]) + suffix
        if _estimated_tokens(system) + _estimated_tokens(candidate) <= limit:
            selected.append(index)
            parts.append(part)
    if not selected:
        raise ValueError("输入预算不足以容纳完整资料片段；请缩短问题、拆分长资料，或按服务实际窗口配置 LLM_CONTEXT_TOKENS。")
    return selected, instruction + "\n\n" + "\n\n---\n\n".join(parts) + suffix


def _complete_text(runtime: Runtime, system: str, prompt: str, *, temperature: float = 0.3, json_output: bool = False) -> str:
    if runtime.client is None:
        raise RuntimeError("请先在“设置”页配置模型服务。")
    request: dict[str, Any] = {
        "model": runtime.config.llm_model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": runtime.config.llm_max_tokens,
    }
    if json_output:
        request["response_format"] = {"type": "json_object"}
    if urlsplit(runtime.config.llm_base_url).hostname == "api.deepseek.com":
        request["extra_body"] = {"thinking": {"type": "disabled"}}
    response = runtime.client.chat.completions.create(
        **request,
    )
    choice = response.choices[0]
    text = choice.message.content or ""
    if not text.strip():
        raise ValueError("模型没有返回正文，请检查模型是否将输出预算用于推理。")
    if getattr(choice, "finish_reason", None) == "length":
        if json_output:
            raise ValueError("模型输出达到长度上限，题目尚未生成完整，未载入测评。请提高 LLM_MAX_TOKENS，并为输入保留足够窗口。")
        text += "\n\n⚠️ 模型输出达到长度上限，以上回答尚未完成；请缩小问题范围或提高 LLM_MAX_TOKENS，并确认服务窗口足够。"
    return text


def query_knowledge(
    message: str,
    runtime: Runtime,
    return_contexts: bool = True,
    source_filter: str | list[str] | tuple[str, ...] | None = None,
) -> str | dict[str, Any]:
    """Retrieve evidence and generate an answer.

    When ``return_contexts`` is true, ``contexts`` is exactly the list joined
    into the generation prompt. IDs and metadata are returned for evaluation.
    ``source_filter`` is the source filename allowlist from
    the UI's document selection or an isolated evaluation; empty means all documents.
    """

    if not message or not message.strip():
        result = {"answer": "请输入一个问题。", "contexts": [], "context_ids": [], "context_metadatas": []}
        return result if return_contexts else result["answer"]
    if runtime.collection.count() == 0:
        answer = "📚 知识库为空，请先上传文档。"
        return {"answer": answer, "contexts": [], "context_ids": [], "context_metadatas": []} if return_contexts else answer

    allowed_sources = _normalise_source_filter(source_filter)
    vision_answer, vision_error, vision_metadata = _vision_answer(
        message,
        runtime,
        source_filter=allowed_sources,
    )
    if vision_answer is not None:
        result = {
            "answer": vision_answer,
            "contexts": [],
            "context_ids": [],
            "context_metadatas": [vision_metadata] if vision_metadata else [],
        }
        return result if return_contexts else vision_answer

    variants = (
        query_variants(message)
        if runtime.config.query_decomposition
        else [message]
    )
    if not variants:
        variants = [message]
    configured_candidate_k = (
        runtime.config.hybrid_candidate_k
        if runtime.config.retrieval_mode == "hybrid"
        else runtime.config.retrieval_k
    )
    candidate_k = min(configured_candidate_k, runtime.collection.count())
    route = None
    routed_sources: list[str] = []
    routed_terms: dict[str, set[str]] = {}
    routed_variant_ids: list[str] = []
    routed_evidence_ids: list[str] = []
    routed_evidence_results: list[dict[str, Any]] = []
    cover_all_selected_sources = _all_selected_sources_requested(message, allowed_sources)
    if runtime.config.document_routing or len(allowed_sources) > 1:
        route_snapshot = _get_lexical_snapshot(runtime)
        route = route_snapshot.router.route(message) if route_snapshot.router else None
        if route is not None and allowed_sources and route.document_id not in allowed_sources:
            route = None
        if route_snapshot.router:
            # A multi-document UI selection is an explicit scope. Reuse the
            # existing name resolver for named papers within it, not global
            # automatic routing or mandatory coverage of every selected file.
            scope_variants = query_variants(message) if len(allowed_sources) > 1 else variants
            for variant in scope_variants:
                candidate = route_snapshot.router.route(variant)
                if candidate is not None and (
                    not allowed_sources or candidate.document_id in allowed_sources
                ):
                    routed_terms.setdefault(candidate.document_id, set()).update(candidate.distinctive_tokens)
            routed_sources = list(routed_terms)
            if len(allowed_sources) > 1 and routed_sources:
                variants = scope_variants
    if cover_all_selected_sources:
        routed_sources = sorted(allowed_sources)
    dense_results: list[dict[str, Any]] = []
    query_plans: list[tuple[str, str | None]] = [(variant, None) for variant in variants]
    if cover_all_selected_sources:
        query_plans.extend((message, source) for source in sorted(allowed_sources))
    for variant, forced_source in query_plans:
        question_embedding = runtime.embedding_model.encode(variant).tolist()
        query_kwargs = {
            "query_embeddings": [question_embedding],
            "n_results": candidate_k,
            "include": ["documents", "metadatas", "distances"],
        }
        # Formula blocks are a gated supplementary channel.  Excluding them
        # from ordinary candidates keeps recovered equations from perturbing
        # normal prose/table retrieval.  Figure blocks follow the same rule
        # only when that opt-in channel is enabled.
        query_conditions: list[dict[str, Any]] = [{"type": {"$ne": "formula"}}]
        if runtime.config.spatial_figure_evidence:
            query_conditions.append({"type": {"$ne": "figure"}})
        variant_route = DocumentRoute(forced_source, ()) if forced_source else route
        if forced_source is None and len(routed_sources) > 1 and route_snapshot.router:
            variant_route = route_snapshot.router.route(variant)
        if variant_route is not None and allowed_sources and variant_route.document_id not in allowed_sources:
            variant_route = None
        source_clause = _source_where_clause({forced_source} if forced_source else allowed_sources)
        if source_clause is not None:
            query_conditions.append(source_clause)
        if variant_route is not None and forced_source is None:
            # ``source`` is written for every uploaded chunk.  The filter is
            # only applied after the conservative lexical router found one
            # unique source; ambiguous questions intentionally keep the global
            # search for every variant.
            query_conditions.append({"source": {"$eq": variant_route.document_id}})
        query_kwargs["where"] = (
            query_conditions[0]
            if len(query_conditions) == 1
            else {"$and": query_conditions}
        )
        variant_result = runtime.collection.query(**query_kwargs)
        dense_results.append(variant_result)
        # A UI document selection or isolated evaluation supplies the source
        # even when the router cannot infer an identifier from the question.
        # Reuse bounded evidence fallbacks without changing unfiltered retrieval.
        evidence_route = variant_route
        if evidence_route is None and len(allowed_sources) == 1:
            evidence_route = DocumentRoute(next(iter(allowed_sources)), ())
        if evidence_route is not None:
            routed_variant_ids.extend(
                str(value) for value in _flat_result_values(variant_result, "ids")[:12]
            )
            source_local_requested = (
                _source_local_evidence_requested(message)
                and figure_reference_from_question(message) is None
            )
            if (
                len(routed_sources) > 1
                or (evidence_route is not None and source_local_requested)
                or (len(allowed_sources) == 1 and source_local_requested)
            ):
                # A source-only clause (for example ``TANQ`` after splitting
                # ``TANQ 和 FigEx ...``) is useful for routing but carries no
                # retrieval intent. Reuse the full question for the bounded
                # source-local evidence fallbacks so the route does not lose
                # the shared predicate.
                # Keep a real decomposed clause for source-local lexical
                # evidence.  A bare routed identifier still needs the full
                # question because it carries no retrieval intent.
                variant_has_question_cue = bool(
                    re.search(
                        r"什么|哪些|多少|如何|是否|分别|配置|指标|问题|条件|"
                        r"what|which|how|whether|number|count|score|value",
                        variant,
                        re.IGNORECASE,
                    )
                )
                evidence_question = (
                    message
                    if (
                        (len(allowed_sources) == 1 and not variant_has_question_cue)
                        or (
                            not variant_has_question_cue
                            and not _is_composite_fact_question(variant)
                            and _is_composite_fact_question(message)
                        )
                    )
                    else variant
                )
                variant_numeric = _numeric_route_evidence_result(
                    evidence_question,
                    runtime,
                    evidence_route,
                )
                if variant_numeric is not None:
                    routed_evidence_results.append(variant_numeric)
                    routed_evidence_ids.extend(
                        str(value)
                        for value in _flat_result_values(variant_numeric, "ids")
                    )
                lexical_question = evidence_question
                if len(routed_sources) > 1:
                    # The source is already fixed. Repeating its identifier
                    # otherwise ranks title/abstract mentions above the predicate.
                    for token in evidence_route.distinctive_tokens:
                        lexical_question = re.sub(
                            rf"(?<![A-Za-z0-9]){re.escape(token)}(?![A-Za-z0-9])",
                            " ", lexical_question, flags=re.I,
                        )
                variant_lexical = _lexical_route_evidence_result(
                    lexical_question,
                    runtime,
                    evidence_route,
                )
                if variant_lexical is not None:
                    routed_evidence_results.append(variant_lexical)
                    routed_evidence_ids.extend(
                        str(value)
                        for value in _flat_result_values(variant_lexical, "ids")
                    )
                variant_section = _section_expansion_result(
                    evidence_question,
                    variant_result,
                    runtime,
                    route=evidence_route,
                    source_filter=allowed_sources,
                )
                if variant_section is not None:
                    routed_evidence_results.append(variant_section)
                    routed_evidence_ids.extend(
                        str(value)
                        for value in _flat_result_values(variant_section, "ids")
                    )
    dense = _fuse_dense_results(
        dense_results,
        candidate_k,
        runtime.config.hybrid_rrf_k,
    )
    # Keep a chunk endorsed by both explicit subquestions before source-local
    # lexical candidates can consume the whole context budget.
    shared_clause_result = None
    clauses = [part.strip() for part in re.split(r"[?？]+", message) if len(part.strip()) >= 3]
    if (
        len(allowed_sources) == 1
        and len(clauses) == 2
        and runtime.config.retrieval_mode == "dense"
        and _source_local_evidence_requested(message)
        and not is_table_question(message)
        and figure_reference_from_question(message) is None
    ):
        clause_ids = [
            set(_flat_result_values(runtime.collection.query(
                **{
                    **query_kwargs,
                    "query_embeddings": [runtime.embedding_model.encode(clause).tolist()],
                    "n_results": min(3, candidate_k),
                }
            ), "ids"))
            for clause in clauses
        ]
        shared_ids = clause_ids[0] & clause_ids[1]
        for index, doc_id in enumerate(_flat_result_values(dense, "ids")[:4]):
            if doc_id in shared_ids:
                shared_clause_result = {
                    key: [[_flat_result_values(dense, key)[index]]]
                    for key in ("ids", "documents", "metadatas")
                }
                break
    if runtime.config.retrieval_mode == "hybrid":
        dense = _hybrid_fused_result(
            message,
            dense,
            runtime,
            candidate_k,
            source_filter=(
                allowed_sources
                or (route.document_id if route is not None else None)
            ),
            lexical_queries=variants,
        )
        dense = _cross_encoder_reranked_result(message, dense, runtime)
    section_result = _section_expansion_result(
        message,
        dense,
        runtime,
        route=route,
        source_filter=allowed_sources,
    )
    formula_result = None
    use_formula_evidence = formula_evidence_enabled(message, runtime.config)
    if use_formula_evidence:
        formula_snapshot = _get_lexical_snapshot(runtime)
        source_ids = {
            str(metadata.get("source", "")).strip()
            for metadata in formula_snapshot.metadatas
            if isinstance(metadata, dict) and str(metadata.get("source", "")).strip()
        }
        formula_indices: list[int] = []
        # Composite questions may route each clause to a different paper. Run
        # the opt-in formula aid per formula-bearing variant so a route chosen
        # for the first clause cannot hide an equation needed by a later one.
        for variant in variants:
            if not formula_evidence_enabled(variant, runtime.config):
                continue
            variant_route = (
                formula_snapshot.router.route(variant)
                if formula_snapshot.router is not None
                else None
            )
            allowed_formula_indices: list[int]
            if allowed_sources:
                allowed_formula_indices = [
                    index
                    for index in range(len(formula_snapshot.texts))
                    if str(formula_snapshot.metadatas[index].get("source", ""))
                    in allowed_sources
                ]
            elif variant_route is not None:
                allowed_formula_indices = [
                    index
                    for index in range(len(formula_snapshot.texts))
                    if str(formula_snapshot.metadatas[index].get("source", ""))
                    == variant_route.document_id
                ]
            elif len(source_ids) == 1:
                # A single-source collection is safe for this opt-in lexical aid.
                allowed_formula_indices = list(range(len(formula_snapshot.texts)))
            else:
                # Do not let an unqualified multi-paper question borrow an
                # equation from an unrelated source.
                allowed_formula_indices = []
            for index in formula_evidence_indices(
                variant,
                formula_snapshot.texts,
                formula_snapshot.metadatas,
                max_results=8,
                allowed_indices=allowed_formula_indices,
            ):
                if index not in formula_indices:
                    formula_indices.append(index)
        if formula_indices:
            formula_result = {
                "ids": [[formula_snapshot.ids[index] for index in formula_indices]],
                "documents": [[formula_snapshot.texts[index] for index in formula_indices]],
                "metadatas": [[
                    {**formula_snapshot.metadatas[index], "formula_evidence": True}
                    for index in formula_indices
                ]],
            }
    limitation_result = None
    if is_limitation_question(message):
        limitation_snapshot = _get_lexical_snapshot(runtime)
        source_ids = {
            str(metadata.get("source", "")).strip()
            for metadata in limitation_snapshot.metadatas
            if isinstance(metadata, dict) and str(metadata.get("source", "")).strip()
        }
        allowed_limitation_indices: list[int] | None
        if allowed_sources:
            allowed_limitation_indices = [
                index
                for index in range(len(limitation_snapshot.texts))
                if str(limitation_snapshot.metadatas[index].get("source", ""))
                in allowed_sources
            ]
        elif route is not None:
            allowed_limitation_indices = [
                index
                for index in range(len(limitation_snapshot.texts))
                if str(limitation_snapshot.metadatas[index].get("source", ""))
                == route.document_id
            ]
        elif len(source_ids) == 1:
            allowed_limitation_indices = list(range(len(limitation_snapshot.texts)))
        else:
            # Keep an unqualified multi-paper query from borrowing a limitation
            # passage from an unrelated source.
            allowed_limitation_indices = []
        limitation_indices = limitation_evidence_indices(
            message,
            limitation_snapshot.texts,
            limitation_snapshot.metadatas,
            max_results=6,
            allowed_indices=allowed_limitation_indices,
        )
        if limitation_indices:
            limitation_result = {
                "ids": [[limitation_snapshot.ids[index] for index in limitation_indices]],
                "documents": [[limitation_snapshot.texts[index] for index in limitation_indices]],
                "metadatas": [[
                    {**limitation_snapshot.metadatas[index], "limitation_evidence": True}
                    for index in limitation_indices
                ]],
            }
    figure_result = None
    figure_page_result = None
    explicit_figure_reference = figure_reference_from_question(message)
    if runtime.config.spatial_figure_evidence and explicit_figure_reference is not None:
        # Build the lexical snapshot before the exact figure lookup so the
        # same-page prose helper reuses it instead of issuing a second get().
        _get_lexical_snapshot(runtime)
        figure_kind, figure_number = explicit_figure_reference
        figure_conditions: list[dict[str, Any]] = [
            {"type": {"$eq": "figure"}},
            {"figure_kind": {"$eq": figure_kind}},
            {"figure_number": {"$eq": figure_number}},
        ]
        source_clause = _source_where_clause(allowed_sources)
        if source_clause is not None:
            figure_conditions.append(source_clause)
        elif route is not None:
            figure_conditions.append({"source": {"$eq": route.document_id}})
        raw_figure_result = runtime.collection.get(
            where={"$and": figure_conditions},
            include=["documents", "metadatas"],
        )
        # ``collection.get`` is flat whereas normal query results are nested.
        # Wrap it so the common merge path can place exact Figure N evidence
        # before dense candidates without special-casing IDs or metadata.
        figure_result = {
            "ids": [raw_figure_result.get("ids") or []],
            "documents": [raw_figure_result.get("documents") or []],
            "metadatas": [raw_figure_result.get("metadatas") or []],
        }
        figure_page_result = _figure_page_text_result(
            message,
            figure_result,
            runtime,
            source_filter=allowed_sources,
        )
    table_results = None
    if is_table_question(message):
        table_where: dict[str, Any] = {"type": "table"}
        source_clause = _source_where_clause(allowed_sources)
        if source_clause is not None:
            table_where = {"$and": [{"type": {"$eq": "table"}}, source_clause]}
        elif route is not None:
            # A source route already narrowed the dense/lexical candidate pool.
            # Keep the deterministic table scan in that same scope; otherwise
            # Table N from a different uploaded paper could satisfy the same
            # row/column names before the routed paper is inspected.
            table_where = {
                "$and": [
                    {"type": {"$eq": "table"}},
                    {"source": {"$eq": route.document_id}},
                ]
            }
        table_results = runtime.collection.get(
            where=table_where,
            include=["documents", "metadatas"],
        )
    additional_results = [
        result
        for result in (
            limitation_result,
            figure_result,
            figure_page_result,
            formula_result,
            shared_clause_result,
            *routed_evidence_results,
            section_result,
        )
        if result is not None
    ]
    if len(allowed_sources) == 1 and not is_table_question(message) and explicit_figure_reference is None:
        preview_texts, _, _ = _merge_results(dense, table_results, additional_results)
        rescue = _missing_identifier_result(
            message,
            preview_texts[: runtime.config.context_k],
            runtime,
            next(iter(allowed_sources)),
        )
        if rescue is not None:
            additional_results.insert(0, rescue)
    retrieved_texts, retrieved_ids, retrieved_metas = _merge_results(
        dense, table_results, additional_results or None
    )
    if allowed_sources:
        filtered = [
            (text, doc_id, metadata)
            for text, doc_id, metadata in zip(
                retrieved_texts, retrieved_ids, retrieved_metas
            )
            if str(metadata.get("source", "")).strip() in allowed_sources
        ]
        retrieved_texts = [row[0] for row in filtered]
        retrieved_ids = [row[1] for row in filtered]
        retrieved_metas = [row[2] for row in filtered]
    if not retrieved_texts:
        answer = "未找到相关内容，请换个问法。"
        if vision_error:
            answer += f"\n\n⚠️ {vision_error}"
        return {"answer": answer, "contexts": [], "context_ids": [], "context_metadatas": []} if return_contexts else answer

    explicit_table_number = table_number_from_question(message)
    matching_tables = matching_table_indices(message, retrieved_texts, retrieved_metas)
    if explicit_table_number is not None and not matching_tables:
        answer = f"在 Table {explicit_table_number} 中未找到可用的结构化表格，不能用其他表格替代。"
        result = {
            "answer": answer,
            "contexts": [],
            "context_ids": [],
            "context_metadatas": [],
        }
        return result if return_contexts else answer

    order, note, filtered_texts = rerank_table_first(message, retrieved_texts, retrieved_metas)
    if explicit_figure_reference is not None and runtime.config.spatial_figure_evidence:
        matching_figures = matching_figure_indices(
            message,
            filtered_texts,
            retrieved_metas,
        )
        if matching_figures:
            matching_set = set(matching_figures)
            order = matching_figures + [
                index
                for index in order
                if index not in matching_set
                and retrieved_metas[index].get("type") != "figure"
                and not _is_picture_text_chunk(filtered_texts[index])
            ]
        else:
            figure_kind, figure_number = explicit_figure_reference
            label = (
                f"Extended Data Figure {figure_number}"
                if figure_kind == "extended_data_figure"
                else f"Figure {figure_number}"
            )
            figure_note = f"未找到 {label} 的坐标化文字证据，已回退普通文本检索。"
            note = f"{note} {figure_note}".strip()

    if len(routed_sources) > 1:
        id_to_index = {str(doc_id): index for index, doc_id in enumerate(retrieved_ids)}
        order = ensure_source_coverage(
            order,
            retrieved_metas,
            routed_sources,
            runtime.config.context_k,
            preferred_order=[
                id_to_index[doc_id]
                for doc_id in [*routed_evidence_ids, *routed_variant_ids]
                if doc_id in id_to_index
            ],
        )

    # A question that names a table, row, and column is a deterministic cell
    # lookup.  Answer it from the parsed table rather than asking a language
    # model to choose between similarly worded Table 1/Table 2 narratives.
    cell_match = find_table_cell_in_chunks(message, filtered_texts, retrieved_metas)
    if cell_match is not None:
        cell_index, cell = cell_match
        table_number = (
            cell.get("table_number")
            or cell.get("table_label")
            or explicit_table_number
            or "?"
        )
        if "rows" in cell:
            row_texts = []
            for row in cell["rows"]:
                values = "；".join(
                    f"{item['column']}={item['value']}"
                    for item in row.get("values", [])
                )
                descriptors = "；".join(
                    f"{item['column']}={item['value']}"
                    for item in row.get("descriptors", [])
                )
                qualifier = f"（{descriptors}）" if descriptors else ""
                row_label = row["row"]
                if row.get("outer_group"):
                    row_label = f"{row['outer_group']} / {row_label}"
                row_texts.append(f"{row_label}{qualifier}：{values}")
            value_text = "；".join(row_texts)
            cell_context = (
                f"Table {table_number} 结构化多行：{value_text}\n\n"
                f"{filtered_texts[cell_index]}"
            )
            answer = f"根据 Table {table_number} 中相关行，列值为：{value_text}。"
        elif "values" in cell:
            value_text = "；".join(
                f"{item['column']}={item['value']}"
                for item in cell["values"]
            )
            descriptors = "；".join(
                f"{item['column']}={item['value']}"
                for item in cell.get("descriptors", [])
            )
            if descriptors:
                value_text = f"{descriptors}；{value_text}"
            row_label = cell["row"]
            if cell.get("outer_group"):
                row_label = f"{cell['outer_group']} / {row_label}"
            cell_context = (
                f"Table {table_number} 结构化行：行={row_label}；{value_text}\n\n"
                f"{filtered_texts[cell_index]}"
            )
            answer = f"根据 Table {table_number} 中“{row_label}”行，相关列值为：{value_text}。"
        else:
            cell_context = (
                f"Table {table_number} 结构化单元格：行={cell['row']}；"
                f"列={cell['column']}；值={cell['value']}。\n\n{filtered_texts[cell_index]}"
            )
            answer = (
                f"根据 Table {table_number} 中“{cell['row']}”行的“{cell['column']}”列，"
                f"数值为 **{cell['value']}**。"
            )
        table_metadata = (
            retrieved_metas[cell_index]
            if cell_index < len(retrieved_metas)
            and isinstance(retrieved_metas[cell_index], dict)
            else {}
        )
        table_caption = str(table_metadata.get("table_caption", "") or "")
        if table_caption:
            # Keep the caption beside the structured evidence so a returned
            # trace retains shared units/scale notes that are not part of a
            # parsed header (for example a table-wide ×10^-2 footnote).
            cell_context = f"{table_caption}\n\n{cell_context}"
            if re.search(
                r"总计|共有|一共|总数|合计|total|comprises|contains",
                message,
                re.IGNORECASE,
            ) and re.search(r"\d", table_caption):
                answer = f"{table_caption}\n\n{answer}"
        unit_note = _table_caption_unit_note(table_metadata)
        if unit_note and not _TABLE_UNIT_HINT_RE.search(answer):
            answer += unit_note
        metric_note = _table_caption_metric_note(table_metadata, message)
        if metric_note and metric_note not in answer:
            answer += metric_note
        ordered_texts = [cell_context]
        ordered_ids = [retrieved_ids[cell_index]]
        ordered_metas = [retrieved_metas[cell_index]]
        if return_contexts:
            return {
                "answer": answer,
                "contexts": ordered_texts,
                "context_ids": ordered_ids,
                "context_metadatas": ordered_metas,
            }
        source = ordered_metas[0].get("source", "未知")
        page = ordered_metas[0].get("page")
        suffix = f"，第 {page} 页" if page else ""
        suffix += f"（Table {table_number}）"
        return answer + f"\n\n📌 **参考来源：**\n- {source}{suffix}"

    order = order[: runtime.config.context_k]
    ordered_texts = [filtered_texts[index] for index in order]
    ordered_ids = [retrieved_ids[index] for index in order]
    ordered_metas = [retrieved_metas[index] for index in order]
    ordered_texts, ordered_metas = _parent_window_contexts(
        ordered_texts,
        ordered_ids,
        ordered_metas,
        runtime,
    )
    ordered_texts, ordered_metas = _attach_matching_footnotes(
        ordered_texts, ordered_metas, runtime
    )
    ordered_texts = [
        _attach_table_caption(
            _annotate_spatial_context(text)
            if metadata.get("type") == "figure"
            else text,
            metadata,
        )
        for text, metadata in zip(ordered_texts, ordered_metas)
    ]

    source_titles = {}
    if len(routed_sources) > 1:
        for metadata in _get_lexical_snapshot(runtime).metadatas:
            title = re.match(r"H1:\s*(.+?)(?:\s+>\s+|$)", str(metadata.get("headers", "")))
            if title:
                source_titles.setdefault(str(metadata.get("source", "")), title.group(1))
    elif len(allowed_sources) == 1:
        source = next(iter(allowed_sources))
        first_chunk = runtime.collection.get(
            where={"$and": [{"source": {"$eq": source}}, {"chunk_index": {"$eq": 0}}]},
            include=["metadatas"],
            limit=1,
        )
        for metadata in first_chunk.get("metadatas") or []:
            title = re.match(r"H1:\s*(.+?)(?:\s+>\s+|$)", str(metadata.get("headers", "")))
            if title:
                source_titles[source] = title.group(1).strip("*_` ")
    context_labels = []
    for metadata in ordered_metas:
        source = str(metadata.get("source") or "未知来源")
        page = metadata.get("page")
        if metadata.get("window_pages"):
            page = "–".join(str(value) for value in metadata["window_pages"])
        location = f"，第 {page} 页" if page is not None else ""
        if source in source_titles:
            location += f"，文档：{source_titles[source]}"
        if len(routed_sources) > 1 and source in routed_terms:
            location += f"，问题词：{', '.join(sorted(routed_terms[source]))}"
        if metadata.get("type") == "table":
            table_label = metadata.get("table_label") or metadata.get("table_number")
            kind = (
                f"[表格，Table {table_label}]"
                if table_label is not None and str(table_label).strip()
                else "[表格]"
            )
        elif metadata.get("type") == "figure":
            kind = "[图形坐标文字]"
        elif metadata.get("formula_evidence"):
            kind = "[公式候选]"
        elif metadata.get("limitation_evidence"):
            kind = "[限制证据]"
        else:
            kind = ""
        context_labels.append(f"{kind}[来源：{source}{location}]")

    system_prompt = _scientific_system_prompt(message, ordered_metas)
    instructions = ["【参考资料】"]
    if any(metadata.get("type") == "figure" for metadata in ordered_metas):
        instructions.append(
            "【图形坐标约定】图形文字证据使用 PDF 页面坐标：原点在左上角，x 向右增加，"
            "y 向下增加；因此较大的 y 值位于页面下方。"
        )
    if note:
        instructions.append(f"【检索提示】{note}")
    if vision_error:
        instructions.append(f"【检索提示】{vision_error}")
    try:
        selected, user_prompt = _pack_contexts(
            runtime,
            system_prompt,
            "\n\n".join(instructions),
            ordered_texts,
            labels=context_labels,
            suffix=f"\n\n【问题】\n{message}",
        )
    except ValueError as exc:
        answer = f"❌ 调用出错：{exc}"
        result = {
            "answer": answer,
            "contexts": [],
            "context_ids": [],
            "context_metadatas": [],
        }
        return result if return_contexts else answer
    input_limited = len(selected) < len(ordered_texts)
    ordered_texts = [ordered_texts[index] for index in selected]
    ordered_ids = [ordered_ids[index] for index in selected]
    ordered_metas = [ordered_metas[index] for index in selected]
    evidence_ledger = build_evidence_ledger(
        message,
        ordered_texts,
        ordered_metas,
    )
    if evidence_ledger:
        ledger_text = (
            "【事实核对清单】以下内容仅逐字摘自后面的参考片段，不是新增事实；"
            "回答复合问题时请逐项核对其中与问题相关的数字、阈值、工具名和步骤。"
        )
        limit = runtime.config.llm_context_tokens - runtime.config.llm_max_tokens - 128
        for line in evidence_ledger:
            candidate = f"{ledger_text}\n- {line}\n\n{user_prompt}"
            if _estimated_tokens(system_prompt) + _estimated_tokens(candidate) > limit:
                break
            ledger_text += f"\n- {line}"
        if "\n- " in ledger_text:
            user_prompt = f"{ledger_text}\n\n{user_prompt}"

    try:
        answer = _complete_text(
            runtime,
            system_prompt,
            user_prompt,
        )
        answer = supplement_formula_with_evidence(
            message,
            answer,
            ordered_texts,
            ordered_metas,
        )
    except Exception as exc:
        answer = f"❌ 调用出错：{exc}"
    if input_limited:
        answer += (
            f"\n\n⚠️ 输入窗口有限，本次使用了 {len(ordered_texts)} 个完整片段；"
            "其余候选未发送给模型。"
        )
    if vision_error:
        answer += f"\n\n⚠️ {vision_error}"

    has_spatial_figure_context = any(
        metadata.get("type") == "figure" for metadata in ordered_metas
    )
    if (
        _is_composite_fact_question(message)
        and not is_table_question(message)
        and not has_spatial_figure_context
        and re.search(
            r"管道|流程|步骤|构建|pipeline|process|construct|steps",
            message,
            re.IGNORECASE,
        )
    ):
        answer = supplement_answer_with_evidence(answer, message, evidence_ledger)

    answer_validation = None
    if runtime.config.answer_validation:
        answer_validation = validate_answer_against_evidence(
            message,
            answer,
            evidence_ledger,
        )
        if answer_validation.get("status") == "review":
            reasons = "、".join(answer_validation.get("reasons") or [])
            answer += (
                "\n\n⚠️ **证据核对提示**：生成答案可能遗漏参考片段中的高信号内容，"
                f"请人工核对（{reasons or '未分类原因'}）；系统未自动改写或重试。"
            )

    if return_contexts:
        result = {
            "answer": answer,
            "contexts": ordered_texts,
            "context_ids": ordered_ids,
            "context_metadatas": ordered_metas,
        }
        if answer_validation is not None:
            result["answer_validation"] = answer_validation
        return result
    sources = []
    for metadata in ordered_metas:
        source = metadata.get("source", "未知")
        page = metadata.get("page")
        if metadata.get("window_pages"):
            page = "–".join(str(value) for value in metadata["window_pages"])
        suffix = f"，第 {page} 页" if page else ""
        if metadata.get("type") == "table":
            suffix += f"（{metadata.get('table_id', '表格')}）"
        elif metadata.get("type") == "figure":
            suffix += f"（{metadata.get('figure_label', '图形坐标文字')}）"
        sources.append(f"- {source}{suffix}")
    unique_sources = list(dict.fromkeys(sources))
    return answer + "\n\n📌 **参考来源：**\n" + "\n".join(unique_sources)


SCIENTIFIC_SYSTEM_PROMPT = """你是个人知识库助手中的严谨学术问答助手，职责是从给定的参考片段中抽取事实、数值与实验方法论。必须遵守以下规则：

【强制规则 1：数值必须原样引用并指明出处】
- 若参考文本中存在具体数值，回答时必须原样引用，不得四舍五入、改写或推算。
- 引用数值后必须指明出处，格式为：根据参考片段 [X] 所示。
- 问题明确要求计算时，只用原文操作数列式计算并遵守指定精度；其余数值原样引用。

【强制规则 1A：图形坐标文字不得跨视觉组拼接】
- 标记为“图形坐标文字”的片段只来自 PDF 文字层坐标，不等同于图片识别。
- 只有标签和值的水平 x 范围重叠时，才可把它们视为同一视觉组；不得把相邻子图、柱或类别中的数值移接到另一标签。
- 如果图中信息只存在于像素而未出现在文字层，必须说明参考片段不足，不能猜测。

【强制规则 2：趋势判断必须有明确对比依据】
- 仅当问题询问数值升降或因果趋势时，才根据明确对比依据回答；缺少依据则说明无法判断。其他问题不得套用趋势拒答句。仍可分别引用不同论文的目标或方法进行比较。

【强制规则 2A：限制问题要区分限制本身与示例现象】
- 先回答机制性限制及限定条件，再说示例现象；保留原文对照关系的两端，不能用“训练数据限制”等参考片段未出现的机制替代。

【强制规则 3：实验步骤按时间顺序重组】
- 若回答涉及实验步骤或方法流程，请按“第一、第二、第三”的逻辑重组叙述，不得调换核心操作顺序或省略中间步骤。

【强制规则 4：实体联合约束】
- 若问题同时指定表格编号和行/列实体名称，必须将两者视为联合约束条件。
- 若在指定表格中找不到实体名称，回复：“在 Table [编号] 中未找到 [实体名称] 的条目。”
- 严禁跨表取数。

【强制规则 5：结构化单元格证据】
- 参考资料中若出现“结构化表格单元格”行，该行是从指定表格的真实行列交叉处解析出的证据，必须优先采用其中的值。
- 不得用其他 Table 的同名行、叙述性段落或相似数值覆盖该单元格证据。

【强制规则 5A：表格证据一致性】
- 行、列、值已与问题对应时，不因未重复显示题注而拒答。不得在已经给出具体表格数值后否认该值；仅缺失子项说明资料不足。

【强制规则 6：公式与符号必须按证据转录】
- 若问题询问公式、形式化定义、初始化/更新表达式或激活函数，优先转录参考片段中明确出现的等式、括号、参数顺序、上下标和运算符；不得凭语义改写、交换参数或自行补充等价形式。
- 若同一问题涉及多个公式或变量，必须逐项回答并区分每个变量的定义；不要用流程概述替代显式公式。
- 同一公式若因 PDF 分块分布在相邻参考片段，可在不改变符号、顺序和运算符的前提下按片段拼接；只要各部分分别出现在参考片段中，不得仅因没有单行完整公式而拒答。
- 如果参考片段只有“表达如下/producing ... as”之类引导语而没有等式本身，必须说明该公式未出现在参考片段中，不能猜测。

【其他要求】
- 直接回答问题，不展示内部核对过程，不重复结论或自我评价。
- 表格以 Markdown 形式给出，数值问题请直接依据表格行列作答。
- 若问题涉及图片内容：如果参考片段没有标记为“图形坐标文字”的坐标证据，必须说明“该图内容未纳入文本检索范围”；如果存在这类坐标证据，只能依据其中明确出现的标签、数值和坐标范围作答，不能把它当作像素级图片识别。
- 逐一检查全部参考片段，按问题子项分点作答，每项给出结论和引用。完整保留相关数字、阈值、工具/模型名、实体和步骤，不以概述代替具体事实。
- 局部缺证据时，先回答有依据的子项，仅标明缺失项；不整题拒答，不用常识补写。
- 跨论文比较先分别说明各文观点，再标明综合推论；不要求原文已直接比较另一篇论文，不混淆来源。
- 区分论文已经运行的实验与另行提供的数据、摘要或未来工作；“提供了”不能推断为“已输入模型并评测”。
- 若参考片段无法回答问题，请如实说明“资料未提供相关信息”，严禁编造。"""


def _scientific_system_prompt(question: str, metadatas: list[dict[str, Any]]) -> str:
    """Include specialized constraints only when that evidence is in scope."""
    kinds = {metadata.get("type") for metadata in metadatas}
    irrelevant = []
    if not is_table_question(question) and "table" not in kinds:
        irrelevant.extend(("4", "5", "5A"))
    if not is_formula_question(question) and "formula" not in kinds:
        irrelevant.append("6")
    if not irrelevant:
        return SCIENTIFIC_SYSTEM_PROMPT
    return re.sub(
        rf"【强制规则 (?:{'|'.join(irrelevant)})：[^【]*", "", SCIENTIFIC_SYSTEM_PROMPT
    )


def format_evidence_panel(result: dict[str, Any]) -> str:
    """Render the exact contexts used for an answer as readable source excerpts."""

    contexts = list(result.get("contexts") or [])
    metadatas = list(result.get("context_metadatas") or [])
    if not contexts:
        return "提问后，这里会显示回答实际使用的原文片段。"
    sections = []
    for index, context in enumerate(contexts, start=1):
        metadata = metadatas[index - 1] if index <= len(metadatas) else {}
        metadata = metadata if isinstance(metadata, dict) else {}
        source = html.escape(str(metadata.get("source") or "未知来源"))
        page = metadata.get("page")
        if metadata.get("window_pages"):
            page = "–".join(str(value) for value in metadata["window_pages"])
        location = f"，第 {page} 页" if page is not None else ""
        kind = {
            "table": "表格",
            "figure": "图形文字",
            "formula": "公式",
        }.get(str(metadata.get("type", "text")), "正文")
        quoted = "\n".join(
            f"> {line}" if line else ">" for line in str(context).strip().splitlines()
        )
        sections.append(
            f"### 片段 {index}\n**{source}{location} · {kind}**\n\n{quoted}"
        )
    return "\n\n".join(sections)


def _outline_section_candidates(
    documents: list[str],
    metadatas: list[dict[str, Any]],
) -> tuple[list[str], list[str]]:
    """Select one ordered prose block per section, balanced across sources."""

    by_source: dict[str, list[tuple[int, int, str, dict[str, Any]]]] = {}
    for position, (document, raw_metadata) in enumerate(zip(documents, metadatas)):
        metadata = raw_metadata if isinstance(raw_metadata, dict) else {}
        source = str(metadata.get("source") or "未知资料")
        chunk_index = metadata.get("chunk_index")
        order = chunk_index if isinstance(chunk_index, int) else position
        by_source.setdefault(source, []).append((order, position, str(document), metadata))

    source_sections: list[list[tuple[str, str]]] = []
    for source, rows in by_source.items():
        rows.sort(key=lambda row: (row[0], row[1]))
        has_headers = any(
            re.match(r"H[1-6]:", str(row[3].get("headers") or ""))
            for row in rows
        )
        if not has_headers:
            count = min(8, len(rows))
            positions = (
                [0]
                if count == 1
                else sorted({round(index * (len(rows) - 1) / (count - 1)) for index in range(count)})
            )
            source_sections.append([
                (f"{source} · 全文位置 {index + 1}/{len(positions)}", rows[position][2])
                for index, position in enumerate(positions)
            ])
            continue

        grouped: dict[str, list[tuple[int, int, str, dict[str, Any]]]] = {}
        current_header = ""
        for row in rows:
            header = str(row[3].get("headers") or "").strip()
            if re.match(r"H[1-6]:", header):
                leaf = re.sub(
                    r"^H[1-6]:\s*", "", header.split(" > ")[-1]
                ).replace("**", "").strip()
                if leaf and not leaf[0].islower():
                    current_header = header
            if re.search(r"references?|bibliography|参考文献", current_header, re.IGNORECASE):
                break
            if not current_header or re.search(
                r"acknowledg(?:e)?ments?|author contributions?|conflicts? of interest|"
                r"data availability|funding|ethics statement|致谢|作者贡献|利益冲突|数据可用性",
                current_header,
                re.IGNORECASE,
            ):
                continue
            grouped.setdefault(current_header, []).append(row)

        sections = []
        for header, section_rows in grouped.items():
            prose = [
                row
                for row in section_rows
                if str(row[3].get("type") or "text") == "text"
                and len(row[2].strip()) >= 200
            ] or [
                row for row in section_rows
                if str(row[3].get("type") or "text") == "text"
            ] or section_rows
            header_parts = []
            for part in header.split(" > "):
                match = re.match(r"H([1-6]):\s*(.*)", part)
                if match:
                    header_parts.append(
                        (int(match.group(1)), match.group(2).replace("**", "").strip())
                    )
            if len(header_parts) > 1 and header_parts[0][0] == 1:
                header_parts = header_parts[1:]
            clean_header = " > ".join(text for _level, text in header_parts)
            representative = (
                max(prose, key=lambda row: len(row[2]))
                if header.startswith("H1:")
                else prose[0]
            )
            sections.append((f"{source} · {clean_header}", representative[2]))
        source_sections.append(sections)

    candidates: list[str] = []
    labels: list[str] = []
    for section_index in range(max((len(items) for items in source_sections), default=0)):
        for sections in source_sections:
            if section_index < len(sections):
                label, document = sections[section_index]
                labels.append(f"【{label}】")
                candidates.append(document)
    return candidates, labels


def _outline_candidates_within_budget(
    runtime: Runtime,
    system: str,
    instruction: str,
    candidates: list[str],
    labels: list[str],
) -> list[str]:
    """Give every section an equal share of the available input window."""

    if not candidates:
        return []
    empty_parts = [
        f"【片段 {index}】{label}\n" for index, label in enumerate(labels, start=1)
    ]
    fixed_prompt = instruction + "\n\n" + "\n\n---\n\n".join(empty_parts)
    limit = runtime.config.llm_context_tokens - runtime.config.llm_max_tokens - 128
    available = max(
        1,
        limit - _estimated_tokens(system) - _estimated_tokens(fixed_prompt) - 64,
    )
    per_section = max(1, available // len(candidates))
    compacted = []
    for candidate in candidates:
        if _estimated_tokens(candidate) <= per_section:
            compacted.append(candidate)
            continue
        low, high = 0, len(candidate)
        while low < high:
            middle = (low + high + 1) // 2
            if _estimated_tokens(candidate[:middle]) <= per_section:
                low = middle
            else:
                high = middle - 1
        compacted.append(candidate[:low].rstrip() + "…")
    return compacted


def generate_mindmap(runtime: Runtime, source_filter: list[str] | None = None) -> str:
    if runtime.collection.count() == 0:
        return "📚 知识库为空，请先上传文档。"
    if runtime.client is None:
        return "⚙️ 请先在“设置”页配置模型服务。"
    where = _source_where_clause(_normalise_source_filter(source_filter))
    all_chunks = runtime.collection.get(
        include=["documents", "metadatas"], **({"where": where} if where else {})
    )
    documents = all_chunks.get("documents") or []
    metadatas = all_chunks.get("metadatas") or [{} for _document in documents]
    if not documents:
        return "📚 所选范围没有可用内容，请重新选择文档。"
    system = "你是一位严谨的学术助教。请根据提供的课程资料，生成精炼、完整、便于复习的学习大纲。"
    instruction = (
        "请用中文，基于以下按原文顺序提供的章节代表片段生成 Markdown 层级大纲（使用 # ## ### - 表示层级）。"
        "覆盖每个已提供的学术章节，保留章节从属关系；每个小节只写 1–2 条最重要的信息，"
        "全文控制在约 1200 个中文字以内。重点保留研究问题、核心方法、实验设计、关键结果、结论与局限。"
        "不要列出作者、单位、作者贡献、致谢、利益冲突、数据链接或参考文献，也不要编造片段中没有的信息。"
        "不要包含开场白或结尾总结，直接输出大纲结构。"
    )
    candidates, labels = _outline_section_candidates(documents, metadatas)
    candidates = _outline_candidates_within_budget(
        runtime, system, instruction, candidates, labels
    )
    try:
        selected, prompt = _pack_contexts(
            runtime,
            system,
            instruction,
            candidates,
            labels=labels,
        )
        answer = _complete_text(runtime, system, prompt)
        if len(selected) < len(candidates):
            answer += (
                f"\n\n⚠️ 输入窗口有限，本次大纲覆盖了 {len(selected)}/{len(candidates)} 个章节代表片段。"
            )
        return answer
    except Exception as exc:
        return f"❌ 生成大纲失败：{exc}"


def generate_quiz(runtime: Runtime, source_filter: list[str] | None = None) -> str:
    if runtime.collection.count() == 0:
        return "📚 知识库为空，请先上传文档。"
    if runtime.client is None:
        return "⚙️ 请先在“设置”页配置模型服务。"
    where = _source_where_clause(_normalise_source_filter(source_filter))
    all_chunks = runtime.collection.get(
        include=["documents"], **({"where": where} if where else {})
    )
    documents = all_chunks.get("documents") or []
    if not documents:
        return "📚 所选范围没有可用内容，请重新选择文档。"
    sample_chunks = random.sample(
        documents, min(runtime.config.context_k, len(documents))
    )
    system = "你是一个严谨的大学教师。请根据资料出5道单项选择题，用于考察学生对知识的掌握程度。"
    instruction = (
        "请根据以下资料生成5道单项选择题。只输出合法 JSON 对象，不要使用 Markdown 代码块。"
        "对象格式必须是 {\"questions\": [...]}，questions 中每项必须包含 question、options、"
        "answer、explanation；options 是4个选项文本组成的数组，answer 只能是 A、B、C、D。"
    )
    try:
        _selected, prompt = _pack_contexts(
            runtime,
            system,
            instruction,
            sample_chunks,
        )
        return _complete_text(
            runtime,
            system,
            prompt,
            temperature=0.4,
            json_output=True,
        )
    except Exception as exc:
        return f"出题失败：{exc}"


def parse_quiz_items(response: str) -> list[dict[str, Any]]:
    """Parse the small, fixed JSON shape requested from the model."""

    text = str(response or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("题目格式不正确")
    payload = payload.get("questions")
    if not isinstance(payload, list) or len(payload) != 5:
        raise ValueError("题目数量不是 5 道")
    items = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("题目格式不正确")
        question = str(item.get("question") or "").strip()
        options = item.get("options")
        answer = str(item.get("answer") or "").strip().upper()
        explanation = str(item.get("explanation") or "").strip()
        if (
            not question
            or not isinstance(options, list)
            or len(options) != 4
            or any(not str(option).strip() for option in options)
            or answer not in {"A", "B", "C", "D"}
            or not explanation
        ):
            raise ValueError("题目字段不完整")
        items.append(
            {
                "question": question,
                "options": [str(option).strip() for option in options],
                "answer": answer,
                "explanation": explanation,
            }
        )
    return items


def score_quiz(quiz_items: list[dict[str, Any]], answers: list[Any]) -> str:
    if not quiz_items:
        return "请先生成题目。"
    if len(answers) < len(quiz_items) or any(not answer for answer in answers):
        return "请完成全部 5 道题后再提交。"
    selected = [str(answer).strip()[:1].upper() for answer in answers]
    score = sum(
        answer == str(item["answer"]).upper()
        for item, answer in zip(quiz_items, selected)
    )
    details = [f"## 得分：{score} / {len(quiz_items)}"]
    for index, (item, answer) in enumerate(zip(quiz_items, selected), start=1):
        status = "正确" if answer == item["answer"] else "错误"
        details.append(
            f"### 第 {index} 题：{status}\n"
            f"你的答案：{answer}　正确答案：{item['answer']}\n\n"
            f"{item['explanation']}"
        )
    return "\n\n".join(details)


def build_demo(
    runtime: Runtime,
    *,
    managed_local_model: bool = False,
    on_exit: Callable[[], None] | None = None,
) -> Any:
    """Build the UI around an explicitly supplied runtime."""

    import gradio as gr

    def inventory_view() -> tuple[list[tuple[str, int]], str]:
        documents = document_inventory(runtime)
        inventory = (
            "**已上传文档：**\n" + "\n".join(
                f"- {source}（{count} 个文本块）" for source, count in documents
            )
            if documents
            else "**已上传文档：** 暂无"
        )
        return documents, inventory

    def library_state() -> tuple[Any, ...]:
        documents, inventory = inventory_view()
        sources = [source for source, _count in documents]
        return (
            f"**当前知识库文本块数：** {runtime.collection.count()}",
            inventory,
            gr.update(choices=sources, value=[]),
            gr.update(choices=sources, value=[]),
            gr.update(choices=sources, value=[]),
            gr.update(choices=sources, value=[]),
        )

    def page_header(title: str, description: str) -> None:
        gr.HTML(
            f"""
            <section class="kb-page-head">
                <h1>{title}</h1>
                <p>{description}</p>
            </section>
            """,
            apply_default_css=False,
        )

    def answer_with_evidence(
        message: str,
        history: Any,
        sources: list[str] | None,
    ) -> tuple[str, str]:
        result = query_knowledge(
            message,
            runtime=runtime,
            source_filter=sources,
        )
        return str(result["answer"]), format_evidence_panel(result)

    def prepare_quiz(sources: list[str] | None) -> tuple[Any, ...]:
        response = generate_quiz(runtime, source_filter=sources)
        try:
            items = parse_quiz_items(response)
        except (json.JSONDecodeError, ValueError):
            status = (
                response
                if response.startswith(("📚", "⚙️", "出题失败"))
                else "题目格式不完整，请重新生成。"
            )
            hidden = [
                gr.update(choices=[], value=None, visible=False) for _ in range(5)
            ]
            return (
                status,
                *hidden,
                [],
                gr.update(interactive=False),
                gr.update(value="", visible=False),
            )
        updates = []
        for index, item in enumerate(items, start=1):
            choices = [
                f"{letter}. {option}"
                for letter, option in zip("ABCD", item["options"])
            ]
            updates.append(
                gr.update(
                    choices=choices,
                    label=f"第 {index} 题：{item['question']}",
                    value=None,
                    visible=True,
                )
            )
        return (
            "已生成 5 道题。完成作答后提交，即可查看得分与解析。",
            *updates,
            items,
            gr.update(interactive=True),
            gr.update(value="", visible=False),
        )

    initial_documents, initial_inventory = inventory_view()
    initial_sources = [source for source, _count in initial_documents]
    with gr.Blocks(title=APP_DISPLAY_NAME) as demo:
        gr.HTML(
            f"""
            <header class="kb-header">
                <div class="kb-brand">
                    <span class="kb-mark" aria-hidden="true">文</span>
                    <div>
                        <h1 class="kb-title">{APP_DISPLAY_NAME}</h1>
                        <p class="kb-subtitle">本地文献阅读与学习工作台</p>
                    </div>
                </div>
                <div class="kb-crumb">工作台&nbsp;&nbsp;/&nbsp;&nbsp;<strong>个人知识库</strong></div>
            </header>
            """,
            elem_id="kb-header",
            apply_default_css=False,
        )
        with gr.Tabs(elem_id="kb-workspace"):
            with gr.Tab("资料库"):
                page_header(
                    "研究资料库",
                    "导入 PDF、TXT 或 DOCX，建立只保存在当前设备上的检索资料库。",
                )
                with gr.Row(equal_height=False, elem_classes="kb-library-grid"):
                    with gr.Column(
                        scale=3,
                        elem_classes=["kb-panel", "kb-library-panel"],
                    ):
                        gr.HTML(
                            """
                            <div class="kb-panel-title">
                                <h3>添加本地资料</h3>
                                <p>选择一个文档，确认后加入知识库。</p>
                            </div>
                            """,
                            apply_default_css=False,
                        )
                        file_input = gr.File(
                            label="选择文档",
                            file_types=[".pdf", ".txt", ".docx"],
                            height=170,
                        )
                        upload_button = gr.Button(
                            "添加到知识库",
                            variant="primary",
                            elem_id="kb-upload-button",
                        )
                        upload_output = gr.Textbox(
                            label="上传状态",
                            lines=3,
                            interactive=False,
                        )
                    with gr.Column(
                        scale=2,
                        elem_classes=["kb-panel", "kb-library-panel"],
                    ):
                        gr.HTML(
                            """
                                <div class="kb-panel-title">
                                    <h3>知识库概览</h3>
                                    <p>查看已解析资料，并在需要时批量移除文档。</p>
                            </div>
                            """,
                            apply_default_css=False,
                        )
                        count_output = gr.Markdown(
                            f"**当前知识库文本块数：** {runtime.collection.count()}"
                        )
                        inventory_output = gr.Markdown(initial_inventory)
                        with gr.Accordion("管理已上传文档", open=False):
                            delete_select = gr.Dropdown(
                                label="选择一篇或多篇要删除的文档",
                                choices=initial_sources,
                                value=[],
                                multiselect=True,
                                filterable=False,
                            )
                            delete_confirm = gr.Checkbox(
                                label="确认删除所选文档及其本地数据",
                                elem_id="kb-delete-confirm",
                            )
                            delete_button = gr.Button(
                                "删除所选文档",
                                variant="stop",
                            )
                            delete_output = gr.Textbox(
                                label="删除状态",
                                lines=2,
                                interactive=False,
                            )
            with gr.Tab("资料问答"):
                page_header(
                    "资料问答",
                    "限定资料范围后提问，并对照回答实际使用的原文片段与页码。",
                )
                source_select = gr.Dropdown(
                    label="限定回答范围",
                    choices=initial_sources,
                    value=[],
                    multiselect=True,
                    filterable=False,
                    elem_id="kb-answer-sources",
                    elem_classes="kb-source-select",
                    info="范围会应用到下一次提问；比较多篇时请在问题中写出各篇名称。",
                    render=False,
                )
                with gr.Row(equal_height=True, elem_classes="kb-chat-layout"):
                    with gr.Column(
                        scale=3,
                        elem_classes=["kb-panel", "kb-context-panel"],
                    ):
                        gr.HTML(
                            """
                            <div class="kb-panel-title">
                                <h3>原文依据</h3>
                                <p>本次回答实际使用的检索片段与页码。</p>
                            </div>
                            """,
                            apply_default_css=False,
                        )
                        gr.HTML(
                            """
                            <div class="kb-context-note">
                                这里展示检索原文，不是模型重新生成的摘要。
                            </div>
                            """,
                            apply_default_css=False,
                        )
                        evidence_output = gr.Markdown(
                            "提问后，这里会显示回答实际使用的原文片段。",
                            elem_classes="kb-evidence",
                        )
                    with gr.Column(
                        scale=7,
                        elem_classes=["kb-panel", "kb-chat-panel"],
                    ):
                        gr.HTML(
                            """
                            <div class="kb-panel-title">
                                <h3>向资料库提问</h3>
                                <p>回答仅基于当前知识库中的可检索内容。</p>
                            </div>
                            """,
                            apply_default_css=False,
                        )
                        source_select.render()
                        gr.ChatInterface(
                            fn=answer_with_evidence,
                            title=None,
                            description=None,
                            chatbot=gr.Chatbot(
                                height="clamp(300px, 42vh, 560px)",
                                label="对话",
                            ),
                            textbox=gr.Textbox(
                                label="问题",
                                show_label=False,
                                placeholder="例如：这篇论文的核心结论是什么？",
                                scale=7,
                                submit_btn=True,
                                stop_btn=True,
                            ),
                            additional_inputs=[source_select],
                            additional_outputs=[evidence_output],
                        )
            with gr.Tab("学习大纲"):
                page_header(
                    "学习大纲",
                    "按章节顺序整理所选资料的主要内容，参考文献列表不纳入大纲。",
                )
                outline_sources = gr.Dropdown(
                    label="大纲资料范围", choices=initial_sources, value=[], multiselect=True,
                    filterable=False,
                    elem_id="kb-outline-sources",
                    elem_classes="kb-source-select",
                    info="不选择时使用全部资料；更改范围后请重新生成。",
                )
                with gr.Row(elem_classes=["kb-panel", "kb-action-bar"]):
                    with gr.Column(scale=4):
                        gr.HTML(
                            """
                            <div class="kb-action-copy">
                                <h3>从所选资料生成大纲</h3>
                                <p>每个正文章节至少选取一个代表片段，生成 Markdown 层级结构。</p>
                            </div>
                            """,
                            apply_default_css=False,
                        )
                    with gr.Column(scale=1, min_width=150):
                        button = gr.Button(
                            "生成学习大纲",
                            variant="primary",
                            elem_id="kb-mindmap-button",
                        )
                output = gr.Markdown(
                    label="大纲内容",
                    value="生成结果会显示在这里。",
                    elem_classes="kb-output",
                )
                button.click(
                    lambda sources: generate_mindmap(runtime, source_filter=sources),
                    inputs=[outline_sources], outputs=output,
                )
            with gr.Tab("自测练习"):
                page_header(
                    "自测习题",
                    "从所选资料中抽取部分片段生成 5 道单项选择题，提交后显示得分与解析。",
                )
                quiz_sources = gr.Dropdown(
                    label="自测资料范围", choices=initial_sources, value=[], multiselect=True,
                    filterable=False,
                    elem_id="kb-quiz-sources",
                    elem_classes="kb-source-select",
                    info="不选择时使用全部资料；更改范围后请重新生成题目。",
                )
                with gr.Row(elem_classes=["kb-panel", "kb-action-bar"]):
                    with gr.Column(scale=4):
                        gr.HTML(
                            """
                            <div class="kb-action-copy">
                                <h3>生成一组资料自测题</h3>
                                <p>作答前不会显示答案，提交后逐题核对。</p>
                            </div>
                            """,
                            apply_default_css=False,
                        )
                    with gr.Column(scale=1, min_width=150):
                        quiz_generate_button = gr.Button(
                            "生成 5 道练习题",
                            variant="primary",
                            elem_id="kb-quiz-button",
                        )
                quiz_status = gr.Markdown("生成题目后开始作答。")
                quiz_state = gr.State([])
                with gr.Column(elem_classes=["kb-panel", "kb-quiz-stack"]):
                    quiz_answers = [
                        gr.Radio(
                            choices=[],
                            label=f"第 {index} 题",
                            visible=False,
                            elem_classes="kb-quiz-question",
                        )
                        for index in range(1, 6)
                    ]
                    quiz_submit_button = gr.Button(
                        "提交答案",
                        variant="primary",
                        interactive=False,
                    )
                quiz_result = gr.Markdown(
                    elem_classes=["kb-output", "kb-quiz-result"],
                    visible=False,
                )
                quiz_generate_button.click(
                    prepare_quiz,
                    inputs=[quiz_sources],
                    outputs=[
                        quiz_status,
                        *quiz_answers,
                        quiz_state,
                        quiz_submit_button,
                        quiz_result,
                    ],
                )
                quiz_submit_button.click(
                    lambda items, *answers: gr.update(
                        value=score_quiz(items, list(answers)),
                        visible=True,
                    ),
                    inputs=[quiz_state, *quiz_answers],
                    outputs=quiz_result,
                )
            if not managed_local_model:
                initial_service = next(
                    (
                        name
                        for name, (base_url, _model) in MODEL_SERVICE_PRESETS.items()
                        if base_url.rstrip("/") == runtime.config.llm_base_url.rstrip("/")
                    ),
                    "自定义 OpenAI 兼容服务",
                )
                with gr.Tab("模型设置"):
                    page_header(
                        "模型设置",
                        "优先使用本机 Ollama；也可以连接你自己的 OpenAI 兼容模型服务。",
                    )
                    with gr.Row(equal_height=True, elem_classes="kb-settings-grid"):
                        with gr.Column(scale=2, elem_classes="kb-panel"):
                            gr.HTML(
                                """
                                <div class="kb-panel-title">
                                    <h3>选择模型</h3>
                                    <p>切换预设后仍可手动调整模型名称。</p>
                                </div>
                                """,
                                apply_default_css=False,
                            )
                            service_select = gr.Dropdown(
                                label="模型服务",
                                choices=list(MODEL_SERVICE_PRESETS),
                                value=initial_service,
                            )
                            model_input = gr.Textbox(
                                label="模型名称",
                                value=runtime.config.llm_model,
                            )
                        with gr.Column(scale=3, elem_classes="kb-panel"):
                            gr.HTML(
                                """
                                <div class="kb-panel-title">
                                    <h3>连接信息</h3>
                                    <p>API Key 仅保留在当前运行进程中。</p>
                                </div>
                                """,
                                apply_default_css=False,
                            )
                            base_url_input = gr.Textbox(
                                label="Base URL",
                                value=runtime.config.llm_base_url,
                            )
                            api_key_input = gr.Textbox(
                                label="API Key（Ollama 本地服务可留空）",
                                type="password",
                                placeholder="云端服务请输入自己的 Key",
                                info="应用成功后会保留为隐藏圆点，仅在当前运行进程中使用。",
                            )
                    model_service_button = gr.Button(
                        "检测连接并应用",
                        variant="primary",
                        elem_id="kb-settings-button",
                    )
                    model_service_status = gr.Markdown(
                        f"✅ 已从环境变量配置模型 {html.escape(runtime.config.llm_model)}。"
                        if runtime.client is not None
                        else "尚未配置模型服务；本地文档管理仍可使用。"
                    )
        exit_button = (
            gr.Button(
                "退出工作台",
                variant="secondary",
                size="sm",
                elem_id="kb-exit-button",
            )
            if managed_local_model and on_exit is not None
            else None
        )

        def handle_upload(
            file: Any,
            progress: Any = gr.Progress(),
        ) -> tuple[Any, ...]:
            return (upload_file(file, runtime, progress), *library_state())

        def handle_delete(sources: list[str], confirmed: bool) -> tuple[Any, ...]:
            return (
                delete_document(sources, confirmed, runtime=runtime),
                *library_state(),
                False,
            )

        if not managed_local_model:
            service_select.change(
                model_service_defaults,
                inputs=service_select,
                outputs=[base_url_input, model_input],
            )
            model_service_button.click(
                lambda base_url, model, api_key: (
                    configure_model_service(base_url, model, api_key, runtime=runtime),
                    api_key,
                ),
                inputs=[base_url_input, model_input, api_key_input],
                outputs=[model_service_status, api_key_input],
            )

        upload_button.click(
            handle_upload,
            inputs=file_input,
            outputs=[
                upload_output,
                count_output,
                inventory_output,
                delete_select,
                source_select,
                outline_sources,
                quiz_sources,
            ],
            show_progress_on=upload_output,
        )
        delete_button.click(
            handle_delete,
            inputs=[delete_select, delete_confirm],
            outputs=[
                delete_output,
                count_output,
                inventory_output,
                delete_select,
                source_select,
                outline_sources,
                quiz_sources,
                delete_confirm,
            ],
        )
        if exit_button is not None:
            exit_button.click(on_exit, inputs=[], outputs=[])
    return demo

if __name__ == "__main__":
    import gradio as gr

    build_demo(create_runtime()).launch(
        theme=app_theme(gr),
        css=APP_CSS,
        inbrowser=True,
    )
