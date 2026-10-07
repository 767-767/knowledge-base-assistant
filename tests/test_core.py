import importlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import app
import sci_rag_core as core
from sci_rag_retrieval import DocumentRoute, RankedItem
from sci_rag_core import (
    Chunk,
    build_evidence_ledger,
    extract_spatial_figure_chunks,
    formula_evidence_indices,
    extract_table_cell,
    extract_table_row_values,
    extract_tables,
    find_table_cell_in_chunks,
    filter_table_rows_by_entity,
    figure_reference_from_question,
    is_comparative_table_question,
    is_derived_value_question,
    is_formula_question,
    is_limitation_question,
    limitation_evidence_indices,
    is_table_question,
    matching_table_indices,
    table_labels_from_question,
    normalize_for_match,
    parse_markdown_table,
    missing_pdf_formula_blocks,
    repair_latex_json_escapes,
    rerank_table_first,
    select_row_entity,
    split_to_chunks,
    supplement_answer_with_evidence,
    supplement_formula_with_evidence,
    validate_answer_against_evidence,
)


TABLE_1 = """|Baseline|Score|
|---|---|
|**DrugR**|**0.2712**|
|GPT5|0.1969|"""
TABLE_2 = """|Baseline|F1|
|---|---|
|**DrugR**<sup>_∗_</sup>|**0.3404**|
|_SFT_<sup>_∗_</sup>|0.2997|"""
TABLE_2_FULL = """|Baseline|Overall Optimization Score|Target property F1 score|
|---|---|---|
|**DrugR**<sup>_∗_</sup>|**0.2060**|**0.3404**|
|_SFT_<sup>_∗_</sup>|0.1949|0.2997|"""
TABLE_6 = """|**Metric**|**Mean / Value**|**Range / Definition|
|---|---|---|
|Heavy atoms|24.12|2–43|
|Unique SMILES|3,863 / 4,826|80.05%|
|Unique scaffolds|1,117 / 4,826|23.15%|"""
TABLE_5 = """|**Drug category**|**Target set**|**Representative drugs (examples)**|
|---|---|---|
|Anti-inflammatory (NSAIDs)|COX1,COX2|aspirin,ibuprofen|
|Antihypertensive<br>(ACEi/ARB/_β_-blockers)|ACE, AGTR1, ADRB1,<br>ADRB2|captopril,losartan|"""
TABLE_HIERARCHICAL = """|Test Dataset|Setting|Model|Parameters|METEOR|ROUGE-1|BertS|
|---|---|---|---|---|---|---|
|||FlanT5-xl|3B|0.08|0.10|0.78|
|||FlanT5-xl|3B|0.08|0.09|0.78|"""
TABLE_MULTI_VARIANT = """|Setting|Data|MSE|F1-score|
|---|---|---|---|
|WikiTQ+SQA+SciGen|Title + Abstract|2.61<br>**2.30**|0.28<br>**0.38**|"""
TABLE_GROUPED = """|Setting|Model|METEOR|ROUGE-1|BertS|
|---|---|---|---|---|
||**Test(C&L**|**)**|||
|WikiTQ+ SQA + SciGen|FlanT5-xl|0.15|0.24|0.85|
||**Test(Othe**|**r)**|||
|WikiTQ + SQA + SciGen|FlanT5-xl|0.14|0.23|0.85|"""
TABLE_SETTING_QUALIFIER = """|Setting|Model|METEOR|ROUGE-1|BertS|
|---|---|---|---|---|
||**Test(C&L**|**)**|||
||FlanT5-xl|0.14(+0.06)|0.23(+0.13)|0.85(+0.07)|
|WikiTQ|FlanT5-xl|0.08|0.12(+0.02)|0.81(+0.03)|
|WikiTQ + SQA|FlanT5-xl|0.08|0.10|0.79(+0.01)|
|WikiTQ + SQA + SciGen|FlanT5-xl|0.15(+0.07)|0.24(+0.14)|0.85(+0.07)|
||**Test(Othe**|**r)**|||
||FlanT5-xl|0.13(+0.05)|0.23(+0.14)|0.85(+0.07)|"""
TABLE_SPLIT_MULTI_HEADER = """|**Mthd**|**H**|**as-Ans**|**wer**|**M**|**iss-Ans**|**wer**|**Inter**|**nal Kno**|**wledge**|
|---|---|---|---|---|---|---|---|---|---|
|**eo**|PopQA|NQ|TriviaQA|PopQA|NQ|TriviaQA|PopQA|NQ|TriviaQA|
|**Llama-3-Ins-8B**||||||||||
|THINKNOTE|90.1|80.4|97.0|8.3|25.8|36.0|99.2|92.9|97.3|
|**Llama-3-Ins-70**|**B**|||||||||
|THINKNOTE|95.9|88.3|97.5|12.2|33.3|54.0|93.6|94.1|98.0|"""
TABLE_QUALIFIED_METRICS = """|Model|Darcy smooth L2|Darcy smooth H1|Darcy rough L2|Darcy rough H1|
|---|---|---|---|---|
|MgNO|0.176|0.576|0.339|1.380|"""
TABLE_MULTI_ROW = """|Model Configuration|L2 Error (×10−2)|
|---|---|
|MgNO, 4 levels|2.10|
|MgNO, 6 layers|1.47|
|Baseline MgNO|1.63|"""
TABLE_CONFIG_VARIANTS = """|Model|RAG|FT|
|---|---|---|
|GPT-4o|46.63|54.03|"""
TABLE_MULTI_DATASET = """|Model|Dataset|BLEU|ROUGE-1|
|---|---|---|---|
|Llama-2|MedICaT|0.10|0.20|
|FigEx-7B||0.11|0.21|
|Llama-2|BioSci-Fig|0.30|0.40|
|FigEx-7B||0.72|0.81|"""
TABLE_MULTI_SETTING = """|Model|all F1|
|---|---|
||||Oracle|setting||||
|Gemini Flash|60.7|
|Human|73.0|
||||Closed book|setting||||
|PaLM-2|47.6|"""
TABLE_SECTION_GROUPS = """|Question Type|Count|Freq (%)|
|---|---|---|
|Skill|||
|1. No skill|160|14.9|
|2. Filtering numeric|66|6.1|
|Skills per question|||
|1. No skill|300|27.9|
|2. One skill|657|61.2|
|3. Two skills|94|8.8|
|4. Three skills|23|2.1|"""
TABLE_RANK_GROUPED = """||Model||FinH|ybrid|||Pap|erTab|||Pape|rText|||Feta|Tab|||Nq|Text||
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|||@1|@5|@10|@20|@1|@5|@10|@20|@1|@5|@10|@20|@1|@5|@10|@20|@1|@5|@10|@20|
|Sparse|BM-25|65.6|83.7|87.4|90.0|46.0|79.7|90.0|92.3|47.4|80.0|88.0|89.9|68.3|91.9|95.2|96.2|42.0|69.2|75.8|80.3|"""
TABLE_LIVEXIV_AVG = """||LiveXiv|Verified Subset|Absolute Avg.|
|---|---|---|---|
|**VQA**|46.734|47.273|2.336|
|**TQA**|45.101|46.028|2.105|"""


class CoreTests(unittest.TestCase):
    def test_stopping_limits_do_not_request_shortcoming_evidence(self):
        from sci_rag_core import is_limitation_question

        self.assertFalse(is_limitation_question("seed retrieval 和 graph expansion 的停止限制是什么？"))
        self.assertFalse(is_limitation_question("检索结果的字数限制是多少？"))
        for question in (
            "TANQ 中答案单元格条目数的筛选与问题改写都出现了数字五：它们分别限制什么对象、在什么情况下丢弃样本或停止改写？最后添加推理技能时，最多组合多少种不同技能？",
            "这个阈值限制了哪些样本？",
        ):
            with self.subTest(question=question):
                self.assertFalse(is_limitation_question(question))
        self.assertTrue(is_limitation_question("字数限制带来了哪些局限和失败模式？"))
        self.assertTrue(is_limitation_question("模型的限制有哪些？"))
        self.assertTrue(is_limitation_question("这个阈值限制了哪些样本，这带来了什么局限？"))

    def test_routed_numeric_evidence_normalizes_thousands_separator(self):
        class Collection:
            def count(self):
                return 2

            def get(self, **_kwargs):
                return {
                    "ids": ["noise", "answer"],
                    "documents": [
                        "A nearby result mentions 700.",
                        "Annotators reviewed 7000 candidate instances.",
                    ],
                    "metadatas": [
                        {"source": "paper.pdf", "type": "text"},
                        {"source": "paper.pdf", "type": "text"},
                    ],
                }

        runtime = app.Runtime(
            app.RuntimeConfig(document_routing=True),
            object(),
            object(),
            Collection(),
        )
        result = app._numeric_route_evidence_result(
            "SciDQA 审阅了 7,000 个候选实例",
            runtime,
            DocumentRoute("paper.pdf", ("scidqa",)),
        )
        self.assertEqual(result["ids"], [["answer"]])
        self.assertTrue(result["metadatas"][0][0]["route_evidence"])
        self.assertIn("3", app._explicit_number_tokens("top-3 chunks"))

    def test_generation_prompt_prevents_contradictory_refusal(self):
        self.assertIn("不得在已经给出具体表格数值后", app.SCIENTIFIC_SYSTEM_PROMPT)
        self.assertIn("明确要求计算", app.SCIENTIFIC_SYSTEM_PROMPT)
        self.assertIn("逐一检查全部原文引用", app.SCIENTIFIC_SYSTEM_PROMPT)
        self.assertIn("限制问题要区分限制本身与示例现象", app.SCIENTIFIC_SYSTEM_PROMPT)
        self.assertIn("不能用“训练数据限制”等原文未出现的机制替代", app.SCIENTIFIC_SYSTEM_PROMPT)

    def test_generation_prompt_labels_table_number_from_metadata(self):
        class Vector(list):
            def tolist(self):
                return list(self)

        class Embedding:
            def encode(self, _message):
                return Vector([0.1, 0.2])

        class Collection:
            def count(self):
                return 1

            def query(self, **_kwargs):
                return {
                    "ids": [["table"]],
                    "documents": [["|Information Source|% in Dataset|\n|---|---|\n|Multiple documents|10.9%|"]],
                    "metadatas": [[{"type": "table", "table_number": 2}]],
                }

            def get(self, **_kwargs):
                return {"ids": [], "documents": [], "metadatas": []}

        class Client:
            def __init__(self):
                self.prompt = ""
                self.chat = self
                self.completions = self

            def create(self, **kwargs):
                self.prompt = kwargs["messages"][1]["content"]
                return type(
                    "Response",
                    (),
                    {"choices": [type("Choice", (), {"message": type("Message", (), {"content": "10.9%"})()})()]},
                )()

        client = Client()
        app.query_knowledge(
            "Table 2 中需要多文档信息才能回答的问题占比是多少？",
            runtime=app.Runtime(app.RuntimeConfig(retrieval_k=1, context_k=1), client, Embedding(), Collection()),
        )
        self.assertIn("[表格，Table 2]", client.prompt)

    def test_generation_prompt_includes_table_caption_for_narrative_questions(self):
        class Vector(list):
            def tolist(self):
                return list(self)

        class Embedding:
            def encode(self, _message):
                return Vector([0.1, 0.2])

        class Collection:
            def count(self):
                return 1

            def query(self, **_kwargs):
                return {
                    "ids": [["table"]],
                    "documents": [["|Model|Blind|Oracle|\n|---|---|---|\n|GPT-4o|33.46|71.42|"]],
                    "metadatas": [[
                        {
                            "type": "table",
                            "table_number": 1,
                            "table_caption": "Table 1: Random baseline is 25%.",
                        }
                    ]],
                }

            def get(self, **_kwargs):
                return {"ids": [], "documents": [], "metadatas": []}

        class Client:
            def __init__(self):
                self.prompt = ""
                self.chat = self
                self.completions = self

            def create(self, **kwargs):
                self.prompt = kwargs["messages"][1]["content"]
                return type(
                    "Response",
                    (),
                    {"choices": [type("Choice", (), {"message": type("Message", (), {"content": "25%"})()})()]},
                )()

        client = Client()
        result = app.query_knowledge(
            "Table 1 的随机选择基线准确率是多少？",
            runtime=app.Runtime(
                app.RuntimeConfig(retrieval_k=1, context_k=1),
                client,
                Embedding(),
                Collection(),
            ),
        )
        self.assertIn("Random baseline is 25%", result["contexts"][0])
        self.assertIn("Random baseline is 25%", client.prompt)

    def test_selected_paper_title_identifies_its_own_dataset_in_prompt(self):
        from unittest.mock import Mock

        collection = Mock()
        collection.count.return_value = 1
        collection.query.return_value = {
            "ids": [["evidence"]],
            "documents": [["Only 25% of answers in our dataset exceed 80% overlap."]],
            "metadatas": [[{"source": "paper.pdf", "type": "text", "page": 4}]],
        }
        collection.get.return_value = {
            "metadatas": [{"headers": "H1: **SCIDQA: Scientific QA Dataset**"}]
        }
        embedding = Mock()
        embedding.encode.return_value.tolist.return_value = [1.0, 0.0]
        client = Mock()
        client.chat.completions.create.return_value.choices = [
            Mock(message=Mock(content="25%"), finish_reason="stop")
        ]
        runtime = app.Runtime(app.RuntimeConfig(), client, embedding, collection)

        app.query_knowledge("Summarize this study.", runtime=runtime, source_filter=["paper.pdf"])

        prompt = client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
        self.assertIn("文档：SCIDQA: Scientific QA Dataset", prompt)
        self.assertIn("Only 25% of answers in our dataset", prompt)

    def test_table_caption_unit_note_preserves_shared_scale(self):
        note = app._table_caption_unit_note(
            {
                "table_caption": (
                    "Table 1: Darcy errors (×10<sup>−2</sup>) and runtime (s/iter)."
                )
            }
        )
        self.assertIn("表注：Darcy errors", note)
        self.assertIn("×10", note)
        self.assertEqual(
            app._table_caption_unit_note({"table_caption": "Table 2: Results"}),
            "",
        )

    def test_evidence_ledger_keeps_complementary_numbers_and_entities(self):
        question = "显式推理数据集包含多少个样本？标注管道是什么？"
        texts = [
            "DeepSeek-R1 proposes structurally comparable candidates.\n"
            "ADMET properties are predicted with ADMETLab.\n"
            "Candidates are retained with fingerprint similarity greater than 0.6.",
            "The reverse-engineering pipeline generates rationales from two SMILES strings.",
            "The dataset contains 4,855 samples.",
        ]
        ledger = build_evidence_ledger(
            question,
            texts,
            [{"source": "paper.pdf"}] * len(texts),
        )
        joined = "\n".join(ledger)
        for fact in ("DeepSeek-R1", "ADMETLab", "0.6", "SMILES", "4,855"):
            self.assertIn(fact, joined)
        self.assertTrue(all(line.startswith("【原文引用：paper.pdf") for line in ledger))

    def test_formula_evidence_candidates_are_opt_in_and_ranked_by_terms(self):
        question = "线性有限元离散后的椭圆 PDE 系统写成什么形式，卷积核尺寸是多少？"
        texts = [
            "The paper introduces a model with 3 layers.",
            "With linear FEM discretization, the elliptic PDE system is A*u=f. "
            "The kernel A has dimensions 3 × 3.",
            "The convolution operator uses a kernel and residual update.",
        ]
        metas = [{"type": "text"}] * len(texts)
        self.assertEqual(formula_evidence_indices("样本数量是多少？", texts, metas), [])
        selected = formula_evidence_indices(question, texts, metas)
        self.assertEqual(selected[0], 1)
        self.assertEqual(
            formula_evidence_indices(question, texts, metas, allowed_indices=[0, 2]),
            [2],
        )

    def test_formula_evidence_aliases_conditional_perplexity_and_uncertainty(self):
        question = "TC–RAG 如何定义 conditional perplexity 和 uncertainty？"
        texts = [
            "cppl(Mtop | Mbottom) = exp ...",
            "uct(Mtop) = -P(wi | w1, ..., wi−1) log P(wi | w1, ..., wi−1)",
            "An unrelated transition formula δ(s, a) = (s', op, b).",
        ]
        metas = [{"type": "formula"}] * len(texts)
        selected = formula_evidence_indices(question, texts, metas, max_results=2)
        self.assertEqual(selected, [0, 1])

    def test_formula_evidence_uses_explicit_symbol_for_isolated_formula_chunk(self):
        question = "Thought Accommodation 阶段的最终输出 y 如何定义？"
        texts = [
            "The final output is produced as: y = M (Ita, q, T, R).",
            "A separate equation describes an unrelated score.",
        ]
        metas = [{"type": "formula"}, {"type": "formula"}]

        self.assertEqual(formula_evidence_indices(question, texts, metas), [0])

    def test_formula_evidence_prefers_requested_variable_on_left_hand_side(self):
        texts = [
            "y = M (Ita, q, T, R)",
            "T = M (Ika, q, D)",
        ]
        metas = [{"type": "formula"}, {"type": "formula"}]

        self.assertEqual(
            formula_evidence_indices(
                "Knowledge Assimilation 阶段的形式化输出 T 如何定义？",
                texts,
                metas,
            )[0],
            1,
        )

    def test_formula_output_restores_json_control_character_escapes(self):
        corrupted = "若 j " + "\n" + "otin V_i，则屏蔽。"
        self.assertEqual(
            repair_latex_json_escapes(corrupted),
            "若 j \\notin V_i，则屏蔽。",
        )

    def test_formula_escape_repair_does_not_rewrite_ordinary_words(self):
        answer = "说明：\nequation (1) defines the score."
        self.assertEqual(repair_latex_json_escapes(answer), answer)

    def test_formula_supplement_leaves_non_formula_newlines_untouched(self):
        answer = "第一段。\nequation (1) defines the score."
        self.assertEqual(
            supplement_formula_with_evidence(
                "这篇论文的主要贡献是什么？",
                answer,
                [],
            ),
            answer,
        )

    def test_formula_supplement_quotes_missing_explicit_operator_line(self):
        answer = supplement_formula_with_evidence(
            "NaviRAG 如何决定展开还是吸收当前节点的摘要？",
            "中间节点继续展开，叶节点吸收当前摘要。",
            ["π(n) ∈ {absorb, expand} based on similarity."],
            [{"type": "formula", "formula_evidence": True}],
        )
        self.assertIn("【公式原文核对项】", answer)
        self.assertIn("π(n) ∈ {absorb, expand}", answer)

    def test_formula_supplement_skips_unrelated_formula_prose(self):
        answer = supplement_formula_with_evidence(
            "NaviRAG 最终上下文集合 C 由哪三部分组成？",
            "C = Cvec ∪ Csum ∪ Craw。",
            [
                "and NaviRAG, we set top-k = 5",
                "C = Cvec ∪ Csum ∪ Craw",
            ],
            [{"type": "formula", "formula_evidence": True}] * 2,
        )
        self.assertNotIn("top-k", answer)
        self.assertNotIn("【公式原文核对项】", answer)

    def test_formula_supplement_does_not_append_other_named_symbols(self):
        answer = supplement_formula_with_evidence(
            "离散系统 A∗u=f 中卷积核 A 的尺寸是多少？",
            "离散系统为 A∗u=f，A 的尺寸是 3×3。",
            [
                "K_i,j(x, x′) defines another operator",
                "i,j ∈ L(Y, Y)",
            ],
            [{"type": "formula", "formula_evidence": True}] * 2,
        )
        self.assertNotIn("【公式原文核对项】", answer)

    def test_formula_supplement_keeps_explicit_multi_character_formula_labels(self):
        question = "Definition 4 如何定义 Retrieval Graph G2？请给出 G2 和 Type Filtering 的公式。"
        answer = supplement_formula_with_evidence(
            question,
            "参考片段给出了类型过滤公式。",
            [
                "G2 = Ψ1",
                "Type Filtering : V = {v ∈D | ϕtype(v) ∈T}",
            ],
            [{"type": "formula", "formula_evidence": True}] * 2,
        )
        self.assertIn("G2 = Ψ1", answer)
        self.assertIn("V = {v ∈D | ϕtype(v) ∈T}", answer)

    def test_pdf_formula_text_recovery_merges_adjacent_spans_and_parentheses(self):
        class Page:
            def get_text(self, kind, sort=True):
                self.kind = kind
                return {
                    "blocks": [
                        {
                            "type": 0,
                            "lines": [
                                {
                                    "bbox": (10, 10, 20, 20),
                                    "spans": [{"text": "u = B *"}],
                                },
                                {
                                    "bbox": (21, 11, 50, 20),
                                    "spans": [{"text": chr(16) + "f - A * u" + chr(17)}],
                                },
                            ],
                        }
                    ]
                }

        recovered = app._pdf_text_layer_for_formula_recovery(Page())

        self.assertEqual(recovered, "u = B *(f - A * u)")

    def test_limitation_evidence_candidates_preserve_mechanism_and_example(self):
        question = "AlphaFold 3 对分子动力学状态的建模有什么限制，cereblon 示例展示了什么？"
        texts = [
            "The method improves average accuracy on the benchmark.",
            "A key limitation is that models predict static structures as seen in the PDB, "
            "not the dynamical behaviour of biomolecular systems in solution.",
            "Conformation coverage is limited. Ground-truth cereblon is open in apo and "
            "closed in holo conformations; predictions of both are closed.",
        ]
        metas = [{"type": "text"}] * len(texts)
        self.assertTrue(is_limitation_question(question))
        selected = limitation_evidence_indices(question, texts, metas)
        self.assertIn(1, selected)
        self.assertIn(2, selected)
        self.assertEqual(
            limitation_evidence_indices(question, texts, metas, allowed_indices=[0]),
            [],
        )
        self.assertFalse(is_limitation_question("AlphaFold 3 的准确率是多少？"))
        self.assertFalse(
            is_limitation_question("MgNO 的限制和延拓操作如何改变网格，论文区分哪两种循环？")
        )

    def test_limitation_evidence_handles_singular_failure_mode(self):
        question = "论文指出 AF3 的两类主要立体化学失败模式是什么？"
        texts = [
            "The second class is a failure mode; chirality violations remain at 4.4%.",
            "A generic training detail without a failure marker.",
        ]
        metas = [{"type": "text"}, {"type": "text"}]
        self.assertEqual(limitation_evidence_indices(question, texts, metas), [0])

    def test_limitation_evidence_is_promoted_into_application_context(self):
        class Vector(list):
            def tolist(self):
                return list(self)

        class Embedding:
            def encode(self, _message):
                return Vector([0.1, 0.2])

        records = [
            ("generic", "The method improves benchmark accuracy.", {"source": "paper.pdf", "type": "text"}),
            (
                "mechanism",
                "A key limitation is predicting static structures as seen in the PDB, not dynamical behaviour in solution.",
                {"source": "paper.pdf", "type": "text"},
            ),
            (
                "example",
                "Cereblon is open in apo and closed in holo conformations; both predictions are closed.",
                {"source": "paper.pdf", "type": "text"},
            ),
        ]

        class Collection:
            def count(self):
                return len(records)

            def query(self, **_kwargs):
                return {
                    "ids": [[records[0][0]]],
                    "documents": [[records[0][1]]],
                    "metadatas": [[records[0][2]]],
                }

            def get(self, **_kwargs):
                return {
                    "ids": [row[0] for row in records],
                    "documents": [row[1] for row in records],
                    "metadatas": [row[2] for row in records],
                }

        class Client:
            def __init__(self):
                self.chat = self
                self.completions = self
                self.prompt = ""

            def create(self, **kwargs):
                self.prompt = kwargs["messages"][1]["content"]
                return type(
                    "Response",
                    (),
                    {"choices": [type("Choice", (), {"message": type("Message", (), {"content": "ok"})()})()]},
                )()

        client = Client()
        runtime = app.Runtime(
            app.RuntimeConfig(retrieval_k=1, context_k=3),
            client,
            Embedding(),
            Collection(),
        )
        result = app.query_knowledge(
            "AlphaFold 3 对分子动力学状态的建模有什么限制，cereblon 示例展示了什么？",
            runtime=runtime,
        )
        self.assertIn("static structures", result["contexts"][0])
        self.assertTrue(result["context_metadatas"][0]["limitation_evidence"])
        self.assertIn("[限制证据]", client.prompt)

    def test_formula_question_gate_distinguishes_pde_definition_from_architecture_questions(self):
        self.assertTrue(
            is_formula_question(
                "线性有限元离散后的椭圆 PDE 系统写成什么形式，卷积核尺寸是多少？"
            )
        )
        self.assertTrue(
            is_formula_question(
                "MgNO 讨论的二维椭圆 PDE 定义在哪个区域，并考虑哪些边界条件？"
            )
        )
        self.assertTrue(
            is_formula_question(
                "MgNO 的非线性激活函数是什么，W_Mg 多通道线性算子本身是否包含非线性激活？"
            )
        )
        self.assertTrue(
            is_formula_question(
                "MgNO 多重网格平滑迭代开始时如何初始化状态，更新时使用什么量？"
            )
        )
        self.assertFalse(
            is_formula_question(
                "LiveXiv 更新基准时，如何避免每一版都重测所有旧模型？"
            )
        )
        self.assertTrue(
            is_formula_question(
                "TC–RAG 如何定义 conditional perplexity 和 uncertainty？"
            )
        )
        self.assertTrue(
            is_formula_question("Knowledge Assimilation 阶段的形式化输出 T 如何定义？")
        )
        self.assertTrue(
            is_formula_question("Thought Accommodation 阶段的最终输出 y 如何定义？")
        )
        self.assertTrue(
            is_formula_question(
                "MgNO 的限制和延拓操作如何改变网格，论文区分哪两种循环？"
            )
        )
        self.assertTrue(is_formula_question("NaviRAG 最终上下文集合 C 由哪三部分组成？"))
        self.assertTrue(
            is_formula_question("NaviRAG 如何决定展开还是吸收当前节点的摘要？")
        )

    def test_answer_supplement_quotes_missing_high_signal_evidence(self):
        question = "显式推理数据集包含多少个样本？标注管道如何构建？"
        ledger = [
            "【片段 1，paper.pdf，Explicit Reasoning Dataset】"
            "DeepSeek-R1 proposes candidates; ADMETLab evaluates them with similarity greater than 0.6.",
            "【片段 2，paper.pdf，Dataset Statistics】The dataset contains 4,855 samples.",
            "【片段 3，paper.pdf，Explicit Reasoning Dataset】"
            "Starting molecules cover COX-1/COX-2, ACE and other therapeutic targets.",
            "【片段 4，paper.pdf，Explicit Reasoning Dataset】**Table 5** Category-specific target sets use ACE and AGTR1.",
        ]
        answer = supplement_answer_with_evidence(
            "数据集包含 4,855 个样本。",
            question,
            ledger,
        )
        self.assertIn("【补充原文核对项】", answer)
        self.assertIn("ADMETLab", answer)
        self.assertIn("0.6", answer)
        self.assertNotIn("Dataset Statistics", answer)
        self.assertNotIn("COX-1/COX-2", answer)
        self.assertNotIn("Table 5", answer)
        self.assertEqual(
            supplement_answer_with_evidence(
                "Pareto 重加权可以缓解失衡。",
                "强化学习阶段如何解决目标主导与目标饥饿？",
                ledger,
            ),
            "Pareto 重加权可以缓解失衡。",
        )

    def test_answer_supplement_skips_training_corpus_statistics(self):
        question = "显式推理数据集包含多少个样本？标注管道如何构建？"
        ledger = [
            "【片段 1，paper.pdf，Explicit Reasoning Dataset】"
            "The training mixture integrates four data sources. ChemicalQA (∼150K), "
            "MoleculeNet (∼160K), UltraChat-200K (∼200K), and CPT text corpus (∼300K).",
            "【片段 2，paper.pdf，Explicit Reasoning Dataset】"
            "The dataset contains 4,855 samples and uses DeepSeek-R1 for annotation.",
        ]
        answer = supplement_answer_with_evidence(
            "数据集包含 4,855 个样本，并使用 DeepSeek-R1。",
            question,
            ledger,
        )
        self.assertNotIn("training mixture", answer)
        self.assertNotIn("ChemicalQA", answer)

    def test_numeric_answer_supplement_skips_unrelated_neighboring_lines(self):
        question = "人工标注中审阅了多少个实例，保留多少个问答对，一致率是多少？"
        ledger = [
            "【片段 1，paper.pdf，Human Expert Annotation】"
            "Two annotators reviewed 7,000 instances and identified 2,937 QA pairs; "
            "the common subset agreement rate was 85%.",
            "【片段 2，paper.pdf，Related Work】"
            "QASPER has 40% short answers, while QASA has 52% high-overlap answers.",
        ]
        answer = supplement_answer_with_evidence(
            "共审阅 7,000 个实例，保留 2,937 个问答对，一致率为 85%。",
            question,
            ledger,
        )
        self.assertNotIn("【补充原文核对项】", answer)
        self.assertNotIn("QASPER", answer)

    def test_evidence_validator_flags_missing_number_without_using_gold(self):
        ledger = [
            "【片段 1，paper.pdf，Dataset Statistics】The dataset contains 4,855 samples.",
            "【片段 2，paper.pdf，Introduction】The paper studies molecular optimization.",
        ]
        result = validate_answer_against_evidence(
            "显式推理数据集包含多少个样本？",
            "资料未提供相关信息。",
            ledger,
        )
        self.assertEqual(result["status"], "review")
        self.assertIn("missing_relevant_number", result["reasons"])
        self.assertIn("4,855", [item["text"] for item in result["missing_markers"]])

    def test_evidence_validator_skips_unasked_table_narrative(self):
        result = validate_answer_against_evidence(
            "显式推理数据集包含多少个样本？",
            "数据集包含 4,855 个样本。",
            [
                "【片段 1，paper.pdf】As shown in Table 1, the annotated dataset has score 0.1653.",
                "【片段 2，paper.pdf，Dataset】The dataset contains 4,855 samples.",
            ],
        )
        self.assertEqual(result["status"], "ok")
        self.assertNotIn("0.1653", [item["text"] for item in result["missing_markers"]])

    def test_evidence_validator_flags_partial_composite_line(self):
        ledger = [
            "【片段 1，paper.pdf，Explicit Reasoning Dataset】"
            "The annotation pipeline uses DeepSeek-R1 for rationales and ADMETLab evaluates candidates above 0.6.",
        ]
        result = validate_answer_against_evidence(
            "数据集的标注管道如何构建？",
            "使用 DeepSeek R1 生成推理标注。",
            ledger,
        )
        self.assertEqual(result["status"], "review")
        self.assertIn("partial_high_signal_line", result["reasons"])
        self.assertTrue(any(item["text"] == "ADMETLab" for item in result["missing_markers"]))

    def test_evidence_validator_ignores_unmatched_context_and_empty_ledger(self):
        ok = validate_answer_against_evidence(
            "采用什么算法？",
            "采用 GRPO。",
            ["【片段 1，paper.pdf，Training】The reinforcement algorithm is GRPO."],
        )
        self.assertEqual(ok["status"], "ok")
        insufficient = validate_answer_against_evidence("问题是什么？", "答案。", [])
        self.assertEqual(insufficient["status"], "insufficient_evidence")

    def test_evidence_validator_defers_structured_semantics(self):
        table = validate_answer_against_evidence(
            "Table 2 中模型的得分是多少？",
            "根据表格回答。",
            ["【片段 1】|Model|Score|\n|---|---|\n|DrugR|0.2|"],
        )
        self.assertEqual(table["status"], "not_applicable")
        self.assertIn("structured_table_path", table["reasons"])
        spatial = validate_answer_against_evidence(
            "图 1 中的样本数是多少？",
            "n=25。",
            ["【片段 1】[x=1-2%; y=1-2%] n = 25"],
        )
        self.assertEqual(spatial["status"], "not_applicable")
        formula = validate_answer_against_evidence(
            "线性 PDE 方程写成什么形式？",
            "A*u=f。",
            ["【片段 1】The equation is A*u=f."],
        )
        self.assertEqual(formula["status"], "not_applicable")

    def test_composite_question_prioritizes_matching_section_siblings(self):
        class Vector(list):
            def tolist(self):
                return list(self)

        class Embedding:
            def encode(self, _message):
                return Vector([0.1, 0.2])

        section = "H1: Paper > H3: Explicit Reasoning Dataset"
        other = "H1: Paper > H3: Introduction"

        class Collection:
            def count(self):
                return 3

            def query(self, **_kwargs):
                return {
                    "ids": [["overview", "other"]],
                    "documents": [["Dataset overview.", "Unrelated introduction."]],
                    "metadatas": [[
                        {"source": "paper.pdf", "headers": section, "type": "text"},
                        {"source": "paper.pdf", "headers": other, "type": "text"},
                    ]],
                }

            def get(self, **_kwargs):
                return {
                    "ids": ["overview", "sibling", "other"],
                    "documents": [
                        "Dataset overview.",
                        "ADMETLab is used and similarity must be greater than 0.6.",
                        "Unrelated introduction.",
                    ],
                    "metadatas": [
                        {"source": "paper.pdf", "headers": section, "type": "text"},
                        {"source": "paper.pdf", "headers": section, "type": "text"},
                        {"source": "paper.pdf", "headers": other, "type": "text"},
                    ],
                }

        class Client:
            def __init__(self):
                self.prompt = ""
                self.chat = self
                self.completions = self

            def create(self, **kwargs):
                self.prompt = kwargs["messages"][1]["content"]
                return type(
                    "Response",
                    (),
                    {"choices": [type("Choice", (), {"message": type("Message", (), {"content": "ok"})()})()]},
                )()

        client = Client()
        runtime = app.Runtime(
            app.RuntimeConfig(retrieval_k=2, context_k=2),
            client,
            Embedding(),
            Collection(),
        )
        result = app.query_knowledge(
            "显式推理数据集包含多少个样本？标注管道如何构建？",
            runtime=runtime,
        )
        self.assertIn("ADMETLab", result["contexts"][1])
        self.assertNotIn("Unrelated introduction", result["contexts"])
        self.assertIn("ADMETLab", client.prompt)

    def test_composite_section_expansion_keeps_headerless_text_continuations(self):
        class Vector(list):
            def tolist(self):
                return list(self)

        class Embedding:
            def encode(self, _message):
                return Vector([0.1, 0.2])

        section = "H3: **4.4 Supervised fine-tuning** > H4: **4.4.1 Explicit Reasoning Dataset**"
        records = [
            ("overview", "Dataset overview.", {"source": "paper.pdf", "headers": section, "type": "text", "chunk_index": 53}),
            ("table", "|Indicator|Target|\n|---|---|\n|DILI|0.8|", {"source": "paper.pdf", "type": "table", "chunk_index": 54}),
            ("table-2", "|Indicator|Reward|\n|---|---|\n|HLM|Relative|", {"source": "paper.pdf", "type": "table", "chunk_index": 55}),
            ("bridge", "The pipeline works backward from verified outcomes.", {"source": "paper.pdf", "type": "text", "chunk_index": 56}),
            ("candidate", "DeepSeek-R1 proposes candidates; ADMETLab evaluates them; similarity > 0.6.", {"source": "paper.pdf", "type": "text", "chunk_index": 57}),
            ("threshold", "Fingerprint similarity must be greater than 0.6.", {"source": "paper.pdf", "type": "text", "chunk_index": 58}),
            ("next", "#### **4.4.2 Dataset Statistics**", {"source": "paper.pdf", "headers": "H4: **4.4.2 Dataset Statistics**", "type": "text", "chunk_index": 59}),
            ("other", "Unrelated introduction.", {"source": "paper.pdf", "headers": "H1: Introduction", "type": "text", "chunk_index": 1}),
            ("previous-heading", "#### **4.3 Previous Section**", {"source": "paper.pdf", "headers": "H4: **4.3 Previous Section**", "type": "text", "chunk_index": 50}),
            ("previous-continuation", "Previous section continuation.", {"source": "paper.pdf", "type": "text", "chunk_index": 51}),
            ("previous-continuation-2", "More previous section content.", {"source": "paper.pdf", "type": "text", "chunk_index": 52}),
        ]

        class Collection:
            def count(self):
                return len(records)

            def query(self, **_kwargs):
                return {
                    "ids": [["overview", "other"]],
                    "documents": [["Dataset overview.", "Unrelated introduction."]],
                    "metadatas": [[records[0][2], records[-1][2]]],
                }

            def get(self, **_kwargs):
                return {
                    "ids": [row[0] for row in records],
                    "documents": [row[1] for row in records],
                    "metadatas": [row[2] for row in records],
                }

        class Client:
            def __init__(self):
                self.chat = self
                self.completions = self

            def create(self, **_kwargs):
                return type(
                    "Response",
                    (),
                    {
                        "choices": [
                            type(
                                "Choice",
                                (),
                                {
                                    "message": type(
                                        "Message",
                                        (),
                                        {"content": "数据集包含 4,855 个样本。"},
                                    )()
                                },
                            )()
                        ]
                    },
                )()

        runtime = app.Runtime(
            app.RuntimeConfig(retrieval_k=2, context_k=6),
            Client(),
            Embedding(),
            Collection(),
        )
        result = app.query_knowledge(
            "显式推理数据集包含多少个样本？标注管道如何构建？",
            runtime=runtime,
        )
        joined = "\n".join(result["contexts"])
        self.assertIn("ADMETLab", joined)
        self.assertIn("0.6", joined)
        self.assertNotIn("Previous section", joined)
        self.assertIn("ADMETLab", result["contexts"][2])
        self.assertIn("0.6", result["contexts"][3])
        self.assertIn("【补充原文核对项】", result["answer"])
        self.assertIn("ADMETLab", result["answer"])
        self.assertIn("0.6", result["answer"])
        self.assertEqual(
            result["context_metadatas"][2]["section_context"],
            section,
        )
        self.assertEqual(
            result["context_metadatas"][3]["section_context"],
            section,
        )

    def test_section_continuation_accepts_repeated_section_headers(self):
        section = "H3: Dataset Statistics"
        metadatas = [
            {"source": "paper.pdf", "headers": section, "type": "text", "chunk_index": 32},
            {"source": "paper.pdf", "headers": section, "type": "text", "chunk_index": 33},
            {"source": "paper.pdf", "headers": "H3: Next Section", "type": "text", "chunk_index": 34},
        ]
        texts = ["overview", "continuation", "next"]
        self.assertEqual(
            app._section_continuation_indices(0, metadatas, texts, "paper.pdf", 2),
            [1],
        )

    def test_footnote_attachment_stays_on_source_page_and_skips_ambiguity(self):
        from types import SimpleNamespace

        snapshot = SimpleNamespace(
            ids=["right", "other-source", "other-page", "conflict-a", "conflict-b"],
            texts=[
                "> 2No human annotation or proprietary API costs.",
                "> 2Unrelated other-paper claim.",
                "> 2Unrelated other-page claim.",
                "> 3First interpretation.",
                "> 3Conflicting interpretation.",
            ],
            metadatas=[
                {"source": source, "page": page}
                for source, page in [
                    ("paper.pdf", 3), ("other.pdf", 3), ("paper.pdf", 4),
                    ("paper.pdf", 3), ("paper.pdf", 3),
                ]
            ],
        )
        with patch.object(app, "_get_lexical_snapshot", return_value=snapshot):
            texts, metas = app._attach_matching_footnotes(
                ["Cost-free<sup>2</sup> and another point<sup>3</sup>."],
                [{"source": "paper.pdf", "page": 3, "type": "text"}],
                None,
            )
        self.assertIn("No human annotation", texts[0])
        self.assertNotIn("Unrelated", texts[0])
        self.assertNotIn("interpretation", texts[0])
        self.assertEqual(metas[0]["footnote_chunk_ids"], ["right"])

    def test_list_expansion_ignores_table_headers_that_look_relevant(self):
        records = [
            ("anchor", "Rubric overview.", {"source": "paper.pdf", "type": "text", "chunk_index": 1,
                                           "headers": "H2: YESciEval rubric"}),
            ("bridge", "Three dimensions follow.", {"source": "paper.pdf", "type": "text", "chunk_index": 2}),
            ("rubrics", "Cohesion, Conciseness, Readability.", {"source": "paper.pdf", "type": "text", "chunk_index": 3}),
            ("table", "Unrelated comparison.", {"source": "paper.pdf", "type": "table", "chunk_index": 9,
                                              "headers": "|YESciEval rubric|YESciEval rubric|"}),
        ]

        class Collection:
            def get(self, **_kwargs):
                return {
                    "ids": [row[0] for row in records],
                    "documents": [row[1] for row in records],
                    "metadatas": [row[2] for row in records],
                }

        base = {
            "ids": [["anchor", "table"]],
            "documents": [[records[0][1], records[3][1]]],
            "metadatas": [[records[0][2], records[3][2]]],
        }
        runtime = app.Runtime(app.RuntimeConfig(), None, None, Collection())
        result = app._section_expansion_result(
            "YESciEval rubric 请列出每个维度的名称", base, runtime,
            source_filter="paper.pdf",
        )
        self.assertEqual(result["ids"][0][:3], ["anchor", "bridge", "rubrics"])
        self.assertNotIn("table", result["ids"][0])

    def test_section_expansion_keeps_cross_page_sentence_in_one_slot(self):
        records = [
            ("anchor", "Pipeline overview.", {"source": "paper.pdf", "type": "text", "page": 4,
                                               "chunk_index": 20, "headers": "H2: Pipeline"}),
            ("first", "Step 1 uses a seed dataset.", {"source": "paper.pdf", "type": "text", "page": 4,
                                                         "chunk_index": 21}),
            ("bridge", "Step 2 collects evidence for ( _a_ ,\n464", {"source": "paper.pdf", "type": "text",
                                                                         "page": 4, "chunk_index": 22}),
            ("continued", "_b_ ) from Wikipedia.", {"source": "paper.pdf", "type": "text",
                                                       "page": 5, "chunk_index": 23}),
            ("later", "Step 3 uses PaLM-2 to evaluate evidence.", {"source": "paper.pdf", "type": "text",
                                                                   "page": 5, "chunk_index": 24}),
        ]

        class Collection:
            def get(self, **_kwargs):
                return {
                    "ids": [row[0] for row in records],
                    "documents": [row[1] for row in records],
                    "metadatas": [row[2] for row in records],
                }

        runtime = app.Runtime(app.RuntimeConfig(context_k=4), None, None, Collection())
        base = {key: [[values[0]]] for key, values in {
            "ids": [records[0][0]], "documents": [records[0][1]], "metadatas": [records[0][2]],
        }.items()}
        result = app._section_expansion_result(
            "What roles does the pipeline use, and how does it evaluate evidence?",
            base, runtime, source_filter="paper.pdf",
        )
        self.assertEqual(result["ids"][0][:4], ["anchor", "first", "bridge", "later"])
        self.assertIn("_b_ ) from Wikipedia", result["documents"][0][2])
        self.assertEqual(result["metadatas"][0][2]["window_pages"], [4, 5])
        self.assertEqual(result["metadatas"][0][2]["window_chunk_ids"], ["bridge", "continued"])
        self.assertIn("第 4–5 页", app.format_evidence_panel({
            "contexts": [result["documents"][0][2]],
            "context_metadatas": [result["metadatas"][0][2]],
        }))
        records[2] = ("bridge", "Step 2 is complete.\n464", records[2][2])
        separate = app._section_expansion_result(
            "What roles does the pipeline use, and how does it evaluate evidence?",
            base, runtime, source_filter="paper.pdf",
        )
        self.assertEqual(separate["ids"][0][:5], ["anchor", "first", "bridge", "continued", "later"])

    def test_composite_section_expansion_skips_multi_source_collection(self):
        class Vector(list):
            def tolist(self):
                return list(self)

        class Embedding:
            def encode(self, _message):
                return Vector([0.1, 0.2])

        class Collection:
            def count(self):
                return 3

            def query(self, **_kwargs):
                return {
                    "ids": [["overview", "other"]],
                    "documents": [["Dataset overview.", "Other paper context."]],
                    "metadatas": [[
                        {"source": "paper.pdf", "headers": "H2: Pipeline", "type": "text"},
                        {"source": "other.pdf", "headers": "H2: Pipeline", "type": "text"},
                    ]],
                }

            def get(self, **_kwargs):
                return {
                    "ids": ["overview", "sibling", "other"],
                    "documents": ["Dataset overview.", "Should not be injected.", "Other paper context."],
                    "metadatas": [
                        {"source": "paper.pdf", "headers": "H2: Pipeline", "type": "text"},
                        {"source": "paper.pdf", "headers": "H2: Pipeline", "type": "text"},
                        {"source": "other.pdf", "headers": "H2: Pipeline", "type": "text"},
                    ],
                }

        class Client:
            def __init__(self):
                self.chat = self
                self.completions = self

            def create(self, **kwargs):
                return type(
                    "Response",
                    (),
                    {"choices": [type("Choice", (), {"message": type("Message", (), {"content": "ok"})()})()]},
                )()

        result = app.query_knowledge(
            "显式推理数据集包含多少个样本？标注管道如何构建？",
            runtime=app.Runtime(
                app.RuntimeConfig(retrieval_k=2, context_k=2),
                Client(),
                Embedding(),
                Collection(),
            ),
        )
        self.assertNotIn("Should not be injected.", result["contexts"])

    def test_table_intent_requires_an_explicit_table_reference(self):
        self.assertFalse(is_table_question("显式推理数据集包含多少个样本？"))
        self.assertFalse(is_table_question("训练样本量和成功比率是多少？"))
        self.assertFalse(is_table_question("该数值是否稳定？"))
        self.assertFalse(is_table_question("表格理解训练流程用了哪些数据集？"))
        self.assertFalse(is_table_question("哪些架构用于科学表格表示学习？"))
        self.assertTrue(is_table_question("Table 2 中 DrugR* 的得分是多少？"))
        self.assertTrue(is_table_question("表2中哪个模型最好？"))
        self.assertTrue(is_table_question("下表给出了哪些结果？"))
        self.assertTrue(is_table_question("该表格中哪个模型最好？"))

    def test_answer_table_description_keeps_prose_before_unrelated_tables(self):
        question = "每个扩展关系和三元组分别对应答案表中的什么？"
        texts = ["Each relation corresponds to a column in the answer table.", "| Model | Score |\n|---|---|\n| Example | 1 |"]
        metas = [{"type": "text"}, {"type": "table"}]
        self.assertFalse(is_table_question(question))
        self.assertEqual(rerank_table_first(question, texts, metas)[0], [0, 1])
        self.assertTrue(is_table_question("答案表2中哪个模型最好？"))
        self.assertTrue(is_table_question("该表格中的答案是什么？"))
        self.assertTrue(is_table_question("该答案表中的数值是多少？"))

    def test_figure_reference_distinguishes_main_and_extended_data_figures(self):
        self.assertEqual(
            figure_reference_from_question("Figure 1 中有哪些测试集？"),
            ("figure", 1),
        )
        self.assertEqual(figure_reference_from_question("图1中的样本数？"), ("figure", 1))
        self.assertEqual(figure_reference_from_question("Fig. 1d 中的模块"), ("figure", 1))
        self.assertEqual(
            figure_reference_from_question("Extended Data Fig. 1 显示什么？"),
            ("extended_data_figure", 1),
        )
        self.assertIsNone(figure_reference_from_question("论文中的图说明了什么？"))

    def test_spatial_figure_evidence_preserves_visual_groups(self):
        blocks = [
            (178.2, 298.6, 209.4, 311.2, "PDB\nprotein–RNA\n", 1, 0),
            (186.0, 310.9, 201.6, 317.3, "n = 25\n", 2, 0),
            (214.2, 298.6, 251.6, 311.2, "PDB\nprotein–dsDNA\n", 3, 0),
            (225.1, 310.9, 240.7, 317.3, "n = 38\n", 4, 0),
            (263.8, 298.6, 284.6, 305.1, "CASP15\n", 5, 0),
            (267.9, 304.8, 280.4, 317.3, "RNA\nn = 8\n", 6, 0),
            (334.3, 298.6, 367.4, 305.1, "Glycosylation\n", 7, 0),
            (343.1, 304.8, 358.6, 311.2, "n = 28\n", 8, 0),
            (
                39.7,
                449.7,
                293.6,
                598.5,
                "Fig. 1 | AF3 accurately predicts structures.\n",
                9,
                0,
            ),
            (20.0, 200.0, 100.0, 250.0, "ignored image", 10, 1),
        ]

        chunks = extract_spatial_figure_chunks(
            blocks,
            "paper.pdf",
            2,
            595.276,
            790.866,
        )

        self.assertEqual(len(chunks), 1)
        chunk = chunks[0]
        self.assertEqual(chunk.metadata["type"], "figure")
        self.assertEqual(chunk.metadata["figure_number"], 1)
        self.assertLess(chunk.page_content.index("protein–RNA"), chunk.page_content.index("n = 25"))
        self.assertLess(chunk.page_content.index("n = 25"), chunk.page_content.index("protein–dsDNA"))
        self.assertLess(chunk.page_content.index("RNA / n = 8"), chunk.page_content.index("n = 28"))
        self.assertIn("x=29.9-35.2%", chunk.page_content)
        self.assertIn("top-left origin", chunk.page_content)
        self.assertIn("y increases downward", chunk.page_content)
        self.assertNotIn("ignored image", chunk.page_content)

    def test_spatial_figure_evidence_separates_collapsed_decimal_labels(self):
        blocks = [
            (0, 10, 100, 20, "67.4068.80", 0, 0),
            (0, 30, 100, 50, "Figure 3 | chart", 1, 0),
        ]
        chunks = extract_spatial_figure_chunks(
            blocks,
            "paper.pdf",
            7,
            100,
            100,
            max_region_height=100,
        )
        self.assertEqual(len(chunks), 1)
        self.assertIn("67.40 / 68.80", chunks[0].page_content)

    def test_spatial_figure_evidence_keeps_side_by_side_captions_independent(self):
        blocks = [
            (0, 10, 40, 20, "left evidence", 0, 0),
            (60, 40, 100, 50, "right evidence", 1, 0),
            (0, 80, 40, 90, "Figure 1: left", 2, 0),
            (60, 100, 100, 110, "Figure 2: right", 3, 0),
        ]

        chunks = extract_spatial_figure_chunks(
            blocks,
            "paper.pdf",
            7,
            100,
            120,
            max_region_height=80,
        )

        self.assertEqual([chunk.metadata["figure_number"] for chunk in chunks], [1, 2])
        self.assertIn("right evidence", chunks[1].page_content)

    def test_split_preserves_pretyped_figure_chunk(self):
        figure = Chunk(
            "Figure 2 spatial text evidence\n[x=10.0-20.0%] n = 8",
            {"type": "figure", "figure_number": 2, "page": 3},
        )

        chunks = split_to_chunks([figure], "paper.pdf")

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].metadata["type"], "figure")
        self.assertEqual(chunks[0].metadata["figure_number"], 2)

    def test_split_preserves_pretyped_formula_chunk(self):
        formula = Chunk("A ∗ u = f", {"type": "formula", "page": 4})

        chunks = split_to_chunks([formula], "paper.pdf")

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].metadata["type"], "formula")
        self.assertEqual(chunks[0].page_content, "A ∗ u = f")

    def test_missing_pdf_formula_blocks_keep_normal_markdown_unchanged(self):
        blocks = missing_pdf_formula_blocks(
            "The discretized system can be expressed as:\nwhere u and f are vectors.",
            "The discretized system can be expressed as:\nA ∗ u = f\nwhere u and f are vectors.",
        )

        self.assertEqual(blocks, [
            "The discretized system can be expressed as:\nA ∗ u = f\nwhere u and f are vectors."
        ])

    def test_missing_pdf_formula_blocks_keep_wrapped_constraint_tail(self):
        blocks = missing_pdf_formula_blocks(
            "Relation Pruning : E = {(vi, r, vj) ∈D | r ∈R",
            "Relation Pruning : E = {(vi, r, vj) ∈D | r ∈R\n∧{ϕtype(vi), ϕtype(vj)} ⊆T}\nHere, ϕtype maps entities to types.",
        )

        self.assertIn("∧{ϕtype(vi), ϕtype(vj)} ⊆T}", blocks[0])

    def test_table_spans_are_removed_from_text_chunks(self):
        markdown = f"# Results\n\n**Table 1**\n{TABLE_1}\n\n**Table 2**\n{TABLE_2}\n\nNarrative."
        tables, body = extract_tables(markdown, "paper.pdf", {"page": 2})
        self.assertEqual([table.metadata["table_id"] for table in tables], ["table-1", "table-2"])
        self.assertNotIn("0.2712", body)
        self.assertNotIn("0.3404", body)
        chunks = split_to_chunks([Chunk(markdown, {"source": "paper.pdf", "page": 2})], "paper.pdf")
        self.assertEqual(sum(chunk.metadata.get("type") == "table" for chunk in chunks), 2)
        self.assertFalse(
            any(chunk.metadata.get("type") == "text" and "|---|---|" in chunk.page_content for chunk in chunks)
        )

    def test_after_table_captions_are_not_reused_for_the_next_table(self):
        markdown = (
            "|Metric|Value|\n|---|---|\n|A|1|\n\n"
            "Table 2: first\n\n"
            "|Metric|Value|\n|---|---|\n|B|2|\n\n"
            "Table 3: second\n"
        )
        tables, _body = extract_tables(markdown, "paper.pdf")
        self.assertEqual([table.metadata["table_number"] for table in tables], [2, 3])

    def test_single_page_caption_can_label_a_distant_table(self):
        markdown = (
            "|Dataset|Samples|\n|---|---|\n|WTQ|13,706|\n\n"
            "### Training data\n\nTable 1: Overview of datasets\n"
        )
        tables, _body = extract_tables(markdown, "paper.pdf")
        self.assertEqual(tables[0].metadata["table_number"], 1)

    def test_stacked_group_and_dataset_headers_are_combined(self):
        markdown = (
            "||In-domain|Out-of-domain|\n"
            "|---|---|---|\n"
            "|Model|FF-TQA|TFV|\n"
            "||FeTaQA|TabFact|\n"
            "|Table-R1-Zero|30.6|87.6|"
        )
        tables, _body = extract_tables(markdown, "paper.pdf")
        headers, rows = parse_markdown_table(tables[0].page_content)
        self.assertEqual(headers, ["Model", "In-domain FF-TQA FeTaQA", "Out-of-domain TFV TabFact"])
        self.assertEqual(rows[0][1:], ["30.6", "87.6"])

    def test_wrapped_group_headers_are_repaired_before_lookup(self):
        markdown = (
            "|**Method**|**Formul**<br>validity|**ation design**<br>success rate|\n"
            "|---|---|---|\n|Qwen|75.5|68.3|"
        )
        tables, _body = extract_tables(markdown, "paper.pdf")
        headers, _rows = parse_markdown_table(tables[0].page_content)
        self.assertIn("Formulation design validity", headers)
        self.assertIn("Formulation design success rate", headers)

    def test_centered_split_group_headers_use_repeated_metric_blocks(self):
        markdown = (
            "| |LL|M Turbo|||Qwen-3|2B|||Pret|rainedQ|wen-32B||\n"
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
            "|**Method**|**Type**|**Dataset**|CMB|MMCU||CMB-Clin||CMB|MMCU||CMB-Cli|n|\n"
            "|TC-RAG|Adaptive RAG|TC-RAG-uct|87.95|93.15|25.89|57.29|56.59|87.33|92.80|24.65|56.94|57.46|\n"
        )
        tables, _body = extract_tables(markdown, "tcrag.pdf")
        headers, _rows = parse_markdown_table(tables[0].page_content)
        self.assertEqual(headers[1:3], ["Type", "Dataset"])
        self.assertEqual(headers[3], "Qwen-32B CMB")
        self.assertEqual(headers[8], "Pretrained Qwen-32B CMB")

    def test_three_level_grouped_headers_consume_leaf_metric_row(self):
        markdown = (
            "| |LL|M Turbo|||Qwen-3|2B|||Pret|rainedQ|wen-32B||\n"
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
            "|**Method**|**Type**|**Dataset**|CMB|MMCU||CMB-Clin||CMB|MMCU||CMB-Cli|n|\n"
            "|||**Metric**|EM|EM|BLEU-1|BLEU-4|ROUGE|EM|EM|BLEU-1|BLEU-4|ROUGE|\n"
            "|TC-RAG|Adaptive RAG|TC-RAG-uct|87.95|93.15|25.89|57.29|56.59|87.33|92.80|24.65|56.94|57.46|\n"
        )
        tables, _body = extract_tables(markdown, "tcrag.pdf")
        headers, rows = parse_markdown_table(tables[0].page_content)
        self.assertEqual(len(rows), 1)
        self.assertIn("Qwen-32B CMB EM", headers)
        self.assertIn("Pretrained Qwen-32B CMB EM", headers)

    def test_chinese_average_metric_aliases_select_all_requested_columns(self):
        question = "Table 2 TC–RAG 的平均交互次数、平均检索器次数、平均耗时和平均 token 数分别是多少？"
        table = (
            "|Method|Avg. Interactions|Avg. Retrievers|Avg. Time(s)|Avg. Token|\n"
            "|---|---|---|---|---|\n"
            "|TC–RAG|4.78|3.37|50.91|458.82|"
        )
        row = extract_table_row_values(question, table, {"table_number": 2})
        self.assertEqual(
            [item["value"] for item in row["values"]],
            ["4.78", "3.37", "50.91", "458.82"],
        )

    def test_rank_metric_group_headers_and_split_row_are_repaired(self):
        question = "Table 6 中 Sparse BM-25 在 FinHybrid 与 PaperTab 的 @10 evidence score 分别是多少？"
        tables, _body = extract_tables(TABLE_RANK_GROUPED, "uda.pdf")
        headers, _rows = parse_markdown_table(tables[0].page_content)
        self.assertIn("FinHybrid @10", headers)
        self.assertIn("PaperTab @10", headers)
        row = extract_table_row_values(question, tables[0].page_content, tables[0].metadata)
        self.assertEqual([item["value"] for item in row["values"]], ["87.4", "90.0"])

    def test_blank_label_second_header_row_is_combined(self):
        markdown = """Table 3: Experimental results

|Models||Accu|racy||
|---|---|---|---|---|
||Comparison|Statistics|Relationship|Overall|
|GPT-4 + RAG|0.763|0.410|0.687|0.593|
"""
        tables, _body = extract_tables(markdown, "paper.pdf")
        content = tables[0].page_content
        headers, _rows = parse_markdown_table(content)
        self.assertEqual(
            headers,
            ["Models", "Models Comparison", "Accuracy Statistics", "Accuracy Relationship", "Accuracy Overall"],
        )
        row = extract_table_row_values(
            "Table 3 中 GPT-4 + RAG 在 Comparison、Statistics、Relationship 和 Overall 四列的准确率分别是多少？",
            content,
            tables[0].metadata,
        )
        self.assertEqual(
            [item["value"] for item in row["values"]],
            ["0.763", "0.410", "0.687", "0.593"],
        )

    def test_repeated_row_defaults_to_all_section_when_unqualified(self):
        markdown = """Table 3: Experimental results

|Models||Accu|racy||
|---|---|---|---|---|
||Comparison|Statistics|Relationship|Overall|
||**All sets**||||
|GPT-4 + RAG|0.763|0.410|0.687|0.593|
||**Set1(0-10**|**)**|||
|GPT-4 + RAG|0.870|0.619|0.740|0.729|
"""
        tables, _body = extract_tables(markdown, "paper.pdf")
        row = extract_table_row_values(
            "Table 3 中 GPT-4 + RAG 在 Comparison、Statistics、Relationship 和 Overall 四列的准确率分别是多少？",
            tables[0].page_content,
            tables[0].metadata,
        )
        self.assertEqual(
            [item["value"] for item in row["values"]],
            ["0.763", "0.410", "0.687", "0.593"],
        )

    def test_derived_average_absolute_column_wins_over_context_columns(self):
        question = "LiveXiv v1 与人工验证子集相比，Table 2 给出的 VQA 和 TQA 平均绝对变化分别是多少？"
        row = extract_table_row_values(
            question,
            TABLE_LIVEXIV_AVG,
            {"type": "table", "table_number": 2},
        )
        self.assertEqual(
            [(item["row"], item["values"][0]["value"]) for item in row["rows"]],
            [("VQA", "2.336"), ("TQA", "2.105")],
        )

    def test_row_entity_ignores_column_name_and_normalizes_sup(self):
        question = "Table 2 中 DrugR* 的 Target property F1 score 是多少？"
        entity = select_row_entity(question, TABLE_2)
        self.assertEqual(entity, "DrugR*")
        filtered = filter_table_rows_by_entity(TABLE_2, entity)
        self.assertIn("0.3404", filtered)
        self.assertNotIn("0.2997", filtered)
        self.assertEqual(normalize_for_match("**DrugR**"), "drugr")
        self.assertEqual(normalize_for_match("**DrugR**<sup>_∗_</sup>"), "drugr*")

    def test_parenthetical_method_variant_selects_the_full_row(self):
        table = (
            "|LLM|Method|F1|EM|\n|---|---|---|---|\n"
            "|**GPT4o-mini**|Ours (CxtInt)|47.87|38.50|"
        )
        row = extract_table_row_values(
            "Table 1 中 GPT4o-mini 的 CxtInt F1 和 EM 分别是多少？",
            table,
            {"type": "table", "table_number": 1},
        )
        self.assertEqual(
            [item["value"] for item in row["values"]],
            ["47.87", "38.50"],
        )

    def test_row_label_with_citation_before_parenthetical_is_selectable(self):
        table = (
            "|Method|Retrieval Accuracy|\n|---|---|\n"
            "|KAPING [1] (top-k triple retrieval)|60.81|\n"
            "|_G-Retriever_|70.49|"
        )
        row = extract_table_row_values(
            "G-Retriever Table 11 中 KAPING 和 G-Retriever 的 retrieval accuracy 分别是多少？",
            table,
            {"type": "table", "table_number": 11},
        )
        self.assertEqual(
            [(item["row"], item["values"][0]["value"]) for item in row["rows"]],
            [
                ("KAPING [1] (top-k triple retrieval)", "60.81"),
                ("G-Retriever", "70.49"),
            ],
        )

    def test_markdown_underscores_inside_row_labels_keep_word_boundaries(self):
        table = (
            "|Hyperparameter|Assignment|\n|---|---|\n"
            "|learning Rate|1e-4|\n|warmup ratio|0.05|\n"
            "|LoRa alpha|64|\n|LoRa_r_|32|"
        )
        row = extract_table_row_values(
            "Table 8 中 learning rate、warmup ratio、LoRA alpha 和 LoRA r 分别设置为多少？",
            table,
            {"type": "table", "table_number": 8},
        )
        self.assertEqual(
            [(item["row"], item["values"][0]["value"]) for item in row["rows"]],
            [
                ("learning Rate", "1e-4"),
                ("warmup ratio", "0.05"),
                ("LoRa alpha", "64"),
                ("LoRa r", "32"),
            ],
        )

    def test_dataset_row_entities_are_not_reused_as_row_filters(self):
        table = (
            "|Dataset|Size|#O.|Avg. L|\n|---|---|---|---|\n"
            "|PubMedQA|500|3|211|\n|BioASQ-Y/N|618|2|36|"
        )
        row = extract_table_row_values(
            "Table 2 中 PubMedQA 和 BioASQ-Y/N 的数据集大小分别是多少？",
            table,
            {"type": "table", "table_number": 2},
        )
        self.assertEqual(
            [(item["row"], item["values"][0]["value"]) for item in row["rows"]],
            [("PubMedQA", "500"), ("BioASQ-Y/N", "618")],
        )

    def test_chinese_table_column_aliases_select_document_length_and_open(self):
        corpus = (
            "|Corpus|#Doc.|#Snippets|Avg. L|Domain|\n|---|---|---|---|---|\n"
            "|MedCorp|30.4M|54.2M|221|Medical|"
        )
        row = extract_table_row_values(
            "Table 3 中 MedCorp 的原始文档数、snippets 数和平均长度分别是多少？",
            corpus,
            {"type": "table", "table_number": 3},
        )
        self.assertEqual(
            [(item["column"], item["value"]) for item in row["values"]],
            [("#Doc.", "30.4M"), ("#Snippets", "54.2M"), ("Avg. L", "221")],
        )

        llms = (
            "|LLM|Size|Context|Open|Domain|\n|---|---|---|---|---|\n"
            "|GPT-4|?|32,768|No|General|\n|Mixtral|?|32,768|Yes|General|"
        )
        row = extract_table_row_values(
            "Table 5 中 GPT-4 和 Mixtral 的 context length 及是否开源分别是什么？",
            llms,
            {"type": "table", "table_number": 5},
        )
        self.assertEqual(
            [[item["value"] for item in entry["values"]] for entry in row["rows"]],
            [["32,768", "No"], ["32,768", "Yes"]],
        )

    def test_multilevel_table_header_row_is_not_an_entity(self):
        table = (
            "|LLMs|Methods|MuS|iQue|2W|iki|\n|---|---|---|---|---|---|\n"
            "|||F1|EM|F1|EM|\n"
            "|**GPT4o-mini**|Ours (AnsInt)|50.54|37.00|62.55|52.00|\n"
            "||Ours (CxtInt)|47.87|38.50|56.54|50.50|"
        )
        row = extract_table_row_values(
            "Table 1 中 GPT4o-mini 的 CxtInt 在 MuSiQue 和 2Wiki 上的 F1/EM 分别是多少？",
            table,
            {"type": "table", "table_number": 1},
        )
        self.assertEqual(
            [item["value"] for item in row["values"]],
            ["47.87", "38.50", "56.54", "50.50"],
        )
        self.assertEqual(row["outer_group"], "GPT4o-mini")
        filtered = filter_table_rows_by_entity(table, "Ours (CxtInt)")
        self.assertIn("GPT4o-mini", filtered)

    def test_normalize_for_match_repairs_spaced_pdf_decimal(self):
        self.assertEqual(normalize_for_match("entropy (≥ 4 _._ 5)"), "entropy (≥ 4.5)")

    def test_structured_cell_lookup_supports_chinese_alias(self):
        cell = extract_table_cell(
            "Table 2 中 DrugR*（在少量新数据上微调后）的整体优化得分是多少？",
            TABLE_2_FULL,
            {"type": "table", "table_number": 2},
        )
        self.assertIsNotNone(cell)
        self.assertEqual(cell["value"], "0.2060")

        cell = extract_table_cell(
            "Table 2 中 DrugR* 的 Target property F1 score 是多少？",
            TABLE_2,
            {"type": "table", "table_number": 2},
        )
        self.assertIsNotNone(cell)
        self.assertEqual(cell["value"], "0.3404")

    def test_structured_row_lookup_returns_multiple_value_columns(self):
        row = extract_table_row_values(
            "Table 6 中 Unique SMILES 的数量和占比是多少？",
            TABLE_6,
            {"type": "table", "table_caption": "**Table 6** Molecular complexity"},
        )
        self.assertIsNotNone(row)
        self.assertEqual(row["table_number"], "6")
        self.assertEqual(row["row"], "Unique SMILES")
        self.assertEqual(
            row["values"],
            [
                {"column": "Mean / Value", "value": "3,863 / 4,826"},
                {"column": "Range / Definition", "value": "80.05%"},
            ],
        )

    def test_parallel_break_cells_keep_dataset_counts_aligned(self):
        table = (
            "|Task|Dataset|Samples|\n|---|---|---|\n"
            "|Short-form QA|WTQ (citation)<br>HiTab (citation)|13,706<br>6,793|\n"
            "|Fact Verification|TabFact (citation)|20,740|"
        )
        row = extract_table_row_values(
            "Table 1 中 WTQ、HiTab 和 TabFact 的训练样本数分别是多少？",
            table,
            {"type": "table", "table_number": 1},
        )
        self.assertEqual(
            [(item["row"], item["values"][-1]["value"]) for item in row["rows"]],
            [("WTQ", "13,706"), ("HiTab", "6,793"), ("TabFact", "20,740")],
        )

    def test_table_caption_metric_note_follows_question(self):
        metadata = {
            "table_caption": "Table 4: Results measured by exact match."
        }
        self.assertEqual(
            app._table_caption_metric_note(
                metadata,
                "Table 4 中使用 Qwen-2.5-72b 的 exact match 分数是多少？",
            ),
            "（表格指标：exact match）",
        )
        self.assertEqual(
            app._table_caption_metric_note(metadata, "Table 4 中的 HeteQA 分数是多少？"),
            "",
        )

    def test_structured_target_set_alias_resolves_table_row(self):
        cell = extract_table_cell(
            "Table 5 中抗高血压药物（Antihypertensive）类别用于结合亲和力评估的靶点集合有哪些？",
            TABLE_5,
            {"type": "table", "table_caption": "**Table 5** Targets"},
        )
        self.assertIsNotNone(cell)
        self.assertEqual(cell["column"], "Target set")
        self.assertIn("ACE", cell["value"])
        self.assertIn("ADRB2", cell["value"])

    def test_table_entity_can_be_in_a_non_first_cell(self):
        question = "Table 1 的 FlanT5-xl 的 METEOR、ROUGE-1 和 BertS 分别是多少？"
        self.assertEqual(select_row_entity(question, TABLE_HIERARCHICAL), "FlanT5-xl")
        row = extract_table_row_values(question, TABLE_HIERARCHICAL, {"type": "table", "table_number": 1})
        self.assertIsNotNone(row)
        self.assertEqual(row["row"], "FlanT5-xl")
        self.assertEqual(
            row["values"],
            [
                {"column": "METEOR", "value": "0.08"},
                {"column": "ROUGE-1", "value": "0.10"},
                {"column": "BertS", "value": "0.78"},
            ],
        )

    def test_table_entity_matches_pdf_dropped_model_punctuation(self):
        question = "Table 1 中 ChatGPT-4o-Mini 使用 THINKNOTE 在 PopQA 和 NQ 上的 accuracy 分别是多少？"
        table = """|Method|LLM|PopQA|NQ|
|---|---|---|---|
|THINKNOTE|Llama-3-Ins-70B|69.8|68.4|
||ChatGPT-4oMINI|69.8|71.0|"""
        self.assertEqual(select_row_entity(question, table), "ChatGPT-4oMINI")
        row = extract_table_row_values(question, table, {"type": "table", "table_number": 1})
        self.assertIsNotNone(row)
        self.assertEqual([item["value"] for item in row["values"]], ["69.8", "71.0"])

    def test_table_entity_lookup_matches_named_columns_with_metric_suffix(self):
        question = "Table 1 中 ChatGPT-4o-Mini 使用 THINKNOTE 在 PopQA 和 NQ 上的 accuracy 分别是多少？"
        table = """|Method|LLM|PopQA<br>(_acc_)|NQ<br>(_acc_)|
|---|---|---|---|
|THINKNOTE|ChatGPT-4oMINI|69.8|71.0|"""
        row = extract_table_row_values(question, table, {"type": "table", "table_number": 1})
        self.assertIsNotNone(row)
        self.assertEqual([item["value"] for item in row["values"]], ["69.8", "71.0"])

    def test_usage_qualified_table_lookup_keeps_inherited_method_group(self):
        question = "Table 1 中 ChatGPT-4o-Mini 使用 THINKNOTE 在 PopQA 和 NQ 上的 accuracy 分别是多少？"
        table = """|Method|LLM|PopQA|NQ|
|---|---|---|---|
|Direct QA|ChatGPT-4oMINI|32.6|51.0|
|THINKNOTE|Llama-3-Ins-70B|69.8|68.4|
||ChatGPT-4oMINI|69.8|71.0|"""
        _order, _note, filtered = rerank_table_first(
            question, [table], [{"type": "table", "table_number": 1}]
        )
        self.assertIn("|THINKNOTE|Llama-3-Ins-70B|69.8|68.4|", filtered[0])
        row = extract_table_row_values(
            question, filtered[0], {"type": "table", "table_number": 1}
        )
        self.assertIsNotNone(row)
        self.assertEqual([item["value"] for item in row["values"]], ["69.8", "71.0"])

    def test_table_lookup_disambiguates_model_and_dataset(self):
        question = "Table 2 中 FigEx-7B 在 BioSci-Fig 上的 BLEU 和 ROUGE-1 分别是多少？"
        row = extract_table_row_values(
            question,
            TABLE_MULTI_DATASET,
            {"type": "table", "table_number": 2},
        )
        self.assertIsNotNone(row)
        self.assertEqual(row["row"], "FigEx-7B")
        self.assertEqual([item["value"] for item in row["values"]], ["0.72", "0.81"])

    def test_table_lookup_keeps_model_when_input_qualifier_shares_row(self):
        markdown = (
            "Table 2: RealHiTBench results\n\n"
            "|**Model**|**Input**|**Fact C**|**hecking**|**Numerical**|**Reasoning**|"
            "**Structure Co**|**mprehendin**|**g**<br>**Data A**|**nalysis**|"
            "**Chart G**|**eneration**|\n"
            "|---|---|---|---|---|---|---|---|---|---|---|---|\n"
            "|||EM|F1|EM|F1|EM|F1|GPT-EVAL|ROUGE|PASS@1|ECR|\n"
            "|||||**Table-**|**oriented Mod**|**els**||||||\n"
            "|**GPT4o(TreeThinker)**|Text|64.50|72.41|53.34|65.08|64.40|75.67|"
            "77.26|37.63|39.47|67.76|\n"
            "|**GPT4o(TreeThinker)**|Image|44.13|52.41|40.57|49.35|49.21|58.32|"
            "70.83|34.44|19.61|67.32|\n"
            "|**GPT4o(TreeThinker)**|Image+Text|65.82|73.32|55.60|64.28|"
            "66.31|77.42|79.45|37.08|33.55|65.13|"
        )
        tables, _body = extract_tables(markdown, "realhitbench.pdf")
        headers, _rows = parse_markdown_table(tables[0].page_content)
        self.assertIn("Numerical Reasoning F1", headers)
        self.assertNotIn("Table-oriented Models", headers)
        question = (
            "RealHiTBench Table 2 中 GPT4o(TreeThinker) 的 Image+Text 输入在 "
            "Fact Checking、Numerical Reasoning 和 Structure Comprehending 三项的 F1 分数分别是多少？"
        )
        row = extract_table_row_values(question, tables[0].page_content, tables[0].metadata)
        self.assertEqual(row["row"], "GPT4o(TreeThinker)")
        self.assertEqual(
            [item["value"] for item in row["values"]], ["73.32", "64.28", "77.42"]
        )

    def test_table_lookup_returns_named_rows_and_setting_groups(self):
        question = (
            "Table 5 中，Gemini Flash 在 oracle setting 的 overall F1、"
            "Human baseline 的 F1，以及 PaLM-2 在 closed book setting 的 overall F1 分别是多少？"
        )
        row = extract_table_row_values(
            question,
            TABLE_MULTI_SETTING,
            {"type": "table", "table_number": 5},
        )
        self.assertIsNotNone(row)
        self.assertEqual(
            [(item["row"], item["values"][0]["value"]) for item in row["rows"]],
            [("Gemini Flash", "60.7"), ("Human", "73.0"), ("PaLM-2", "47.6")],
        )

    def test_table_lookup_selects_section_covering_all_requested_entities(self):
        row = extract_table_row_values(
            "Table 4 中，TANQ 的 no skill、one skill、two skills 和 three skills 分别有多少个样本？",
            TABLE_SECTION_GROUPS,
            {"type": "table", "table_number": 4},
        )
        self.assertIsNotNone(row)
        self.assertEqual(
            [item["values"][0]["value"] for item in row["rows"]],
            ["300", "657", "94", "23"],
        )

    def test_table_multi_line_values_select_requested_representation(self):
        row = extract_table_row_values(
            "Table 3 中 WikiTQ+SQA+SciGen 使用表格表示时的 MSE 和 F1-score 是多少？",
            TABLE_MULTI_VARIANT,
            {"type": "table", "table_number": 3},
        )
        self.assertIsNotNone(row)
        self.assertEqual(row["row"], "WikiTQ+SQA+SciGen")
        self.assertEqual(
            row["values"],
            [
                {"column": "MSE", "value": "2.30"},
                {"column": "F1-score", "value": "0.38"},
            ],
        )

    def test_table_group_marker_selects_matching_duplicate_entity(self):
        question = (
            "Table 2 的 Test (Other) 中，WikiTQ+SQA+SciGen 的 FlanT5-xl "
            "METEOR、ROUGE-1 和 BertS 分别是多少？"
        )
        row = extract_table_row_values(
            question,
            TABLE_GROUPED,
            {"type": "table", "table_number": 2},
        )
        self.assertIsNotNone(row)
        self.assertEqual(row["values"][0]["value"], "0.14")
        self.assertEqual(row["values"][1]["value"], "0.23")
        self.assertEqual(row["values"][2]["value"], "0.85")

    def test_setting_column_qualifier_selects_composite_dataset_row(self):
        question = (
            "Table 2 的 Test (C&L) 中，WikiTQ+SQA+SciGen 的 FlanT5-xl "
            "得分是多少？"
        )
        tables, _ = extract_tables(TABLE_SETTING_QUALIFIER, "paper.pdf")
        row = extract_table_row_values(
            question,
            tables[0].page_content,
            tables[0].metadata,
        )
        self.assertIsNotNone(row)
        self.assertEqual(
            [item["value"] for item in row["values"][-3:]],
            ["0.15(+0.07)", "0.24(+0.14)", "0.85(+0.07)"],
        )

    def test_split_multi_header_and_wrapped_section_select_correct_row(self):
        question = (
            "Table 3 中 Llama-3-Ins-70B 的 THINKNOTE 在 Miss-Answer 场景的 "
            "PopQA、NQ 和 TriviaQA 数值分别是多少？"
        )
        tables, _ = extract_tables(TABLE_SPLIT_MULTI_HEADER, "paper.pdf")
        row = extract_table_row_values(
            question,
            tables[0].page_content,
            tables[0].metadata,
        )
        self.assertIsNotNone(row)
        self.assertEqual(
            [item["value"] for item in row["values"]],
            ["12.2", "33.3", "54.0"],
        )

    def test_split_model_section_is_not_treated_as_first_repeated_row(self):
        question = (
            "Table 3 中 Llama-3-Ins-70B 的 THINKNOTE 在 Miss-Answer 场景的 "
            "PopQA、NQ 和 TriviaQA 数值分别是多少？"
        )
        content = (
            "|Mthd|Miss-Answer PopQA|Miss-Answer NQ|Miss-Answer TriviaQA|\n"
            "|---|---:|---:|---:|\n"
            "|**Llama-3-Ins-8B**||||\n"
            "|THINKNOTE|8.3|25.8|36.0|\n"
            "|**Llama-3-Ins-70**|**B**|||\n"
            "|THINKNOTE|12.2|33.3|54.0|"
        )
        row = extract_table_row_values(
            question,
            content,
            {"type": "table", "table_number": 3},
        )
        self.assertIsNotNone(row)
        self.assertEqual(
            [item["value"] for item in row["values"]],
            ["12.2", "33.3", "54.0"],
        )
        filtered = filter_table_rows_by_entity(content, "THINKNOTE")
        self.assertIsNotNone(filtered)
        self.assertIn("Llama-3-Ins-70", filtered)
        filtered_row = extract_table_row_values(
            question,
            filtered,
            {"type": "table", "table_number": 3},
        )
        self.assertEqual(
            [item["value"] for item in filtered_row["values"]],
            ["12.2", "33.3", "54.0"],
        )

    def test_explicit_group_does_not_match_ungrouped_table(self):
        question = "Table 2 的 Test (C&L) 中，DrugR* 的 Target property F1 score 是多少？"
        self.assertIsNone(
            extract_table_cell(
                question,
                TABLE_2_FULL,
                {"type": "table", "table_number": 2},
            )
        )

    def test_qualified_metric_columns_use_dataset_context(self):
        row = extract_table_row_values(
            "Table 1 中 MgNO 在 Darcy rough 基准上的相对 L2 和 H1 误差是多少？",
            TABLE_QUALIFIED_METRICS,
            {"type": "table", "table_number": 1},
        )
        self.assertIsNotNone(row)
        self.assertEqual([item["value"] for item in row["values"]], ["0.339", "1.380"])

    def test_repeated_rows_use_requested_variants(self):
        row = extract_table_row_values(
            "Table 4 中基线 MgNO 和六层 MgNO 的 L2 Error 分别是多少？",
            TABLE_MULTI_ROW,
            {"type": "table", "table_number": 4},
        )
        self.assertIsNotNone(row)
        self.assertEqual(
            [(item["row"], item["values"][0]["value"]) for item in row["rows"]],
            [("MgNO, 6 layers", "1.47"), ("Baseline MgNO", "1.63")],
        )

    def test_full_text_alias_selects_ft_column(self):
        row = extract_table_row_values(
            "Table 3 中 GPT-4o 在 RAG 和 full-text 配置下的 Avg 分别是多少？",
            TABLE_CONFIG_VARIANTS,
            {"type": "table", "table_number": 3},
        )
        self.assertIsNotNone(row)
        self.assertEqual(
            [(item["column"], item["value"]) for item in row["values"]],
            [("RAG", "46.63"), ("FT", "54.03")],
        )

    def test_comparative_table_question_preserves_all_rows(self):
        question = "结合 Table 1 的数据，DrugR 相比各基线模型在哪些指标上取得了最优结果？"
        self.assertTrue(is_comparative_table_question(question))
        order, _, filtered = rerank_table_first(
            question,
            [TABLE_1],
            [{"type": "table", "table_caption": "**Table 1** Results"}],
        )
        self.assertEqual(order, [0])
        self.assertIn("GPT5", filtered[0])
        self.assertIn("DrugR", filtered[0])

    def test_derived_value_intent_has_english_chinese_and_lookup_boundaries(self):
        self.assertTrue(is_derived_value_question("Table 1 中两项得分相差多少？"))
        self.assertTrue(is_derived_value_question("Compute the difference between the two scores."))
        self.assertTrue(is_derived_value_question("Table 1 三项得分的算术平均是多少？"))
        self.assertTrue(is_derived_value_question("What is the relative increase in percentage?"))
        self.assertFalse(is_derived_value_question("Table 1 中 DrugR 的 Score 是多少？"))
        self.assertFalse(is_derived_value_question("What is the value in the Score column?"))
        self.assertFalse(is_derived_value_question("Table 2 中涉及表格的问题占比是多少？"))
        self.assertFalse(is_derived_value_question("What is the percentage in the table?"))

    def test_multiple_explicit_table_numbers_are_matched_together(self):
        question = "按 Table 1 和 Table 2 计算两项指标的提升。"
        self.assertEqual(table_labels_from_question(question), ("1", "2"))
        self.assertEqual(
            matching_table_indices(
                question,
                [TABLE_1, TABLE_2],
                [
                    {"type": "table", "table_number": 1},
                    {"type": "table", "table_number": 2},
                ],
            ),
            [0, 1],
        )

    def test_derived_table_question_keeps_full_table_and_skips_cell_lookup(self):
        question = "Table 1 中 DrugR 的 Score 比 GPT5 高多少？"
        self.assertIsNone(
            find_table_cell_in_chunks(
                question, [TABLE_1], [{"type": "table", "table_number": 1}]
            )
        )
        order, _, filtered = rerank_table_first(
            question,
            [TABLE_1],
            [{"type": "table", "table_number": 1}],
        )
        self.assertEqual(order, [0])
        self.assertEqual(filtered, [TABLE_1])

    def test_derived_table_question_calls_generation_with_full_operands(self):
        class Vector(list):
            def tolist(self):
                return list(self)

        class Embedding:
            def encode(self, _message):
                return Vector([0.1, 0.2])

        class Collection:
            def count(self):
                return 1

            def query(self, **_kwargs):
                return {
                    "ids": [["table1"]],
                    "documents": [[TABLE_1]],
                    "metadatas": [[{"type": "table", "table_number": 1}]],
                }

            def get(self, **_kwargs):
                return {
                    "ids": ["table1"],
                    "documents": [TABLE_1],
                    "metadatas": [{"type": "table", "table_number": 1}],
                }

        class Client:
            def __init__(self):
                self.prompt = None
                self.chat = self
                self.completions = self

            def create(self, **kwargs):
                self.prompt = kwargs["messages"][1]["content"]
                return type(
                    "Response",
                    (),
                    {
                        "choices": [
                            type(
                                "Choice",
                                (),
                                {"message": type("Message", (), {"content": "0.0743"})()},
                            )
                        ]
                    },
                )()

        client = Client()
        result = app.query_knowledge(
            "Table 1 中 DrugR 的 Score 比 GPT5 高多少？",
            runtime=app.Runtime(
                app.RuntimeConfig(context_k=1), client, Embedding(), Collection()
            ),
        )
        self.assertEqual(result["answer"], "0.0743")
        self.assertIsNotNone(client.prompt)
        self.assertIn("0.2712", client.prompt)
        self.assertIn("0.1969", client.prompt)

    def test_multi_table_derived_question_keeps_all_named_tables(self):
        class Vector(list):
            def tolist(self):
                return list(self)

        class Embedding:
            def encode(self, _message):
                return Vector([0.1, 0.2])

        table_one = """|Setting|Model|ROUGE-1|
|---|---|---|
|Zero|FlanT5-xl|0.10|"""
        table_two = """|Setting|Model|ROUGE-1|
|---|---|---|
|WikiTQ+SQA+SciGen|FlanT5-xl|0.24|"""

        class Collection:
            def count(self):
                return 2

            def query(self, **_kwargs):
                return {
                    "ids": [["table1"]],
                    "documents": [[table_one]],
                    "metadatas": [[{"type": "table", "table_number": 1}]],
                }

            def get(self, **_kwargs):
                return {
                    "ids": ["table1", "table2"],
                    "documents": [table_one, table_two],
                    "metadatas": [
                        {"type": "table", "table_number": 1},
                        {"type": "table", "table_number": 2},
                    ],
                }

        class Client:
            def __init__(self):
                self.prompt = None
                self.chat = self
                self.completions = self

            def create(self, **kwargs):
                self.prompt = kwargs["messages"][1]["content"]
                return type(
                    "Response",
                    (),
                    {
                        "choices": [
                            type(
                                "Choice",
                                (),
                                {"message": type("Message", (), {"content": "0.14"})()},
                            )
                        ]
                    },
                )()

        client = Client()
        result = app.query_knowledge(
            "按 Table 1 和 Table 2，WikiTQ+SQA+SciGen 的 FlanT5-xl ROUGE-1 比零样本提高多少？",
            runtime=app.Runtime(
                app.RuntimeConfig(context_k=2), client, Embedding(), Collection()
            ),
        )
        self.assertEqual(result["answer"], "0.14")
        self.assertIn("0.10", client.prompt)
        self.assertIn("0.24", client.prompt)

    def test_cross_document_question_keeps_one_context_per_routed_source(self):
        class Vector(list):
            def tolist(self):
                return list(self)

        class Embedding:
            def encode(self, _message):
                return Vector([0.1, 0.2])

        documents = [
            "DrugR reinforcement learning uses GRPO.",
            "DrugR overview.",
            "MgNO baseline is configured with 5 levels.",
            "MgNO overview.",
        ]
        metadatas = [
            {"source": "drugr.pdf", "type": "text", "page": 1, "headers": "H1: DrugR Study > H2: Method"},
            {"source": "drugr.pdf", "type": "text", "page": 2},
            {"source": "mgno.pdf", "type": "text", "page": 1, "headers": "H1: MgNO Study"},
            {"source": "mgno.pdf", "type": "text", "page": 2},
        ]

        class Collection:
            def count(self):
                return len(documents)

            def query(self, **_kwargs):
                return {
                    "ids": [["drugr-1", "drugr-2", "mgno-1", "mgno-2"]],
                    "documents": [documents],
                    "metadatas": [metadatas],
                }

            def get(self, **_kwargs):
                return {
                    "ids": ["drugr-1", "drugr-2", "mgno-1", "mgno-2"],
                    "documents": documents,
                    "metadatas": metadatas,
                }

        class Client:
            def __init__(self):
                self.prompt = None
                self.chat = self
                self.completions = self

            def create(self, **kwargs):
                self.prompt = kwargs["messages"][1]["content"]
                return type(
                    "Response",
                    (),
                    {"choices": [type("Choice", (), {"message": type("Message", (), {"content": "ok"})()})()]},
                )()

        client = Client()
        result = app.query_knowledge(
            "结合两篇论文：DrugR 强化学习阶段使用什么算法，MgNO 的基线配置有多少个 levels？",
            runtime=app.Runtime(
                app.RuntimeConfig(
                    retrieval_mode="dense",
                    retrieval_k=4,
                    context_k=2,
                    document_routing=True,
                    query_decomposition=True,
                ),
                client,
                Embedding(),
                Collection(),
            ),
        )
        self.assertEqual(result["answer"], "ok")
        self.assertEqual(
            {metadata["source"] for metadata in result["context_metadatas"]},
            {"drugr.pdf", "mgno.pdf"},
        )
        self.assertIn("GRPO", client.prompt)
        self.assertIn("5 levels", client.prompt)
        self.assertIn("文档：DrugR Study", client.prompt)
        self.assertIn("文档：MgNO Study", client.prompt)

    def test_caption_detection_does_not_match_stable(self):
        markdown = "Figure 5 stable training dynamics.\n\n|x|y|\n|---|---|\n|0|1|"
        tables, _ = extract_tables(markdown, "paper.pdf")
        self.assertEqual(len(tables), 1)
        self.assertEqual(tables[0].metadata["table_id"], "table-unnamed-1")

    def test_rerank_excludes_legacy_table_text_for_explicit_table(self):
        texts = [TABLE_1, "Narrative", TABLE_2]
        metas = [
            {"type": "text"},
            {"type": "text"},
            {"type": "table", "table_number": 2, "table_id": "table-2"},
        ]
        order, _, filtered = rerank_table_first(
            "Table 2 中 DrugR* 的 Target property F1 score 是多少？", texts, metas
        )
        self.assertEqual(order, [2, 1])
        self.assertIn("0.3404", filtered[2])
        self.assertNotIn("0.2712", [filtered[index] for index in order])


class RuntimeContractTests(unittest.TestCase):
    def test_pdf_loader_keeps_column_text_under_its_own_heading(self):
        import pymupdf

        entries = [
            ("text", [70, 70, 290, 110], "Left column continuation.\n\n"),
            ("section-header", [305, 75, 525, 85], "### Implementation Details\n\n"),
            ("text", [305, 90, 525, 115], "Recall uses thirty candidates.\n\n"),
            ("text", [70, 120, 290, 150], "Document Reference belongs to the left column.\n\n"),
            ("section-header", [70, 160, 290, 170], "### Discussion\n\n"),
            ("text", [70, 180, 290, 220], "Dataset discussion.\n\n"),
            ("text", [305, 120, 525, 220], "Reranking keeps three candidates.\n\n"),
            ("page-footer", [285, 780, 315, 792], "12345\n\n"),
        ]
        boxes = []
        text = ""
        for box_class, bbox, content in entries:
            boxes.append({"class": box_class, "bbox": bbox, "pos": (len(text), len(text) + len(content))})
            text += content
        page_chunk = {"text": text, "page_boxes": boxes}
        ordered = app._pdf_markdown_in_column_order(page_chunk, 600)
        self.assertEqual(ordered, "".join(entries[i][2] for i in (0, 3, 4, 5, 1, 2, 6)))
        self.assertEqual(app._pdf_markdown_in_column_order(page_chunk, 1200), text.replace("12345\n\n", ""))
        spanning = {**page_chunk, "page_boxes": [dict(box) for box in boxes]}
        spanning["page_boxes"][0]["bbox"] = [70, 70, 525, 110]
        self.assertEqual(app._pdf_markdown_in_column_order(spanning, 600), text.replace("12345\n\n", ""))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "columns.pdf"
            with pymupdf.open() as document:
                document.new_page(width=600, height=800)
                document.save(path)
            with patch("pymupdf4llm.to_markdown", return_value=[page_chunk]):
                chunks = app.load_and_split_document(str(path))
        parameter = next(chunk for chunk in chunks if "Reranking" in chunk.page_content)
        self.assertIn("Implementation Details", parameter.metadata["headers"])
        self.assertNotIn("Discussion", parameter.metadata["headers"])
        reference = next(chunk for chunk in chunks if "Document Reference" in chunk.page_content)
        self.assertNotIn("Implementation Details", reference.metadata.get("headers", ""))

    def test_pdf_loader_keeps_native_page_chunks(self):
        import pymupdf

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pages.pdf"
            with pymupdf.open() as document:
                for text in ("First page contains retrieval evidence.", "Second page contains model evaluation."):
                    document.new_page().insert_text((72, 72), text)
                document.save(path)
            chunks = app.load_and_split_document(str(path))
        self.assertEqual({chunk.metadata["page"] for chunk in chunks}, {1, 2})
        self.assertTrue(all(chunk.metadata["source"] == "pages.pdf" for chunk in chunks))

    def test_pdf_loader_repairs_decimal_dot_markdown_artifact(self):
        import pymupdf

        text = "TANQ answer tables have 6 _._ 7 rows and 4 columns. Value 0.5 stays unchanged."
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "decimal.pdf"
            with pymupdf.open() as document:
                document.new_page().insert_text((72, 72), text)
                document.save(path)
            with patch("pymupdf4llm.to_markdown", return_value=[{"text": text, "page_boxes": []}]):
                chunks = app.load_and_split_document(str(path))

        content = "\n".join(chunk.page_content for chunk in chunks)
        self.assertIn("6.7 rows", content)
        self.assertIn("Value 0.5", content)
        self.assertNotIn("6 _._ 7", content)

    def test_import_does_not_initialize_external_resources(self):
        with (
            patch("sentence_transformers.SentenceTransformer") as embedding,
            patch("openai.OpenAI") as client,
            patch("chromadb.PersistentClient") as database,
        ):
            importlib.reload(app)
        embedding.assert_not_called()
        client.assert_not_called()
        database.assert_not_called()

    def test_runtime_config_defaults_dense_and_validates_hybrid_settings(self):
        with patch.dict(
            os.environ,
            {
                "SCI_RAG_RETRIEVAL_MODE": "hybrid",
                "SCI_RAG_HYBRID_RRF_K": "45",
                "SCI_RAG_RERANKER_MODEL": "BAAI/bge-reranker-base",
                "SCI_RAG_RERANKER_REVISION": "fixed-revision",
            },
            clear=True,
        ):
            config = app.RuntimeConfig.from_env()
        self.assertEqual(config.retrieval_mode, "hybrid")
        self.assertEqual(config.hybrid_rrf_k, 45)
        self.assertEqual(config.reranker_model, "BAAI/bge-reranker-base")
        self.assertEqual(config.reranker_revision, "fixed-revision")

        with patch.dict(os.environ, {"SCI_RAG_RETRIEVAL_MODE": "unsupported"}, clear=True):
            config = app.RuntimeConfig.from_env()
        self.assertEqual(config.retrieval_mode, "dense")
        self.assertIsNone(config.reranker_model)

    def test_runtime_config_accepts_model_settings(self):
        with patch.dict(
            os.environ,
            {
                "LLM_BASE_URL": "https://generic.example/v1",
                "LLM_MODEL": "generic-model",
            },
            clear=True,
        ):
            config = app.RuntimeConfig.from_env()
        self.assertEqual(config.llm_base_url, "https://generic.example/v1")
        self.assertEqual(config.llm_model, "generic-model")

    def test_runtime_config_document_routing_is_opt_in(self):
        with patch.dict(os.environ, {"SCI_RAG_DOCUMENT_ROUTING": "true"}, clear=True):
            config = app.RuntimeConfig.from_env()
        self.assertTrue(config.document_routing)

        with patch.dict(os.environ, {"SCI_RAG_DOCUMENT_ROUTING": "unexpected"}, clear=True):
            config = app.RuntimeConfig.from_env()
        self.assertFalse(config.document_routing)

    def test_runtime_config_query_decomposition_is_opt_in(self):
        with patch.dict(os.environ, {"SCI_RAG_QUERY_DECOMPOSITION": "true"}, clear=True):
            config = app.RuntimeConfig.from_env()
        self.assertTrue(config.query_decomposition)

        with patch.dict(os.environ, {"SCI_RAG_QUERY_DECOMPOSITION": "unexpected"}, clear=True):
            config = app.RuntimeConfig.from_env()
        self.assertFalse(config.query_decomposition)

    def test_runtime_config_parent_window_is_opt_in(self):
        with patch.dict(os.environ, {"SCI_RAG_PARENT_WINDOW": "true"}, clear=True):
            config = app.RuntimeConfig.from_env()
        self.assertTrue(config.parent_window)

        with patch.dict(os.environ, {"SCI_RAG_PARENT_WINDOW": "unexpected"}, clear=True):
            config = app.RuntimeConfig.from_env()
        self.assertFalse(config.parent_window)

    def test_runtime_config_spatial_figure_evidence_is_opt_in(self):
        with patch.dict(
            os.environ,
            {"SCI_RAG_SPATIAL_FIGURE_EVIDENCE": "true"},
            clear=True,
        ):
            config = app.RuntimeConfig.from_env()
        self.assertTrue(config.spatial_figure_evidence)

        with patch.dict(
            os.environ,
            {"SCI_RAG_SPATIAL_FIGURE_EVIDENCE": "unexpected"},
            clear=True,
        ):
            config = app.RuntimeConfig.from_env()
        self.assertFalse(config.spatial_figure_evidence)

    def test_runtime_config_answer_validation_is_opt_in(self):
        with patch.dict(os.environ, {"SCI_RAG_ANSWER_VALIDATION": "true"}, clear=True):
            config = app.RuntimeConfig.from_env()
        self.assertTrue(config.answer_validation)

        with patch.dict(os.environ, {"SCI_RAG_ANSWER_VALIDATION": "unexpected"}, clear=True):
            config = app.RuntimeConfig.from_env()
        self.assertFalse(config.answer_validation)

    def test_runtime_config_formula_evidence_is_opt_in(self):
        with patch.dict(os.environ, {"SCI_RAG_FORMULA_EVIDENCE": "1"}, clear=False):
            config = app.RuntimeConfig.from_env()
        self.assertTrue(config.formula_evidence)
        with patch.dict(os.environ, {"SCI_RAG_FORMULA_EVIDENCE": "0"}, clear=False):
            config = app.RuntimeConfig.from_env()
        self.assertFalse(config.formula_evidence)

    def test_runtime_config_formula_evidence_auto_defaults_on_and_is_disableable(self):
        with patch.dict(os.environ, {}, clear=True):
            config = app.RuntimeConfig.from_env()
        self.assertTrue(config.formula_evidence_auto)
        with patch.dict(
            os.environ,
            {"SCI_RAG_FORMULA_EVIDENCE_AUTO": "0"},
            clear=True,
        ):
            config = app.RuntimeConfig.from_env()
        self.assertFalse(config.formula_evidence_auto)

    def test_formula_evidence_enabled_combines_manual_and_narrow_auto_switches(self):
        question = "MgNO 多重网格平滑迭代开始时如何初始化状态，更新时使用什么量？"
        ordinary = "DrugR 的显式推理数据集包含多少个样本？"
        self.assertTrue(
            app.formula_evidence_enabled(
                question, app.RuntimeConfig(formula_evidence_auto=True)
            )
        )
        self.assertFalse(
            app.formula_evidence_enabled(
                ordinary, app.RuntimeConfig(formula_evidence_auto=True)
            )
        )
        self.assertFalse(
            app.formula_evidence_enabled(
                question, app.RuntimeConfig(formula_evidence_auto=False)
            )
        )
        self.assertTrue(
            app.formula_evidence_enabled(
                ordinary, app.RuntimeConfig(formula_evidence=True, formula_evidence_auto=False)
            )
        )

    def test_formula_evidence_auto_only_activates_explicit_formula_intent(self):
        formula_question = "MgNO 多重网格平滑迭代开始时如何初始化状态，更新时使用什么量？"
        domain_question = "MgNO 讨论的二维椭圆 PDE 定义在哪个区域，并考虑哪些边界条件？"
        plain_question = "MgNO 论文的主要贡献是什么？"
        auto = app.RuntimeConfig(formula_evidence_auto=True)
        disabled = app.RuntimeConfig(formula_evidence_auto=False)
        self.assertTrue(
            auto.formula_evidence_auto and is_formula_question(formula_question)
        )
        self.assertFalse(
            auto.formula_evidence_auto and is_formula_question(plain_question)
        )
        self.assertTrue(
            auto.formula_evidence_auto and is_formula_question(domain_question)
        )
        self.assertFalse(
            disabled.formula_evidence_auto and is_formula_question(formula_question)
        )

    def test_reinforcement_learning_section_alias_prefers_specific_rl_heading(self):
        question = (
            "DrugR 的强化学习阶段如何解决多目标训练中的目标主导（objective domination）"
            "与目标饥饿（starvation）问题？"
        )
        terms = app._section_query_terms(question)
        rl_score = app._header_match_score(
            "H3: **4.5 Self-balanced Multi-granular Reinforcement Learning**",
            terms,
        )
        generic_score = app._header_match_score(
            "H3: **4.7 Training Settings**",
            terms,
        )
        self.assertGreater(rl_score, generic_score)

    def test_composite_question_detects_shared_predicate_for_named_sources(self):
        self.assertTrue(app._is_composite_fact_question("TANQ 和 FigEx 分别报告了多大规模的数据集？"))
        self.assertTrue(app._is_composite_fact_question("请列出每个维度下的三个 rubric。"))
        self.assertIn("entries", app._section_query_terms("TANQ 和 FigEx 分别报告了多大规模的数据集？"))
        self.assertIn("columns", app._section_query_terms("答案表平均有多少行和列？"))

    def test_spatial_context_adds_page_level_quadrant(self):
        annotated = app._annotate_spatial_context("[x=70.1-71.0%; y=85.4-86.5%] O")
        self.assertIn("page-level region: bottom-right", annotated)

    def test_explicit_figure_query_promotes_exact_spatial_evidence(self):
        class Vector(list):
            def tolist(self):
                return list(self)

        class Embedding:
            def encode(self, _message):
                return Vector([0.1, 0.2])

        class Collection:
            def __init__(self):
                self.figure_where = None
                self.dense_where = None

            def count(self):
                return 2

            def query(self, **kwargs):
                self.dense_where = kwargs.get("where")
                return {
                    "ids": [["prose"]],
                    "documents": [["Nearby prose says n = 28."]],
                    "metadatas": [[{"source": "paper.pdf", "type": "text"}]],
                }

            def get(self, **kwargs):
                self.figure_where = kwargs.get("where")
                return {
                    "ids": ["figure-1"],
                    "documents": [
                        "Figure 1 spatial text evidence\n"
                        "[x=45.0-47.1%] RNA / n = 8\n"
                        "[x=57.6-60.2%] n = 28"
                    ],
                    "metadatas": [
                        {
                            "source": "paper.pdf",
                            "page": 2,
                            "type": "figure",
                            "figure_kind": "figure",
                            "figure_number": 1,
                            "figure_label": "Figure 1",
                        }
                    ],
                }

        class Client:
            def __init__(self):
                self.prompt = ""
                self.chat = self
                self.completions = self

            def create(self, **kwargs):
                self.prompt = kwargs["messages"][1]["content"]
                return type(
                    "Response",
                    (),
                    {
                        "choices": [
                            type(
                                "Choice",
                                (),
                                {
                                    "message": type(
                                        "Message", (), {"content": "CASP15 RNA 使用 n = 8。"}
                                    )()
                                },
                            )()
                        ]
                    },
                )()

        collection = Collection()
        client = Client()
        result = app.query_knowledge(
            "Figure 1 中 CASP15 RNA 的 n 是多少？",
            runtime=app.Runtime(
                app.RuntimeConfig(
                    spatial_figure_evidence=True,
                    retrieval_k=1,
                    context_k=1,
                ),
                client,
                Embedding(),
                collection,
            ),
        )

        self.assertEqual(result["context_ids"], ["figure-1"])
        self.assertEqual(result["context_metadatas"][0]["type"], "figure")
        self.assertIn("RNA / n = 8", client.prompt)
        self.assertIn("原点在左上角", client.prompt)
        self.assertIn("y 向下增加", client.prompt)
        self.assertIn("figure_number", str(collection.figure_where))
        self.assertIn("$ne", str(collection.dense_where))
        self.assertNotIn("【补充原文核对项】", result["answer"])

    def test_parent_window_enriches_prompt_without_adding_context_slot(self):
        class Vector(list):
            def tolist(self):
                return list(self)

        class Embedding:
            def encode(self, _message):
                return Vector([0.1, 0.2])

        class Collection:
            def count(self):
                return 3

            def query(self, **_kwargs):
                return {
                    "ids": [["anchor"]],
                    "documents": [["The finite-element system is A*u=f."]],
                    "metadatas": [[
                        {
                            "source": "paper.pdf",
                            "page": 4,
                            "type": "text",
                            "chunk_index": 1,
                        }
                    ]],
                }

            def get(self, **_kwargs):
                return {
                    "ids": ["detail", "anchor", "other-page"],
                    "documents": [
                        "The kernel has size 3×3.",
                        "The finite-element system is A*u=f.",
                        "Unrelated next page.",
                    ],
                    "metadatas": [
                        {"source": "paper.pdf", "page": 4, "type": "text", "chunk_index": 0},
                        {"source": "paper.pdf", "page": 4, "type": "text", "chunk_index": 1},
                        {"source": "paper.pdf", "page": 5, "type": "text", "chunk_index": 2},
                    ],
                }

        class Client:
            def __init__(self):
                self.prompt = ""
                self.chat = self
                self.completions = self

            def create(self, **kwargs):
                self.prompt = kwargs["messages"][1]["content"]
                return type(
                    "Response",
                    (),
                    {"choices": [type("Choice", (), {"message": type("Message", (), {"content": "ok"})()})()]},
                )()

        client = Client()
        result = app.query_knowledge(
            "What is the finite-element system equation and kernel size?",
            runtime=app.Runtime(
                app.RuntimeConfig(parent_window=True, retrieval_k=1, context_k=1),
                client,
                Embedding(),
                Collection(),
            ),
        )

        self.assertEqual(result["context_ids"], ["anchor"])
        self.assertEqual(len(result["contexts"]), 1)
        self.assertIn("3×3", result["contexts"][0])
        self.assertIn("3×3", client.prompt)
        self.assertEqual(
            result["context_metadatas"][0]["window_chunk_ids"],
            ["detail", "anchor"],
        )

    def test_parent_window_joins_cross_page_sentence_continuation(self):
        class Vector(list):
            def tolist(self):
                return list(self)

        class Embedding:
            def encode(self, _message):
                return Vector([0.1, 0.2])

        class Collection:
            def count(self):
                return 3

            def query(self, **_kwargs):
                return {
                    "ids": [["anchor"]],
                    "documents": [["Answer tables have 6.7 rows and"]],
                    "metadatas": [[
                        {"source": "paper.pdf", "page": 6, "type": "text", "chunk_index": 2}
                    ]],
                }

            def get(self, **_kwargs):
                return {
                    "ids": ["anchor", "continuation", "unrelated"],
                    "documents": [
                        "Answer tables have 6.7 rows and",
                        "4 columns.",
                        "An unrelated next-page paragraph.",
                    ],
                    "metadatas": [
                        {"source": "paper.pdf", "page": 6, "type": "text", "chunk_index": 2},
                        {"source": "paper.pdf", "page": 7, "type": "text", "chunk_index": 3},
                        {"source": "paper.pdf", "page": 7, "type": "text", "chunk_index": 4},
                    ],
                }

        class Client:
            def __init__(self):
                self.chat = self
                self.completions = self

            def create(self, **_kwargs):
                return type(
                    "Response",
                    (),
                    {"choices": [type("Choice", (), {"message": type("Message", (), {"content": "ok"})()})()]},
                )()

        result = app.query_knowledge(
            "How many rows and columns do the answer tables have?",
            runtime=app.Runtime(
                app.RuntimeConfig(parent_window=True, retrieval_k=1, context_k=1),
                Client(),
                Embedding(),
                Collection(),
            ),
        )

        self.assertIn("6.7 rows and\n\n4 columns.", result["contexts"][0])
        self.assertEqual(
            result["context_metadatas"][0]["window_chunk_ids"],
            ["anchor", "continuation"],
        )
        self.assertEqual(result["context_metadatas"][0]["window_pages"], [6, 7])

    def test_parent_window_joins_previous_page_when_anchor_is_continuation(self):
        class Collection:
            def count(self):
                return 2

            def get(self, **_kwargs):
                return {
                    "ids": ["start", "anchor"],
                    "documents": ["Answer tables have 6.7 rows and", "4 columns."],
                    "metadatas": [
                        {"source": "paper.pdf", "page": 6, "type": "text", "chunk_index": 2},
                        {"source": "paper.pdf", "page": 7, "type": "text", "chunk_index": 3},
                    ],
                }

        runtime = app.Runtime(app.RuntimeConfig(parent_window=True), None, None, Collection())
        contexts, metadatas = app._parent_window_contexts(
            ["4 columns."],
            ["anchor"],
            [{"source": "paper.pdf", "page": 7, "type": "text", "chunk_index": 3}],
            runtime,
        )
        self.assertEqual(contexts, ["Answer tables have 6.7 rows and\n\n4 columns."])
        self.assertEqual(metadatas[0]["window_chunk_ids"], ["start", "anchor"])
        self.assertEqual(metadatas[0]["window_pages"], [6, 7])

    def test_formula_evidence_promotes_same_source_formula_candidate(self):
        class Vector(list):
            def tolist(self):
                return list(self)

        class Embedding:
            def encode(self, _message):
                return Vector([0.1, 0.2])

        class Collection:
            def count(self):
                return 3

            def query(self, **_kwargs):
                return {
                    "ids": [["distractor"]],
                    "documents": [["The paper introduces a model."]],
                    "metadatas": [[{"source": "paper.pdf", "type": "text"}]],
                }

            def get(self, **_kwargs):
                return {
                    "ids": ["distractor", "formula", "other"],
                    "documents": [
                        "The paper introduces a model.",
                        "With linear FEM, the elliptic PDE system is A*u=f; the kernel has dimensions 3 × 3.",
                        "A general discussion without equations.",
                    ],
                    "metadatas": [
                        {"source": "paper.pdf", "type": "text"},
                        {"source": "paper.pdf", "type": "formula"},
                        {"source": "paper.pdf", "type": "text"},
                    ],
                }

        class Client:
            def __init__(self):
                self.prompt = ""
                self.chat = self
                self.completions = self

            def create(self, **kwargs):
                self.prompt = kwargs["messages"][1]["content"]
                return type(
                    "Response",
                    (),
                    {"choices": [type("Choice", (), {"message": type("Message", (), {"content": "ok"})()})()]},
                )()

        client = Client()
        result = app.query_knowledge(
            "线性有限元离散后的椭圆 PDE 系统写成什么形式，卷积核尺寸是多少？",
            runtime=app.Runtime(
                app.RuntimeConfig(formula_evidence_auto=True, retrieval_k=1, context_k=1),
                client,
                Embedding(),
                Collection(),
            ),
        )
        self.assertEqual(result["context_ids"], ["formula"])
        self.assertTrue(result["context_metadatas"][0]["formula_evidence"])
        self.assertIn("3 × 3", client.prompt)

    def test_composite_formula_variant_keeps_its_routed_source(self):
        class Vector(list):
            def tolist(self):
                return list(self)

        class Embedding:
            def encode(self, _message):
                return Vector([0.1, 0.2])

        class Collection:
            def count(self):
                return 4

            def query(self, **kwargs):
                where = str(kwargs.get("where", {}))
                if "2026.findings-eacl.12.pdf" in where:
                    return {
                        "ids": [["thinknote-text"]],
                        "documents": [["THINKNOTE knowledge assimilation evidence"]],
                        "metadatas": [[
                            {"source": "2026.findings-eacl.12.pdf", "type": "text"}
                        ]],
                    }
                return {
                    "ids": [["drugr-text"]],
                    "documents": [["DrugR explicit reasoning dataset evidence"]],
                    "metadatas": [[{"source": "2602.08213v1.pdf", "type": "text"}]],
                }

            def get(self, **_kwargs):
                return {
                    "ids": ["drugr-text", "thinknote-text", "thinknote-formula", "drugr-formula"],
                    "documents": [
                        "DrugR explicit reasoning dataset evidence",
                        "THINKNOTE knowledge assimilation evidence",
                        "(1) T = M (Ika, q, D)",
                        "(1) unrelated DrugR equation",
                    ],
                    "metadatas": [
                        {"source": "2602.08213v1.pdf", "type": "text"},
                        {"source": "2026.findings-eacl.12.pdf", "type": "text"},
                        {"source": "2026.findings-eacl.12.pdf", "type": "formula"},
                        {"source": "2602.08213v1.pdf", "type": "formula"},
                    ],
                }

        class Client:
            def __init__(self):
                self.prompt = ""
                self.chat = self
                self.completions = self

            def create(self, **kwargs):
                self.prompt = kwargs["messages"][1]["content"]
                return type(
                    "Response",
                    (),
                    {"choices": [type("Choice", (), {"message": type("Message", (), {"content": "ok"})()})()]},
                )()

        client = Client()
        result = app.query_knowledge(
            "DrugR 的显式推理数据集包含多少个样本？同时，THINKNOTE 的知识同化模块输出 T 的形式化定义是什么？",
            runtime=app.Runtime(
                app.RuntimeConfig(
                    document_routing=True,
                    query_decomposition=True,
                    retrieval_k=2,
                    context_k=3,
                ),
                client,
                Embedding(),
                Collection(),
            ),
        )
        self.assertIn("thinknote-formula", result["context_ids"])
        formula_index = result["context_ids"].index("thinknote-formula")
        self.assertTrue(result["context_metadatas"][formula_index]["formula_evidence"])
        self.assertIn("T = M (Ika, q, D)", client.prompt)

    def test_source_local_fallback_uses_decomposed_question_clause(self):
        class Vector(list):
            def tolist(self):
                return list(self)

        class Embedding:
            def encode(self, _message):
                return Vector([0.1, 0.2])

        class Collection:
            def count(self):
                return 1

            def query(self, **_kwargs):
                return {
                    "ids": [["dense"]],
                    "documents": [["dense evidence"]],
                    "metadatas": [[{"source": "paper.pdf", "type": "text"}]],
                }

            def get(self, **_kwargs):
                return {
                    "ids": ["dense"],
                    "documents": ["dense evidence"],
                    "metadatas": [{"source": "paper.pdf", "type": "text"}],
                }

        class Client:
            def __init__(self):
                self.chat = self
                self.completions = self

            def create(self, **_kwargs):
                return type(
                    "Response",
                    (),
                    {"choices": [type("Choice", (), {"message": type("Message", (), {"content": "ok"})()})()]},
                )()

        question = "SciDQA 的问题和答案主要从什么材料中产生，之后如何降低第一人称偏差？"
        seen: list[str] = []

        def lexical(question_text, _runtime, _route):
            seen.append(question_text)
            return None

        with patch.object(app, "_lexical_route_evidence_result", side_effect=lexical):
            app.query_knowledge(
                question,
                runtime=app.Runtime(
                    app.RuntimeConfig(query_decomposition=True, retrieval_k=1),
                    Client(),
                    Embedding(),
                    Collection(),
                ),
                source_filter="paper.pdf",
            )

        self.assertIn("SciDQA 的问题和答案主要从什么材料中产生", seen)
        self.assertIn("之后如何降低第一人称偏差", seen)

    def test_missing_named_identifier_promotes_one_matching_passage(self):
        lexical = {
            "ids": [["generic", "metric"]],
            "documents": [["General evaluation discussion", "M7Score uses token probabilities"]],
            "metadatas": [[{"source": "paper.pdf"}, {"source": "paper.pdf"}]],
        }
        with patch.object(app, "_lexical_route_evidence_result", return_value=lexical) as search:
            result = app._missing_identifier_result(
                "How does M7Score work?", ["General evaluation discussion"], None, "paper.pdf"
            )
            self.assertEqual(result["ids"], [["metric"]])
            self.assertIsNone(app._missing_identifier_result(
                "How does M7Score work?", ["M7Score overview"], None, "paper.pdf"
            ))
            search.assert_called_once()

    def test_hybrid_lexical_path_excludes_formula_chunk(self):
        class Vector(list):
            def tolist(self):
                return list(self)

        class Embedding:
            def encode(self, _message):
                return Vector([0.1, 0.2])

        class Collection:
            def count(self):
                return 2

            def query(self, **_kwargs):
                return {
                    "ids": [["normal"]],
                    "documents": [["ordinary evidence"]],
                    "metadatas": [[{"source": "paper.pdf", "type": "text"}]],
                }

            def get(self, **_kwargs):
                return {
                    "ids": ["normal", "formula"],
                    "documents": ["ordinary evidence", "A ∗ u = quantum"],
                    "metadatas": [
                        {"source": "paper.pdf", "type": "text"},
                        {"source": "paper.pdf", "type": "formula"},
                    ],
                }

        class Client:
            chat = completions = None

            def __init__(self):
                self.chat = self
                self.completions = self

            def create(self, **_kwargs):
                return type(
                    "Response",
                    (),
                    {"choices": [type("Choice", (), {"message": type("Message", (), {"content": "ok"})()})()]},
                )()

        result = app.query_knowledge(
            "What is quantum?",
            runtime=app.Runtime(
                app.RuntimeConfig(
                    retrieval_mode="hybrid",
                    hybrid_candidate_k=2,
                    context_k=1,
                    formula_evidence_auto=False,
                ),
                Client(),
                Embedding(),
                Collection(),
            ),
        )
        self.assertEqual(result["context_ids"], ["normal"])

    def test_parent_window_does_not_expand_picture_text_blocks(self):
        class Vector(list):
            def tolist(self):
                return list(self)

        class Embedding:
            def encode(self, _message):
                return Vector([0.1, 0.2])

        class Collection:
            def count(self):
                return 2

            def query(self, **_kwargs):
                return {
                    "ids": [["figure"]],
                    "documents": [[
                        "<!-- Start of picture text --> Figure 1 n=25 n=38 n=8"
                    ]],
                    "metadatas": [[
                        {"source": "paper.pdf", "page": 2, "chunk_index": 1, "type": "text"}
                    ]],
                }

            def get(self, **_kwargs):
                return {
                    "ids": ["figure", "neighbor"],
                    "documents": [
                        "<!-- Start of picture text --> Figure 1 n=25 n=38 n=8",
                        "Adjacent prose with unrelated sample counts.",
                    ],
                    "metadatas": [
                        {"source": "paper.pdf", "page": 2, "chunk_index": 1, "type": "text"},
                        {"source": "paper.pdf", "page": 2, "chunk_index": 2, "type": "text"},
                    ],
                }

        class Client:
            def __init__(self):
                self.prompt = ""
                self.chat = self
                self.completions = self

            def create(self, **kwargs):
                self.prompt = kwargs["messages"][1]["content"]
                return type(
                    "Response",
                    (),
                    {"choices": [type("Choice", (), {"message": type("Message", (), {"content": "ok"})()})()]},
                )()

        client = Client()
        result = app.query_knowledge(
            "How many structures are listed in Figure 1?",
            runtime=app.Runtime(
                app.RuntimeConfig(parent_window=True, retrieval_k=1, context_k=1),
                client,
                Embedding(),
                Collection(),
            ),
        )

        self.assertEqual(result["context_ids"], ["figure"])
        self.assertNotIn("Adjacent prose", result["contexts"][0])
        self.assertNotIn("window_chunk_ids", result["context_metadatas"][0])

    def test_inline_picture_markers_can_contain_nonvisual_evidence(self):
        class Vector(list):
            def tolist(self):
                return list(self)

        class Embedding:
            def encode(self, _message):
                return Vector([0.1, 0.2])

        text = (
            "The schema description follows this template.\n"
            '<!-- Start of picture text -->{"table_name": "<Table Name>", '
            '"columns": [["<ColName>", "<Type>", "<Examples>"]]}'
            "<!-- End of picture text -->"
        )

        class Collection:
            def count(self):
                return 1

            def query(self, **_kwargs):
                return {
                    "ids": [["one"]],
                    "documents": [[text]],
                    "metadatas": [[{"source": "paper.pdf", "type": "text", "page": 5}]],
                }

        class Client:
            def __init__(self):
                self.prompt = ""
                self.chat = self.completions = self

            def create(self, **kwargs):
                self.prompt = kwargs["messages"][1]["content"]
                return type(
                    "Response", (),
                    {"choices": [type("Choice", (), {"message": type("Message", (), {"content": "ok"})()})()]},
                )()

        client = Client()
        runtime = app.Runtime(
            app.RuntimeConfig(retrieval_k=1, context_k=1),
            client, Embedding(), Collection(),
        )
        result = app.query_knowledge("What fields does the schema template contain?", runtime=runtime)
        self.assertIn('"table_name"', result["contexts"][0])
        self.assertIn("<ColName>", result["contexts"][0])
        self.assertIn('"table_name"', client.prompt)
        self.assertIn("<ColName>", client.prompt)

    def test_runtime_rejects_reranker_outside_hybrid_mode(self):
        with self.assertRaises(ValueError):
            app.Runtime(app.RuntimeConfig(), None, None, None, reranker=object())

    def test_opt_in_reranker_changes_hybrid_order_and_prompt_context(self):
        class Vector(list):
            def tolist(self):
                return list(self)

        class Embedding:
            def encode(self, _message):
                return Vector([0.1, 0.2])

        class Collection:
            def count(self):
                return 3

            def query(self, **_kwargs):
                return {
                    "ids": [["weak", "strong"]],
                    "documents": [["Weak context.", "The answer is 4,855 samples."]],
                    "metadatas": [[{"type": "text"}, {"type": "text"}]],
                }

            def get(self, **_kwargs):
                return {
                    "ids": ["weak", "strong", "other"],
                    "documents": [
                        "Weak context.",
                        "The answer is 4,855 samples.",
                        "Other context.",
                    ],
                    "metadatas": [{"type": "text"}] * 3,
                }

        class Reranker:
            def __init__(self):
                self.calls = 0

            def rerank(self, _question, candidates, documents):
                self.calls += 1
                order = sorted(
                    candidates,
                    key=lambda item: "4,855" not in documents[int(item.key)],
                )
                ranked = [
                    RankedItem(item.key, float(len(order) - index))
                    for index, item in enumerate(order)
                ]
                return ranked

        class Client:
            def __init__(self):
                self.prompt = ""
                self.chat = self
                self.completions = self

            def create(self, **kwargs):
                self.prompt = kwargs["messages"][1]["content"]
                return type(
                    "Response",
                    (),
                    {"choices": [type("Choice", (), {"message": type("Message", (), {"content": "ok"})()})()]},
                )()

        reranker = Reranker()
        client = Client()
        runtime = app.Runtime(
            app.RuntimeConfig(
                retrieval_mode="hybrid", hybrid_candidate_k=3, context_k=2
            ),
            client,
            Embedding(),
            Collection(),
            reranker=reranker,
        )
        result = app.query_knowledge("How many samples?", runtime=runtime)

        self.assertEqual(reranker.calls, 1)
        self.assertIn("4,855", result["contexts"][0])
        self.assertTrue(all(context in client.prompt for context in result["contexts"]))

    def test_dense_narrative_quantity_question_does_not_scan_tables(self):
        class Vector(list):
            def tolist(self):
                return list(self)

        class Embedding:
            def encode(self, _message):
                return Vector([0.1, 0.2])

        class Collection:
            def count(self):
                return 1

            def query(self, **_kwargs):
                return {
                    "ids": [["dense"]],
                    "documents": [["The explicit-reasoning dataset contains 4,855 samples and uses a reverse-engineering pipeline."]],
                    "metadatas": [[{"type": "text"}]],
                }

            def get(self, **_kwargs):
                raise AssertionError("a narrative quantity question must not scan all tables")

        class Client:
            def __init__(self):
                self.prompt = ""
                self.chat = self
                self.completions = self

            def create(self, **kwargs):
                self.prompt = kwargs["messages"][1]["content"]
                return type(
                    "Response",
                    (),
                    {"choices": [type("Choice", (), {"message": type("Message", (), {"content": "ok"})()})()]},
                )()

        client = Client()
        runtime = app.Runtime(app.RuntimeConfig(), client, Embedding(), Collection())
        result = app.query_knowledge(
            "DrugR 的显式推理数据集包含多少个样本？推理标注是通过什么管道构建的？",
            runtime=runtime,
        )
        self.assertIn("4,855 samples", result["contexts"][0])
        self.assertIn("【事实核对清单】", client.prompt)
        self.assertIn("4,855 samples", client.prompt)
        self.assertIsNone(runtime._lexical_snapshot)

        validated = app.query_knowledge(
            "How many samples?",
            runtime=app.Runtime(
                app.RuntimeConfig(answer_validation=True),
                client,
                Embedding(),
                Collection(),
            ),
        )
        self.assertEqual(validated["answer_validation"]["status"], "review")
        self.assertIn("证据核对提示", validated["answer"])

    def test_hybrid_mode_adds_lexical_candidate_and_reuses_snapshot(self):
        class Vector(list):
            def tolist(self):
                return list(self)

        class Embedding:
            def encode(self, _message):
                return Vector([0.1, 0.2])

        class Collection:
            def __init__(self):
                self.get_calls = 0

            def count(self):
                return 3

            def query(self, **_kwargs):
                return {
                    "ids": [["dense"]],
                    "documents": [["Dense evidence"]],
                    "metadatas": [[{"type": "text"}]],
                }

            def get(self, **_kwargs):
                self.get_calls += 1
                return {
                    "ids": ["dense", "lexical", "other"],
                    "documents": [
                        "Dense evidence",
                        "Chlorophyll flux is controlled by the antenna complex.",
                        "Unrelated appendix.",
                    ],
                    "metadatas": [{"type": "text"}, {"type": "text"}, {"type": "text"}],
                }

        class Client:
            def __init__(self):
                self.prompt = ""
                self.chat = self
                self.completions = self

            def create(self, **kwargs):
                self.prompt = kwargs["messages"][1]["content"]
                return type(
                    "Response",
                    (),
                    {"choices": [type("Choice", (), {"message": type("Message", (), {"content": "ok"})()})()]},
                )()

        collection = Collection()
        config = app.RuntimeConfig(
            retrieval_mode="hybrid", hybrid_candidate_k=3, context_k=2
        )
        client = Client()
        runtime = app.Runtime(config, client, Embedding(), collection)
        first = app.query_knowledge("What controls chlorophyll flux?", runtime=runtime)
        second = app.query_knowledge("What controls chlorophyll flux?", runtime=runtime)
        self.assertTrue(any("Chlorophyll flux" in context for context in first["contexts"]))
        self.assertEqual(first["contexts"], second["contexts"])
        self.assertTrue(all(context in client.prompt for context in second["contexts"]))
        self.assertEqual(collection.get_calls, 1)

    def test_query_decomposition_is_opt_in_and_queries_each_variant(self):
        class Vector(list):
            def tolist(self):
                return list(self)

        class Embedding:
            def encode(self, _message):
                return Vector([0.1, 0.2])

        class Collection:
            def __init__(self):
                self.query_calls = 0

            def count(self):
                return 2

            def query(self, **_kwargs):
                self.query_calls += 1
                return {
                    "ids": [["first", "second"]],
                    "documents": [["First evidence.", "Second evidence."]],
                    "metadatas": [[{"type": "text"}, {"type": "text"}]],
                }

            def get(self, **_kwargs):
                return {
                    "ids": ["first", "second"],
                    "documents": ["First evidence.", "Second evidence."],
                    "metadatas": [{"type": "text"}, {"type": "text"}],
                }

        class Client:
            def __init__(self):
                self.chat = self
                self.completions = self

            def create(self, **_kwargs):
                return type(
                    "Response",
                    (),
                    {"choices": [type("Choice", (), {"message": type("Message", (), {"content": "ok"})()})()]},
                )()

        collection = Collection()
        runtime = app.Runtime(
            app.RuntimeConfig(query_decomposition=True, context_k=2),
            Client(),
            Embedding(),
            collection,
        )
        app.query_knowledge("第一项是什么，第二项是什么？", runtime=runtime)
        self.assertEqual(collection.query_calls, 3)

    def test_document_routing_filters_unique_source_and_expands_only_that_source(self):
        class Vector(list):
            def tolist(self):
                return list(self)

        class Embedding:
            def encode(self, _message):
                return Vector([0.1, 0.2])

        class Collection:
            def __init__(self):
                self.query_kwargs = None

            def count(self):
                return 3

            def query(self, **kwargs):
                self.query_kwargs = kwargs
                return {
                    "ids": [["drugr-overview"]],
                    "documents": [["DrugR dataset overview."]],
                    "metadatas": [[
                        {"source": "drugr.pdf", "headers": "H2: Dataset pipeline", "type": "text"}
                    ]],
                }

            def get(self, **_kwargs):
                return {
                    "ids": ["drugr-overview", "drugr-sibling", "af3-sibling"],
                    "documents": [
                        "DrugR dataset overview.",
                        "ADMETLab evaluates candidates in the DrugR pipeline.",
                        "AlphaFold3 unrelated pipeline.",
                    ],
                    "metadatas": [
                        {"source": "drugr.pdf", "headers": "H2: Dataset pipeline", "type": "text"},
                        {"source": "drugr.pdf", "headers": "H2: Dataset pipeline", "type": "text"},
                        {"source": "af3.pdf", "headers": "H2: Dataset pipeline", "type": "text"},
                    ],
                }

        class Client:
            def __init__(self):
                self.chat = self
                self.completions = self
                self.prompt = ""

            def create(self, **kwargs):
                self.prompt = kwargs["messages"][1]["content"]
                return type(
                    "Response",
                    (),
                    {"choices": [type("Choice", (), {"message": type("Message", (), {"content": "ok"})()})()]},
                )()

        collection = Collection()
        client = Client()
        runtime = app.Runtime(
            app.RuntimeConfig(
                retrieval_mode="hybrid",
                document_routing=True,
                hybrid_candidate_k=2,
                context_k=2,
            ),
            client,
            Embedding(),
            collection,
        )
        result = app.query_knowledge(
            "DrugR dataset pipeline",
            runtime=runtime,
        )

        self.assertEqual(
            collection.query_kwargs["where"],
            {"$and": [{"type": {"$ne": "formula"}}, {"source": {"$eq": "drugr.pdf"}}]},
        )
        self.assertTrue(all("AlphaFold3" not in context for context in result["contexts"]))
        self.assertTrue(
            all(metadata.get("source") == "drugr.pdf" for metadata in result["context_metadatas"])
        )
        self.assertIn("ADMETLab", "\n".join(result["contexts"]))
        self.assertNotIn("AlphaFold3", client.prompt)

    def test_explicit_source_filter_isolates_benchmark_contexts(self):
        class Vector(list):
            def tolist(self):
                return list(self)

        class Embedding:
            def encode(self, _message):
                return Vector([0.1, 0.2])

        class Collection:
            def __init__(self):
                self.query_kwargs = None

            def count(self):
                return 2

            def query(self, **kwargs):
                self.query_kwargs = kwargs
                return {
                    "ids": [["target", "other"]],
                    "documents": [["Target evidence.", "Other paper evidence."]],
                    "metadatas": [[
                        {"source": "target.pdf", "type": "text"},
                        {"source": "other.pdf", "type": "text"},
                    ]],
                }

            def get(self, **_kwargs):
                return {
                    "ids": ["target", "other"],
                    "documents": ["Target evidence.", "Other paper evidence."],
                    "metadatas": [
                        {"source": "target.pdf", "type": "text"},
                        {"source": "other.pdf", "type": "text"},
                    ],
                }

        class Client:
            def __init__(self):
                self.chat = self
                self.completions = self
                self.prompt = ""

            def create(self, **kwargs):
                self.prompt = kwargs["messages"][1]["content"]
                return type(
                    "Response",
                    (),
                    {"choices": [type("Choice", (), {"message": type("Message", (), {"content": "ok"})()})()]},
                )()

        collection = Collection()
        client = Client()
        result = app.query_knowledge(
            "What is the target evidence?",
            runtime=app.Runtime(
                app.RuntimeConfig(context_k=2),
                client,
                Embedding(),
                collection,
            ),
            source_filter="target.pdf",
        )

        self.assertEqual(
            collection.query_kwargs["where"],
            {"$and": [{"type": {"$ne": "formula"}}, {"source": {"$eq": "target.pdf"}}]},
        )
        self.assertEqual(result["context_metadatas"][0]["source"], "target.pdf")
        self.assertEqual(result["context_metadatas"][0]["type"], "text")
        self.assertNotIn("Other paper evidence", client.prompt)

    def test_source_filter_adds_source_local_lexical_evidence_without_router_token(self):
        class Vector(list):
            def tolist(self):
                return list(self)

        class Embedding:
            def encode(self, _message):
                return Vector([0.1, 0.2])

        class Collection:
            def count(self):
                return 3

            def query(self, **_kwargs):
                return {
                    "ids": [["anchor"]],
                    "documents": [["Paper overview."]],
                    "metadatas": [[
                        {"source": "target.pdf", "type": "text", "headers": "H2: Overview"}
                    ]],
                }

            def get(self, **_kwargs):
                return {
                    "ids": ["anchor", "evidence", "other"],
                    "documents": [
                        "Paper overview.",
                        "The answer uses at least 80% unigram overlap and 25% of answers exceed it.",
                        "Other paper evidence about unrelated results.",
                    ],
                    "metadatas": [
                        {"source": "target.pdf", "type": "text", "headers": "H2: Overview"},
                        {"source": "target.pdf", "type": "text", "headers": "H2: Fuzzy Search"},
                        {"source": "other.pdf", "type": "text", "headers": "H2: Results"},
                    ],
                }

        class Client:
            def __init__(self):
                self.chat = self
                self.completions = self

            def create(self, **_kwargs):
                return type(
                    "Response",
                    (),
                    {"choices": [type("Choice", (), {"message": type("Message", (), {"content": "ok"})()})()]},
                )()

        result = app.query_knowledge(
            "作者用什么阈值寻找答案与论文段落的重叠，多少比例的答案超过该重叠阈值？",
            runtime=app.Runtime(
                app.RuntimeConfig(retrieval_mode="hybrid", hybrid_candidate_k=1, context_k=2),
                Client(),
                Embedding(),
                Collection(),
            ),
            source_filter="target.pdf",
        )

        joined = "\n".join(result["contexts"])
        self.assertIn("80% unigram overlap", joined)
        self.assertEqual(result["context_ids"], ["evidence", "anchor"])
        self.assertTrue(all(meta.get("source") == "target.pdf" for meta in result["context_metadatas"]))

    def test_dense_first_hit_does_not_displace_source_local_method_evidence(self):
        from unittest.mock import Mock

        ids = ["filter", "rephrase", "experiment", "skills"]
        texts = [
            "Discard a sample when a single cell has more than five entries.",
            "Rephrase up to five times; stop when all relations are present.",
            "Performance decreases as the number of skills increases.",
            "Combine up to three distinct skills when constructing a question.",
        ]
        metas = [{"source": "paper.pdf", "type": "text"} for _ in ids]
        collection = Mock()
        collection.count.return_value = 5
        collection.query.return_value = {
            "ids": [["overview"]], "documents": [["Paper overview."]],
            "metadatas": [[{"source": "paper.pdf", "type": "text"}]],
        }
        collection.get.return_value = {
            "ids": ids, "documents": texts, "metadatas": metas,
        }
        embedding = Mock()
        embedding.encode.return_value.tolist.return_value = [1.0, 0.0]
        client = Mock()
        client.chat.completions.create.return_value.choices = [
            Mock(message=Mock(content="ok"), finish_reason="stop")
        ]
        runtime = app.Runtime(app.RuntimeConfig(context_k=4), client, embedding, collection)
        with patch.object(app, "_lexical_route_evidence_result", return_value={
            "ids": [ids], "documents": [texts], "metadatas": [metas],
        }), patch.object(app, "_section_expansion_result", return_value=None):
            result = app.query_knowledge(
                "筛选、改写与添加技能分别有什么条件？", runtime=runtime,
                source_filter="paper.pdf",
            )
        self.assertEqual(result["context_ids"], ids)
        prompt = client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
        self.assertIn(texts[3], prompt)

    def test_route_lexical_statistics_ignore_other_sources(self):
        from types import SimpleNamespace

        results = []
        for foreign_text in ("alpha " * 20, "beta " * 20):
            texts = ["alpha alpha", foreign_text, "beta beta", "alpha beta table"]
            snapshot = SimpleNamespace(
                ids=["alpha", "foreign", "beta", "table"], texts=texts,
                metadatas=[
                    {"source": source, "type": kind}
                    for source, kind in (("chosen", "text"), ("other", "text"),
                                         ("chosen", "text"), ("chosen", "table"))
                ],
                lexical_indices=list(range(4)), index=app.BM25Index(texts),
            )
            with patch.object(app, "_get_lexical_snapshot", return_value=snapshot):
                results.append(app._lexical_route_evidence_result(
                    "alpha beta", None, SimpleNamespace(document_id="chosen")
                ))
        self.assertEqual(results[0], results[1])
        self.assertEqual(results[0]["ids"][0], ["alpha", "beta"])

    def test_route_lexical_ignores_unmatched_followup_clause(self):
        from types import SimpleNamespace

        texts = ["Introduction.", "Unanswered ratio is defined here.",
                 "Acceptable ratio is defined here."]
        snapshot = SimpleNamespace(
            ids=["intro", "unanswered", "acceptable"], texts=texts,
            metadatas=[{"source": "paper.pdf", "type": "text"} for _ in texts],
            lexical_indices=list(range(len(texts))),
        )
        with patch.object(app, "_get_lexical_snapshot", return_value=snapshot):
            result = app._lexical_route_evidence_result(
                "unanswered ratio 和 acceptable ratio 如何定义？两者评估行为有何区别？",
                None, SimpleNamespace(document_id="paper.pdf"),
            )
        self.assertCountEqual(result["ids"][0][:2], ["acceptable", "unanswered"])

    def test_route_lexical_follows_same_source_caption_without_adjacent_method(self):
        from types import SimpleNamespace

        snapshot = SimpleNamespace(
            ids=["method", "unrelated", "detail", "foreign", "caption"],
            texts=["Retrieval uses BM25; see Figure 7.", "A different method.",
                   "Retrieval details.", "Figure 7: incorrect source.",
                   "Figure 7: The top-3 passages are supplied to the model."],
            metadatas=[{"source": source} for source in ("a", "a", "a", "b", "a")],
            lexical_indices=list(range(5)),
            index=SimpleNamespace(retrieve=lambda *args, **kwargs: [
                SimpleNamespace(key=0), SimpleNamespace(key=2), SimpleNamespace(key=1)
            ]),
        )
        with patch.object(app, "_get_lexical_snapshot", return_value=snapshot), \
                patch.object(app, "BM25Index", return_value=snapshot.index):
            result = app._lexical_route_evidence_result(
                "如何切分和检索？", None, SimpleNamespace(document_id="a")
            )
            self.assertEqual(result["ids"][0], ["method", "caption", "detail", "unrelated"])
            snapshot.texts[0] = "Retrieval uses BM25 without a figure reference."
            result = app._lexical_route_evidence_result(
                "如何切分和检索？", None, SimpleNamespace(document_id="a")
            )
            self.assertEqual(result["ids"][0], ["method", "detail", "unrelated"])

    def test_route_lexical_keeps_cross_page_sentence_before_footnote_caption(self):
        from types import SimpleNamespace

        texts = [
            "Filter samples with more than five entries.<sup>4</sup>\n"
            "Question rephrasing starts similarly to\n4See Figure 4 for the prompt.\n465",
            "|Count|23|", "the earlier step. Run up to 5 iterations; stop when all relations are present.",
            "Step 5. Combine up to three distinct skills.", "Figure 4: Evidence evaluation prompt.",
            "the wrong source's continuation.",
        ]
        metas = [
            {"source": "paper.pdf", "type": kind, "chunk_index": index, "page": page}
            for index, (kind, page) in enumerate([
                ("text", 5), ("table", 5), ("text", 6), ("text", 6), ("text", 18),
            ])
        ] + [{"source": "other.pdf", "type": "text", "chunk_index": 1, "page": 6}]
        snapshot = SimpleNamespace(
            ids=[str(i) for i in range(6)], texts=texts, metadatas=metas,
            lexical_indices=[0, 2, 3, 4, 5],
            index=SimpleNamespace(retrieve=lambda *args, **kwargs: [
                SimpleNamespace(key=0), SimpleNamespace(key=2), SimpleNamespace(key=3),
            ]),
        )
        with patch.object(app, "_get_lexical_snapshot", return_value=snapshot), \
                patch.object(app, "BM25Index", return_value=snapshot.index):
            result = app._lexical_route_evidence_result(
                "筛选、改写和技能分别如何限制？", None, SimpleNamespace(document_id="paper.pdf")
            )
            self.assertEqual(result["ids"][0][:3], ["0", "2", "3"])
            self.assertEqual(result["documents"][0][:3], [texts[0], texts[2], texts[3]])
            self.assertNotIn("5", result["ids"][0])
            for change in ({"page": 7}, {"headers": "H2: Next section"}, {"chunk_index": 8}):
                with self.subTest(change=change):
                    snapshot.metadatas[2] = {**metas[2], **change}
                    result = app._lexical_route_evidence_result(
                        "筛选、改写和技能分别如何限制？", None, SimpleNamespace(document_id="paper.pdf")
                    )
                    self.assertNotIn("2", result["ids"][0])
                    snapshot.metadatas[2] = {"source": "paper.pdf", "type": "text", "chunk_index": 2, "page": 6}
            snapshot.texts[2] = "Step 5. A separate method starts here."
            result = app._lexical_route_evidence_result(
                "筛选、改写和技能分别如何限制？", None, SimpleNamespace(document_id="paper.pdf")
            )
            self.assertEqual(result["ids"][0][:2], ["0", "4"])
            self.assertNotIn("2", result["ids"][0])
        self.assertFalse(app._is_cross_page_continuation("The count is 465", "the earlier step."))
        self.assertFalse(app._is_cross_page_continuation("A complete sentence.\n465", "the earlier step."))

    def test_route_lexical_method_question_uses_prose_not_version_table(self):
        from types import SimpleNamespace

        texts = [
            "Method Alpha (AnsInt) Method Beta (CxtInt) results.",
            "Method Alpha (AnsInt) Method Beta (CxtInt) results.",
            "The AnsInt variant integrates answers; the CxtInt variant integrates contexts.",
        ]
        snapshot = SimpleNamespace(
            ids=["table1", "table2", "method"], texts=texts,
            metadatas=[
                {"source": "paper.pdf", "type": kind}
                for kind in ("table", "table", "text")
            ],
            lexical_indices=list(range(3)), index=app.BM25Index(texts),
        )
        with patch.object(app, "_get_lexical_snapshot", return_value=snapshot):
            result = app._lexical_route_evidence_result(
                "AnsInt 与 CxtInt 分别整合什么？", None,
                SimpleNamespace(document_id="paper.pdf"),
            )
        self.assertEqual(result["ids"][0], ["method"])

    def test_route_lexical_classification_keeps_contiguous_list(self):
        from types import SimpleNamespace

        texts = [
            "General hallucination statistics.",
            "Hallucination Taxonomy: four types. Evident conflict.",
            "Subtle conflict.",
            "Evident baseless and subtle baseless information.",
            "Response generation starts here.",
        ]
        headers = [
            "H2: Statistics", "H2: Construction > H3: Taxonomy",
            "H2: Construction > H3: Taxonomy",
            "H2: Construction > H3: Evident Baseless",
            "H2: Construction > H3: 3.2 Response Generation",
        ]
        snapshot = SimpleNamespace(
            ids=[str(i) for i in range(5)], texts=texts,
            metadatas=[
                {"source": "paper.pdf", "type": "text", "chunk_index": i,
                 "headers": header}
                for i, header in enumerate(headers)
            ],
            lexical_indices=list(range(5)), index=app.BM25Index(texts),
        )
        with patch.object(app, "_get_lexical_snapshot", return_value=snapshot):
            result = app._lexical_route_evidence_result(
                "幻觉分为哪几类？", None,
                SimpleNamespace(document_id="paper.pdf"),
            )
        self.assertEqual(result["ids"][0][:3], ["1", "2", "3"])
        self.assertNotIn("4", result["ids"][0][:3])

    def test_list_question_keeps_a_cross_page_list_after_a_colon(self):
        from types import SimpleNamespace

        texts = [
            "The paper defines six categories of unanswerable requests.",
            "It rejects six categories of unanswerable requests:",
            "_Underspecified_, _False-presuppositions_, _Nonsensical_, _Modality-limited_, _Safety Concerns_, and _Out-of-Database_.",
            "The conclusion compares evaluation metrics.",
        ]
        headers = ["H2: Taxonomy", "H2: Introduction", "", "H2: Conclusion"]
        snapshot = SimpleNamespace(
            ids=[str(index) for index in range(4)], texts=texts,
            metadatas=[
                {"source": "paper.pdf", "type": "text", "chunk_index": index,
                 "page": 1 if index < 2 else 2, "headers": headers[index]}
                for index in range(4)
            ],
            lexical_indices=list(range(4)),
            index=SimpleNamespace(retrieve=lambda *args, **kwargs: [
                SimpleNamespace(key=0), SimpleNamespace(key=1), SimpleNamespace(key=3),
            ]),
        )
        with patch.object(app, "_get_lexical_snapshot", return_value=snapshot), \
                patch.object(app, "BM25Index", return_value=snapshot.index):
            result = app._lexical_route_evidence_result(
                "不可回答请求分为哪六类？", None, SimpleNamespace(document_id="paper.pdf")
            )
        self.assertEqual(result["ids"][0][:2], ["1", "2"])

    def test_route_lexical_causal_and_list_questions_keep_section_continuation(self):
        from types import SimpleNamespace

        texts = ["Generic overview.", "Method explains the first step.",
                 "|count|2|", "The explanation finishes after the table.",
                 "Next section."]
        metas = [
            {"source": "paper.pdf", "type": kind, "chunk_index": index,
             **({"headers": header} if header else {})}
            for index, (kind, header) in enumerate([
                ("text", ""), ("text", "H2: Ablation Method"),
                ("table", "|count|"), ("text", ""), ("text", "H2: Next"),
            ])
        ]
        snapshot = SimpleNamespace(
            ids=[str(index) for index in range(5)], texts=texts,
            metadatas=metas, lexical_indices=[0, 1, 3, 4],
            index=SimpleNamespace(retrieve=lambda *args, **kwargs: [
                SimpleNamespace(key=0), SimpleNamespace(key=1), SimpleNamespace(key=2),
            ]),
        )
        with patch.object(app, "_get_lexical_snapshot", return_value=snapshot), \
                patch.object(app, "BM25Index", return_value=snapshot.index):
            result = app._lexical_route_evidence_result(
                "为什么这样做？", None, SimpleNamespace(document_id="paper.pdf")
            )
            listed = app._lexical_route_evidence_result(
                "消融版本请列出每个版本的做法。", None,
                SimpleNamespace(document_id="paper.pdf"),
            )
        self.assertEqual(result["ids"][0][:3], ["1", "3", "0"])
        self.assertEqual(listed["ids"][0][:3], ["1", "3", "0"])
        self.assertIn("ablation", app._section_query_terms("消融版本"))
        self.assertNotIn("2", result["ids"][0])

    def test_english_list_uses_first_lexical_hit_and_its_continuation(self):
        from types import SimpleNamespace

        texts = [
            "Overview.", "Evaluation rubrics begin here.",
            "The nine rubrics are grouped into three dimensions.",
            "Cohesion, Conciseness and Readability are the first three.",
            "Related work compares other evaluation rubrics.",
        ]
        snapshot = SimpleNamespace(
            ids=[str(index) for index in range(5)], texts=texts,
            metadatas=[
                {"source": "paper.pdf", "type": "text", "chunk_index": index,
                 **({"headers": header} if header else {})}
                for index, header in enumerate([
                    "H2: Overview", "H2: Evaluation Rubrics", "", "",
                    "H2: Related Work > H3: Evaluation Rubrics",
                ])
            ],
            lexical_indices=list(range(5)),
            index=SimpleNamespace(retrieve=lambda *args, **kwargs: [
                SimpleNamespace(key=2), SimpleNamespace(key=4),
                SimpleNamespace(key=1),
            ]),
        )
        question = "Which three rubrics belong to each dimension?"
        self.assertTrue(app._source_local_evidence_requested(question))
        with patch.object(app, "_get_lexical_snapshot", return_value=snapshot), \
                patch.object(app, "BM25Index", return_value=snapshot.index):
            result = app._lexical_route_evidence_result(
                question, None, SimpleNamespace(document_id="paper.pdf")
            )
        self.assertEqual(result["ids"][0][:3], ["2", "3", "4"])

    def test_chinese_retrieval_terms_find_english_evidence_in_selected_source(self):
        from types import SimpleNamespace

        texts = ["An unrelated overview.", "Candidate sentences are selected.",
                 "Reranking orders the passages.", "Seed sentences start the search.",
                 "Expansion will stop when evidence is sufficient.",
                 "Samples with excessive cell entries are filtered out.",
                 "Question rephrasing preserves relations.",
                 "We combine up to three distinct skills for augmentation.",
                 "Candidate sentences, reranking, seed sentences, expansion stop, filtered entries, rephrasing and skills."]
        snapshot = SimpleNamespace(
            ids=["overview", "candidates", "rerank", "seed", "stop", "filter", "rephrase", "skills", "foreign"],
            texts=texts,
            metadatas=[{"source": "other" if i == 8 else "chosen"} for i in range(9)],
            lexical_indices=list(range(9)), index=app.BM25Index(texts),
        )
        with patch.object(app, "_get_lexical_snapshot", return_value=snapshot):
            for question, expected in (("候选句如何选？", "candidates"),
                                       ("如何重排？", "rerank"),
                                       ("种子怎么选择？", "seed"),
                                       ("扩展何时结束？", "stop"),
                                       ("筛选步骤是什么？", "filter"),
                                       ("如何改写？", "rephrase"),
                                       ("最多组合几种技能？", "skills"),
                                       ("How does filtering work?", "filter"),
                                       ("Which samples are not kept during filtering?", "filter"),
                                       ("How does rephrasing work?", "rephrase"),
                                       ("How many skills can be combined?", "skills")):
                with self.subTest(question=question):
                    self.assertTrue(app._source_local_evidence_requested(question))
                    result = app._lexical_route_evidence_result(
                        question, None, SimpleNamespace(document_id="chosen")
                    )
                    self.assertEqual(result["ids"][0][0], expected)
                    self.assertNotIn("foreign", result["ids"][0])
        terms = app._section_query_terms("单元格条目有多少，筛选、改写和技能有哪些步骤？")
        self.assertTrue({"entries", "filtered", "rephrasing", "skills", "number", "count"} <= terms)
        self.assertFalse({"dataset", "size", "statistics"} & terms)

    def test_bias_question_reaches_english_evidence_in_selected_source(self):
        from types import SimpleNamespace

        texts = [
            "The model led the ranking.",
            "The question-filtering process may introduce a potential bias in evaluation.",
            "An unrelated bias in another paper.",
        ]
        snapshot = SimpleNamespace(
            ids=["ranking", "bias", "other"], texts=texts,
            metadatas=[{"source": source, "type": "text"} for source in ("chosen", "chosen", "other")],
            lexical_indices=[0, 1, 2], index=app.BM25Index(texts),
        )
        question = "结果领先为什么仍可能存在偏差？"
        self.assertTrue(app._source_local_evidence_requested(question))
        with patch.object(app, "_get_lexical_snapshot", return_value=snapshot):
            result = app._lexical_route_evidence_result(
                question, None, SimpleNamespace(document_id="chosen")
            )
        self.assertEqual(result["ids"][0][0], "bias")
        self.assertNotIn("other", result["ids"][0])

    def test_two_questions_keep_evidence_for_each_clause(self):
        from types import SimpleNamespace

        texts = [
            "Task-level ablation of TQA, TFV and FF-TQA reward training.",
            "Format ablation tests response layout and reward supervision.",
            "Accuracy rewards: TQA exact match, TFV label match, FF-TQA BLEU and ROUGE-L.",
            "A format reward requires the template <think>...</think> <answer>{\"answer\": ...}</answer>.",
            "Prompt template for TFV answers.",
            "TFV format reward ablation compares template and output structure.",
        ]
        snapshot = SimpleNamespace(
            ids=["ablation-task", "ablation-format", "accuracy", "format", "prompt", "ablation-other"],
            texts=texts,
            metadatas=[{"source": "chosen", "type": "text"} for _ in texts],
            lexical_indices=list(range(len(texts))),
        )
        question = "TQA、TFV 和 FF-TQA 分别使用什么准确性奖励？格式奖励要求输出什么结构？"
        with patch.object(app, "_get_lexical_snapshot", return_value=snapshot):
            result = app._lexical_route_evidence_result(
                question, None, SimpleNamespace(document_id="chosen")
            )
        self.assertTrue({"accuracy", "format"} <= set(result["ids"][0][:4]))

    def test_shared_dense_clause_evidence_survives_source_local_fallback(self):
        from types import SimpleNamespace

        question = "图像和文本的总体差异是什么？四种格式的最大差距是多少？"
        clauses = [part for part in question.split("？") if part]
        names = ["first", "second", "shared", "fourth", "other"]
        documents = {name: f"Evidence {name}." for name in names}
        metadatas = {name: {"source": "paper.pdf", "type": "text"} for name in names}

        class Embedding:
            def encode(self, value):
                return SimpleNamespace(tolist=lambda: [value])

        class Collection:
            def __init__(self, overlap=True):
                self.overlap = overlap

            def count(self):
                return 5

            def query(self, **kwargs):
                value = kwargs["query_embeddings"][0][0]
                ids = (
                    ["first", "second", "shared", "fourth"] if value == question else
                    ["first", "shared", "fourth"] if value == clauses[0] else
                    ["other", "shared", "second"] if self.overlap else ["other", "second"]
                )[:kwargs["n_results"]]
                return {
                    "ids": [ids],
                    "documents": [[documents[name] for name in ids]],
                    "metadatas": [[metadatas[name] for name in ids]],
                }

            def get(self, **_kwargs):
                return {"ids": [], "documents": [], "metadatas": []}

        lexical = {
            "ids": [["first", "second", "fourth"]],
            "documents": [[documents[name] for name in ("first", "second", "fourth")]],
            "metadatas": [[metadatas[name] for name in ("first", "second", "fourth")]],
        }
        runtime = app.Runtime(
            app.RuntimeConfig(context_k=3), None, Embedding(), Collection()
        )
        with (
            patch.object(app, "_numeric_route_evidence_result", return_value=None),
            patch.object(app, "_lexical_route_evidence_result", return_value=lexical),
            patch.object(app, "_section_expansion_result", return_value=None),
            patch.object(app, "_missing_identifier_result", return_value=None),
        ):
            result = app.query_knowledge(question, runtime, source_filter=["paper.pdf"])
            runtime.collection = Collection(overlap=False)
            no_overlap = app.query_knowledge(question, runtime, source_filter=["paper.pdf"])
        self.assertEqual(result["context_ids"], ["shared", "first", "second"])
        self.assertEqual(no_overlap["context_ids"], ["first", "second", "fourth"])

    def test_blind_test_question_uses_english_filtering_terms(self):
        question = "盲测后如何检查错误答案？"
        self.assertTrue(app._source_local_evidence_requested(question))
        self.assertTrue(
            {"blind test", "incorrect", "ground-truth"}
            <= app._section_query_terms(question)
        )
        self.assertIn("filtered", app._section_query_terms("How does filtering work?"))
        self.assertTrue({"answer", "table", "cell", "entry"} <= app._section_query_terms(
            "During answer-table filtering, what cell-entry threshold keeps a sample?"
        ))
        self.assertTrue({"context", "length"} <= app._section_query_terms(
            "Which context-length cutoff excludes oracle models?"
        ))
        self.assertNotIn("qwen", app._section_query_terms("Qwen-2.5-72b"))
        self.assertTrue({"triple", "triples"} <= app._section_query_terms(
            "每个扩展三元组分别对应答案表中的什么？"
        ))
        self.assertTrue({"chunk", "chunked", "tokens", "iteration", "iterations", "iterative"} <= app._section_query_terms(
            "文本块长度和在线推理迭代次数分别是多少？"
        ))
        self.assertTrue(app._source_local_evidence_requested(
            "TANQ 的问题改写是否必须做满五次才停止？如果第一次改写已经包含所有关系，该如何处理？"
        ))

    def test_acronym_suffix_matches_split_pdf_emphasis(self):
        terms = app._section_query_terms("LLMgen 和 LLMeval 两项任务分别做什么？")
        self.assertTrue({"llmgen", "llmeval", "llm", "gen", "eval"} <= terms)

    def test_named_ratio_requests_local_evidence_without_matching_generation(self):
        self.assertTrue(app._source_local_evidence_requested(
            "unanswered ratio 和 acceptable ratio 分别如何定义？"
        ))
        self.assertFalse(app._source_local_evidence_requested("How is text generation implemented?"))

    def test_scientific_prompt_preserves_in_scope_safety_rules(self):
        ordinary = app._scientific_system_prompt("比较两篇论文的评测目标", [])
        self.assertNotIn("【强制规则 4：", ordinary)
        self.assertNotIn("【强制规则 6：", ordinary)
        for rule in ("数值必须原样引用", "图形坐标文字不得跨视觉组拼接", "严禁编造"):
            self.assertIn(rule, ordinary)
        self.assertNotIn("资料未提供该趋势的明确依据", ordinary)
        self.assertIn("严禁跨表取数", app._scientific_system_prompt("Table 2 中的得分？", []))
        self.assertIn("严禁跨表取数", app._scientific_system_prompt("哪些方法得分较高？", [{"type": "table"}]))
        self.assertIn("公式与符号必须按证据转录", app._scientific_system_prompt("公式是什么？", []))
        self.assertEqual(app._scientific_system_prompt("解释这些证据", [{"type": "table"}, {"type": "formula"}]),
                         app.SCIENTIFIC_SYSTEM_PROMPT)

    def test_answer_table_filtering_does_not_invent_statistics_intent(self):
        from types import SimpleNamespace

        question = "TANQ 在筛选时会丢弃什么情况的答案表？"
        self.assertTrue(app._source_local_evidence_requested(question))
        terms = app._section_query_terms(question)
        self.assertTrue({"answer", "table", "filtered"} <= terms)
        self.assertFalse({"rows", "columns", "statistics"} & terms)
        self.assertTrue({"rows", "columns", "statistics", "average"} <=
                        app._section_query_terms("答案表平均有多少行和多少列？"))
        texts = [
            "Dataset statistics: 6.7 rows and 4 columns in tables.",
            "Samples with more than five entries in any single cell of the answer table are filtered out.",
            "Samples with more than five entries in any single cell of the answer table are filtered out.",
        ]
        snapshot = SimpleNamespace(
            ids=["statistics", "filter", "foreign"], texts=texts,
            metadatas=[{"source": source, "type": "text"} for source in ("chosen", "chosen", "other")],
            lexical_indices=[0, 1, 2], index=app.BM25Index(texts),
        )
        with patch.object(app, "_get_lexical_snapshot", return_value=snapshot):
            result = app._lexical_route_evidence_result(
                question, None, SimpleNamespace(document_id="chosen")
            )
        self.assertEqual(result["ids"][0][0], "filter")
        self.assertNotIn("foreign", result["ids"][0])

    def test_section_aliases_cover_source_local_method_terms(self):
        for question in (
            "实体筛选和相似句的设置是什么？",
            "多文档与图像的实验边界是什么？",
            "幻觉分为哪几类？",
            "两个版本分别整合什么？",
        ):
            self.assertTrue(app._source_local_evidence_requested(question))
        self.assertIn("multimodal", app._section_query_terms("图像像素"))
        terms = app._section_query_terms(
            "作者用什么阈值寻找答案与论文段落的重叠，多少比例的答案超过该重叠阈值？"
        )
        for term in ("threshold", "overlap", "unigram"):
            self.assertIn(term, terms)
        retrieval_terms = app._section_query_terms("RAG 如何切分和检索论文？")
        for term in ("BM25", "retrieval", "ranker"):
            self.assertIn(term, retrieval_terms)
        perspective_terms = app._section_query_terms("如何降低第一人称偏差？")
        for term in ("third-person", "rewrite"):
            self.assertIn(term, perspective_terms)
        self.assertTrue(app._source_local_evidence_requested("RAG 如何切分和检索论文？"))
        self.assertFalse(app._source_local_evidence_requested("同行评审实验使用多少篇论文？"))
        annotation_terms = app._section_query_terms("人工标注审阅了多少候选实例，一致率是多少？")
        for term in ("agreement", "instances", "annotators"):
            self.assertIn(term, annotation_terms)
        architecture_terms = app._section_query_terms(
            "AlphaFold 3 相比 AlphaFold 2 在主干结构和结构模块上做了什么替换？"
        )
        for term in ("trunk", "structure module", "replace"):
            self.assertIn(term, architecture_terms)
        efficiency_terms = app._section_query_terms(
            "SciDC 的效率代价包括 token 长度、再生成次数和实际时间成本吗？"
        )
        for term in ("efficiency", "tokens", "regeneration", "time", "cost"):
            self.assertIn(term, efficiency_terms)
        self.assertTrue(app._source_local_evidence_requested("效率代价是多少？"))
        annotation_terms = app._section_query_terms(
            "问题标注使用什么代理模型？每个表格集合生成多少个问题？"
        )
        for term in ("question annotation", "agent annotator", "table set", "generation"):
            self.assertIn(term, annotation_terms)
        statistics_terms = app._section_query_terms("四类问题的数量和占比分别是多少？")
        for term in ("count", "percentage", "distribution"):
            self.assertIn(term, statistics_terms)
        self.assertTrue(app._source_local_evidence_requested("四类问题的数量和占比分别是多少？"))
        self.assertIn("embedding", app._section_query_terms("PDF 的嵌入模型和索引是什么？"))
        self.assertTrue(app._source_local_evidence_requested("实验如何量化不同设置的规模分布？"))
        construction_terms = app._section_query_terms(
            "如何构造多表集合，使用哪些来源或线索？"
        )
        for term in ("construction", "collection", "source", "cue", "metadata", "headers"):
            self.assertIn(term, construction_terms)
        revision_terms = app._section_query_terms(
            "最终数据集如何由 Correct 和修订后的问题组成？"
        )
        for term in ("revised", "refined", "edit"):
            self.assertIn(term, revision_terms)
        subset_terms = app._section_query_terms(
            "最终包含多少个数据子集、下游任务和实例？"
        )
        for term in ("data subsets", "downstream tasks", "instances"):
            self.assertIn(term, subset_terms)
        self.assertTrue(app._source_local_evidence_requested("修订后的数据集如何构成？"))
        self.assertTrue(app._source_local_evidence_requested("数据子集和下游任务分别是什么？"))
        option_terms = app._section_query_terms("每题有几个选项？")
        for term in ("answer choices", "options", "choices"):
            self.assertIn(term, option_terms)
        self.assertTrue(app._source_local_evidence_requested("每题有几个选项？"))
        quality_terms = app._section_query_terms("问题和答案如何生成与人工质检？")
        for term in ("quality", "quality control", "experts", "kappa"):
            self.assertIn(term, quality_terms)
        self.assertTrue(app._source_local_evidence_requested("问题和答案如何生成与人工质检？"))

    def test_multiple_dataset_rows_keep_qualifier_list_and_prompt_group(self):
        content = """Table 2: Accuracy (%)
|Prompting|Dataset|GPT-4|
|---|---|---|
|X→Y|MedQA-4|78.63|
||Medbullets-5|60.71|
||JAMA|67.32|
|X→RY|MedQA-4|82.64|
||Medbullets-5|63.31|
||JAMA|67.13|"""
        question = "Table 2 的零样本 X→Y 结果中，GPT-4 在 MedQA-4、Medbullets-5 和 JAMA 上的准确率分别是多少？"
        self.assertEqual(
            core._question_relation_qualifiers(question),
            ["medqa-4", "medbullets-5", "jama"],
        )
        self.assertEqual(
            core._table_question_section(question, content),
            "column:x→y",
        )
        values = extract_table_row_values(question, content, {"table_number": 2})
        self.assertEqual(
            [(row["row"], row["values"][-1]["value"]) for row in values["rows"]],
            [("MedQA-4", "78.63"), ("Medbullets-5", "60.71"), ("JAMA", "67.32")],
        )

    def test_horizontal_qualifiers_do_not_filter_dataset_rows(self):
        content = """|Setting|Dataset|CMB|MMCU|CMB-Clin|
|---|---|---|---|---|
|TC-RAG|MMCU-Medical|80.1|78.2|76.4|"""
        question = "TC-RAG 在 CMB、MMCU 和 CMB-Clin 上的主要评价指标分别是什么？"
        values = extract_table_row_values(question, content, {"table_number": 2})
        self.assertIsNotNone(values)
        self.assertEqual(values["row"], "TC-RAG")

    def test_source_local_aliases_cover_metrics_and_acceleration(self):
        terms = app._section_query_terms(
            "实验使用了哪些数据集和主要评价指标？两种加速策略是什么？"
        )
        for term in ("metrics", "accuracy", "acceleration", "speculative"):
            self.assertIn(term, terms)
        self.assertTrue(
            app._source_local_evidence_requested(
                "实验使用了哪些数据集和主要评价指标？两种加速策略是什么？"
            )
        )

    def test_nested_table_labels_and_chinese_split_aliases_keep_requested_rows(self):
        content = (
            "|Domain|Task|# Questions|Metric|Modality|\n"
            "|---|---|---|---|---|\n"
            "||Biology Chart QA|199|Accuracy|Chart|\n"
            "||OLED Property Extraction|13|Recall|Mol., Table|"
        )
        question = "Table 1 中 Biology Chart QA 和 OLED Property Extraction 各有多少道题？"
        values = extract_table_row_values(question, content, {"table_number": 1})
        self.assertEqual(
            [(row["row"], row["values"][1]["value"]) for row in values["rows"]],
            [("Biology Chart QA", "199"), ("OLED Property Extraction", "13")],
        )

        split_content = (
            "| |BioASQ|ORKGSynthesis|\n"
            "|---|---|---|\n"
            "|LLMgen Train|51|234|\n"
            "|LLMgen Test|22|105|\n"
            "|LLMeval Test Set|2,376|11,340|"
        )
        split_question = (
            "Table 2 中，BioASQ 和 ORKGSynthesis 的 LLMgen 训练/测试规模，"
            "以及 LLMeval Test Set 规模分别是多少？"
        )
        split_values = extract_table_row_values(
            split_question, split_content, {"table_number": 2}
        )
        self.assertEqual(
            [(row["row"], row["values"][1]["value"]) for row in split_values["rows"]],
            [("LLMgen Train", "234"), ("LLMgen Test", "105"), ("LLMeval Test Set", "11,340")],
        )

        metric_content = (
            "|Host|Dopant|Td [°C] / Tg [°C] / ET [eV]|Von [V]|"
            "max EQE [%] / CE [cd A−1] / PE [lm W−1]|"
            "EQE [%] / CE [cd A−1] / PE [lm W−1]|CIE [x, y]|\n"
            "|---|---|---|---|---|---|---|\n"
            "|CDPO|5CzCN|455 / 89 / 2.84|4.9|13.2 / 31.6 / 18.1|– / – / –|(0.20, 0.38)|"
        )
        metric_question = (
            "Table 5 中 CDPO 的 Td/Tg/ET、Von、最大 EQE/CE/PE 和 CIE (x,y) 分别是多少？"
        )
        metric_values = extract_table_row_values(
            metric_question, metric_content, {"table_number": 5}
        )
        self.assertEqual(
            [item["value"] for item in metric_values["values"]],
            ["455 / 89 / 2.84", "4.9", "(0.20, 0.38)", "13.2 / 31.6 / 18.1"],
        )

    def test_explicit_figure_query_keeps_same_page_explanatory_text(self):
        self.assertIsNone(
            app._figure_page_text_result(
                "Figure 3 的 Top-K retrieval accuracy 是多少？",
                {"metadatas": [[{"source": "target.pdf", "page": 7}]]},
                None,
            )
        )

        class Vector(list):
            def tolist(self):
                return list(self)

        class Embedding:
            def encode(self, _message):
                return Vector([0.1, 0.2])

        class Collection:
            def count(self):
                return 2

            def query(self, **_kwargs):
                return {
                    "ids": [["figure"]],
                    "documents": [["Figure 2 labels." ]],
                    "metadatas": [[
                        {"source": "target.pdf", "type": "text", "page": 7}
                    ]],
                }

            def get(self, **kwargs):
                where = str(kwargs.get("where", ""))
                if "figure_number" in where and "figure_kind" in where:
                    return {
                        "ids": ["figure"],
                        "documents": ["Figure 2 labels."],
                        "metadatas": [{
                            "source": "target.pdf",
                            "type": "figure",
                            "page": 7,
                            "figure_kind": "figure",
                            "figure_number": 2,
                        }],
                    }
                return {
                    "ids": ["figure", "prose"],
                    "documents": [
                        "Figure 2 labels.",
                        "The correct answer is Jackie Robinson.",
                    ],
                    "metadatas": [
                        {"source": "target.pdf", "type": "figure", "page": 7,
                         "figure_kind": "figure", "figure_number": 2},
                        # Some PDF two-column continuations are conservatively
                        # classified as formula blocks; the helper must still
                        # recover their explanatory prose.
                        {"source": "target.pdf", "type": "formula", "page": 7},
                    ],
                }

        class Client:
            def __init__(self):
                self.chat = self
                self.completions = self

            def create(self, **_kwargs):
                return type(
                    "Response",
                    (),
                    {"choices": [type("Choice", (), {"message": type("Message", (), {"content": "ok"})()})()]},
                )()

        result = app.query_knowledge(
            "Figure 2 的信息流分析示例中，问题的正确答案是什么？",
            runtime=app.Runtime(
                app.RuntimeConfig(
                    retrieval_mode="hybrid",
                    spatial_figure_evidence=True,
                    hybrid_candidate_k=1,
                    context_k=2,
                ),
                Client(),
                Embedding(),
                Collection(),
            ),
            source_filter="target.pdf",
        )

        self.assertIn("Jackie Robinson", "\n".join(result["contexts"]))

    def test_document_routing_leaves_ambiguous_question_unfiltered(self):
        class Vector(list):
            def tolist(self):
                return list(self)

        class Embedding:
            def encode(self, _message):
                return Vector([0.1, 0.2])

        class Collection:
            def __init__(self):
                self.query_kwargs = None

            def count(self):
                return 2

            def query(self, **kwargs):
                self.query_kwargs = kwargs
                return {
                    "ids": [["a"]],
                    "documents": [["Generic evidence."]],
                    "metadatas": [[{"source": "a.pdf", "type": "text"}]],
                }

            def get(self, **_kwargs):
                return {
                    "ids": ["a", "b"],
                    "documents": ["DrugR evidence.", "AlphaFold3 evidence."],
                    "metadatas": [{"source": "a.pdf"}, {"source": "b.pdf"}],
                }

        class Client:
            def __init__(self):
                self.chat = self
                self.completions = self

            def create(self, **_kwargs):
                return type(
                    "Response",
                    (),
                    {"choices": [type("Choice", (), {"message": type("Message", (), {"content": "ok"})()})()]},
                )()

        collection = Collection()
        app.query_knowledge(
            "这两篇论文的方法有什么不同？",
            runtime=app.Runtime(
                app.RuntimeConfig(document_routing=True),
                Client(),
                Embedding(),
                collection,
            ),
        )
        self.assertEqual(collection.query_kwargs["where"], {"type": {"$ne": "formula"}})

    def test_document_routing_scopes_explicit_table_scan_to_selected_source(self):
        class Vector(list):
            def tolist(self):
                return list(self)

        class Embedding:
            def encode(self, _message):
                return Vector([0.1, 0.2])

        class Collection:
            def __init__(self):
                self.table_where = None

            def count(self):
                return 2

            def query(self, **_kwargs):
                return {
                    "ids": [["scidqa-table3"]],
                    "documents": [[TABLE_CONFIG_VARIANTS]],
                    "metadatas": [[
                        {
                            "source": "scidqa.pdf",
                            "type": "table",
                            "table_number": 3,
                        }
                    ]],
                }

            def get(self, **kwargs):
                self.table_where = kwargs.get("where")
                return {
                    "ids": ["scidqa-table3"],
                    "documents": [TABLE_CONFIG_VARIANTS],
                    "metadatas": [
                        {
                            "source": "scidqa.pdf",
                            "type": "table",
                            "table_number": 3,
                        }
                    ],
                }

        class Client:
            def __init__(self):
                self.chat = self
                self.completions = self

            def create(self, **_kwargs):
                raise AssertionError("structured table lookup must not call the model")

        collection = Collection()
        runtime = app.Runtime(
            app.RuntimeConfig(document_routing=True),
            Client(),
            Embedding(),
            collection,
        )
        with patch.object(
            app,
            "_get_lexical_snapshot",
            return_value=type(
                "Snapshot",
                (),
                {
                    "router": type(
                        "Router",
                        (),
                        {
                            "route": lambda _self, _question: DocumentRoute(
                                "scidqa.pdf", ("full-text",)
                            )
                        },
                    )()
                },
            )(),
        ):
            result = app.query_knowledge(
                "Table 3 中 GPT-4o 在 RAG 和 full-text 下的 Avg 分别是多少？",
                runtime=runtime,
            )

        self.assertEqual(
            collection.table_where,
            {
                "$and": [
                    {"type": {"$eq": "table"}},
                    {"source": {"$eq": "scidqa.pdf"}},
                ]
            },
        )
        self.assertEqual(result["context_metadatas"][0]["source"], "scidqa.pdf")

    def test_hybrid_falls_back_to_dense_for_weak_cross_language_bm25(self):
        class Vector(list):
            def tolist(self):
                return list(self)

        class Embedding:
            def encode(self, _message):
                return Vector([0.1, 0.2])

        class Collection:
            def count(self):
                return 3

            def query(self, **_kwargs):
                return {
                    "ids": [["stats", "pipeline", "overview"]],
                    "documents": [[
                        "The explicit-reasoning dataset contains 4,855 samples.",
                        "A closed-loop reverse-engineering pipeline constructs the annotations.",
                        "DrugR overview.",
                    ]],
                    "metadatas": [[{"type": "text"}] * 3],
                }

            def get(self, **kwargs):
                self.assert_no_table_filter(kwargs)
                return {
                    "ids": ["overview", "stats", "pipeline"],
                    "documents": [
                        "DrugR overview.",
                        "The explicit-reasoning dataset contains 4,855 samples.",
                        "A closed-loop reverse-engineering pipeline constructs the annotations.",
                    ],
                    "metadatas": [{"type": "text"}] * 3,
                }

            @staticmethod
            def assert_no_table_filter(kwargs):
                if "where" in kwargs:
                    raise AssertionError("narrative quantity question must not fetch all tables")

        class Client:
            def __init__(self):
                self.chat = self
                self.completions = self

            def create(self, **_kwargs):
                return type(
                    "Response",
                    (),
                    {"choices": [type("Choice", (), {"message": type("Message", (), {"content": "ok"})()})()]},
                )()

        config = app.RuntimeConfig(
            retrieval_mode="hybrid", hybrid_candidate_k=3, context_k=2
        )
        runtime = app.Runtime(config, Client(), Embedding(), Collection())
        result = app.query_knowledge(
            "DrugR 的显式推理数据集包含多少个样本？推理标注是通过什么管道构建的？",
            runtime=runtime,
        )
        self.assertEqual(
            result["contexts"],
            [
                "The explicit-reasoning dataset contains 4,855 samples.",
                "A closed-loop reverse-engineering pipeline constructs the annotations.",
            ],
        )

    def test_upload_invalidates_hybrid_lexical_snapshot(self):
        class Vector(list):
            def tolist(self):
                return list(self)

        class Embedding:
            def encode(self, _message):
                return Vector([0.1, 0.2])

        class Collection:
            def __init__(self):
                self.total = 0

            def count(self):
                return self.total

            def get(self, **_kwargs):
                return {"ids": [], "metadatas": []}

            def upsert(self, **kwargs):
                self.total += len(kwargs["ids"])

        runtime = app.Runtime(app.RuntimeConfig(), None, Embedding(), Collection())
        runtime._lexical_snapshot = object()
        with tempfile.NamedTemporaryFile(suffix=".txt") as handle:
            with patch.object(
                app,
                "load_and_split_document",
                return_value=[Chunk("New evidence", {"type": "text"})],
            ):
                app.add_document_to_db(handle.name, runtime=runtime)
        self.assertIsNone(runtime._lexical_snapshot)

    def test_reimport_replaces_stale_parse_only_after_success(self):
        import chromadb
        from unittest.mock import Mock

        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "paper.txt"
            file.write_text("unchanged source", encoding="utf-8")
            collection = chromadb.PersistentClient(path=str(Path(directory) / "db")).get_or_create_collection("reimport")
            embedding = Mock()
            embedding.encode.side_effect = lambda texts: np.array([[1.0, 0.0] for _ in texts])
            runtime = app.Runtime(app.RuntimeConfig(), None, embedding, collection)
            with patch.object(app, "load_and_split_document", return_value=[Chunk("old parse", {})]):
                app.add_document_to_db(str(file), runtime)
            collection.add(ids=["other"], documents=["other document"], embeddings=[[0.0, 1.0]], metadatas=[{"source": "other.txt"}])
            with patch.object(app, "load_and_split_document", return_value=[Chunk("new parse", {})]):
                embedding.encode.side_effect = RuntimeError("embedding failed")
                with self.assertRaisesRegex(RuntimeError, "embedding failed"):
                    app.add_document_to_db(str(file), runtime, replace_existing=True)
                self.assertIn("old parse", collection.get()["documents"])
                embedding.encode.side_effect = lambda texts: np.array([[1.0, 0.0] for _ in texts])
                app.add_document_to_db(str(file), runtime, replace_existing=True)
                app.add_document_to_db(str(file), runtime, replace_existing=True)
            self.assertCountEqual(collection.get()["documents"], ["new parse", "other document"])
            with patch.object(app, "load_and_split_document", return_value=[]):
                with self.assertRaisesRegex(ValueError, "没有可导入的文本"):
                    app.add_document_to_db(str(file), runtime, replace_existing=True)
            self.assertCountEqual(collection.get()["documents"], ["new parse", "other document"])

    def test_upload_batches_chunks_and_requires_confirmation_for_same_name(self):
        class Embedding:
            def __init__(self):
                self.batch_sizes = []

            def encode(self, texts):
                self.batch_sizes.append(len(texts))
                return np.array([[0.1, 0.2] for _text in texts])

        class Collection:
            def __init__(self):
                self.records = {}

            def count(self):
                return len(self.records)

            def get(self, where=None, **_kwargs):
                records = list(self.records.items())
                if where:
                    field, condition = next(iter(where.items()))
                    records = [
                        row
                        for row in records
                        if row[1][1].get(field) == condition["$eq"]
                    ]
                return {
                    "ids": [row[0] for row in records],
                    "documents": [row[1][0] for row in records],
                    "metadatas": [row[1][1] for row in records],
                }

            def upsert(self, ids, documents, metadatas, **_kwargs):
                for doc_id, document, metadata in zip(ids, documents, metadatas):
                    self.records[doc_id] = (document, metadata)

            def delete(self, ids):
                for doc_id in ids:
                    self.records.pop(doc_id, None)

        embedding = Embedding()
        collection = Collection()
        runtime = app.Runtime(app.RuntimeConfig(), None, embedding, collection)
        with tempfile.TemporaryDirectory() as directory:
            first_dir = Path(directory) / "first"
            second_dir = Path(directory) / "second"
            first_dir.mkdir()
            second_dir.mkdir()
            first = first_dir / "report.txt"
            second = second_dir / "report.txt"
            first.write_text("first", encoding="utf-8")
            second.write_text("second", encoding="utf-8")
            with patch.object(
                app,
                "load_and_split_document",
                side_effect=[
                    [Chunk(f"first-{index}", {}) for index in range(65)],
                    [Chunk("second", {})],
                ],
            ):
                app.add_document_to_db(str(first), runtime)
                with self.assertRaisesRegex(ValueError, "同名文档已存在"):
                    app.add_document_to_db(str(second), runtime)
                app.add_document_to_db(str(second), runtime, replace_existing=True)

        sources = {metadata[1]["source"] for metadata in collection.records.values()}
        self.assertEqual(embedding.batch_sizes, [64, 1, 1])
        self.assertEqual(sources, {"report.txt"})
        self.assertEqual(
            [document for document, _metadata in collection.records.values()],
            ["second"],
        )

    def test_formula_storage_keeps_existing_chunk_indices_and_ids(self):
        class Embedding:
            def encode(self, texts):
                return np.array([[0.1, 0.2] for _text in texts])

        class Collection:
            def __init__(self):
                self.records = []

            def count(self):
                return len(self.records)

            def get(self, **_kwargs):
                return {"ids": [], "metadatas": []}

            def upsert(self, **kwargs):
                self.records.append(kwargs)

        collection = Collection()
        runtime = app.Runtime(app.RuntimeConfig(), None, Embedding(), collection)
        with tempfile.NamedTemporaryFile(suffix=".txt") as handle, patch.object(
            app,
            "load_and_split_document",
            return_value=[
                Chunk("first", {"type": "text"}),
                Chunk("A ∗ u = f", {"type": "formula"}),
                Chunk("second", {"type": "text"}),
            ],
        ):
            app.add_document_to_db(handle.name, runtime=runtime)

        ids = collection.records[0]["ids"]
        metas = collection.records[0]["metadatas"]
        self.assertEqual([metas[0]["chunk_index"], metas[2]["chunk_index"]], [0, 1])
        self.assertNotIn("chunk_index", metas[1])
        self.assertRegex(metas[0]["document_version"], r"^[0-9a-f]{32}$")
        self.assertTrue(ids[0].endswith(":0:text"))
        self.assertTrue(ids[2].endswith(":1:text"))
        self.assertEqual(
            {metadata["document_version"] for metadata in metas},
            {metas[0]["document_version"]},
        )

    def test_query_uses_filtered_contexts_for_generation_and_return(self):
        class Vector(list):
            def tolist(self):
                return list(self)

        class Embedding:
            def encode(self, _message):
                return Vector([0.1, 0.2])

        class Collection:
            def count(self):
                return 3

            def query(self, **_kwargs):
                return {
                    "ids": [["legacy", "narrative"]],
                    "documents": [[TABLE_1, "Method narrative"]],
                    "metadatas": [[{"type": "text"}, {"type": "text"}]],
                }

            def get(self, **_kwargs):
                return {
                    "ids": ["table2"],
                    "documents": [TABLE_2],
                    "metadatas": [{"type": "table", "table_number": 2, "table_id": "table-2"}],
                }

        class Client:
            def __init__(self):
                self.prompt = None
                self.chat = self
                self.completions = self

            def create(self, **kwargs):
                self.prompt = kwargs["messages"][1]["content"]
                return type(
                    "Response",
                    (),
                    {"choices": [type("Choice", (), {"message": type("Message", (), {"content": "0.3404"})()})()]},
                )()

        client = Client()
        runtime = app.Runtime(app.RuntimeConfig(retrieval_k=2, context_k=2), client, Embedding(), Collection())
        result = app.query_knowledge(
            "Table 2 中 DrugR* 的 Target property F1 score 是多少？",
            return_contexts=True,
            runtime=runtime,
        )
        self.assertIn("Table 2 结构化单元格", result["contexts"][0])
        self.assertIn("值=0.3404", result["contexts"][0])
        self.assertEqual(result["answer"], "根据 Table 2 中“DrugR*”行的“F1”列，数值为 **0.3404**。")
        # Deterministic table lookups must not call the generation model.
        self.assertIsNone(client.prompt)

    def test_hybrid_mode_preserves_explicit_table_cell_protection(self):
        class Vector(list):
            def tolist(self):
                return list(self)

        class Embedding:
            def encode(self, _message):
                return Vector([0.1, 0.2])

        class Collection:
            def count(self):
                return 2

            def query(self, **_kwargs):
                return {
                    "ids": [["table1"]],
                    "documents": [[TABLE_1]],
                    "metadatas": [[{"type": "table", "table_number": 1}]],
                }

            def get(self, **_kwargs):
                return {
                    "ids": ["table1", "table2"],
                    "documents": [TABLE_1, TABLE_2],
                    "metadatas": [
                        {"type": "table", "table_number": 1, "table_caption": "Table 1"},
                        {"type": "table", "table_number": 2, "table_caption": "Table 2"},
                    ],
                }

        class Client:
            def __init__(self):
                self.chat = self
                self.completions = self

            def create(self, **_kwargs):
                raise AssertionError("explicit table cell must not call the model")

        config = app.RuntimeConfig(
            retrieval_mode="hybrid", hybrid_candidate_k=2, context_k=2
        )
        class Reranker:
            def rerank(self, _question, candidates, _documents):
                ranked = [RankedItem(item.key, 1.0) for item in reversed(candidates)]
                return ranked

        runtime = app.Runtime(
            config, Client(), Embedding(), Collection(), reranker=Reranker()
        )
        result = app.query_knowledge(
            "Table 2 中 DrugR* 的 Target property F1 score 是多少？",
            runtime=runtime,
        )
        self.assertEqual(
            result["answer"],
            "根据 Table 2 中“DrugR*”行的“F1”列，数值为 **0.3404**。",
        )
        self.assertEqual(result["context_metadatas"][0]["table_number"], 2)

    def test_explicit_missing_table_never_falls_back_to_other_table(self):
        class Vector(list):
            def tolist(self):
                return list(self)

        class Embedding:
            def encode(self, _message):
                return Vector([0.1, 0.2])

        class Collection:
            def count(self):
                return 1

            def query(self, **_kwargs):
                return {
                    "ids": [["table1"]],
                    "documents": [[TABLE_1]],
                    "metadatas": [[{"type": "table", "table_number": 1}]],
                }

            def get(self, **_kwargs):
                return {"ids": ["table1"], "documents": [TABLE_1], "metadatas": [{"type": "table", "table_number": 1}]}

        class Client:
            def __init__(self):
                self.chat = self
                self.completions = self

            def create(self, **_kwargs):
                raise AssertionError("missing explicit table must not call the model")

        runtime = app.Runtime(app.RuntimeConfig(retrieval_k=1, context_k=1), Client(), Embedding(), Collection())
        result = app.query_knowledge("Table 2 中 DrugR* 的整体优化得分是多少？", runtime=runtime)
        self.assertIn("Table 2", result["answer"])
        self.assertIn("不能用其他表格替代", result["answer"])

    def test_document_inventory_and_confirmed_deletion(self):
        version = "a" * 32
        shared_version = "b" * 32
        third_version = "c" * 32

        class Collection:
            def __init__(self):
                self.records = [
                    ("a-1", {"source": "a.pdf", "document_version": version}),
                    ("a-2", {"source": "a.pdf", "document_version": shared_version}),
                    ("b-1", {"source": "b.txt", "document_version": shared_version}),
                    ("c-1", {"source": "c.txt", "document_version": third_version}),
                ]
                self.deleted = []

            def count(self):
                return len(self.records)

            def get(self, where=None, **_kwargs):
                records = self.records
                if where:
                    field, condition = next(iter(where.items()))
                    value = condition["$eq"]
                    records = [row for row in records if row[1].get(field) == value]
                return {
                    "ids": [row[0] for row in records],
                    "metadatas": [row[1] for row in records],
                }

            def delete(self, ids):
                self.deleted.extend(ids)
                self.records = [row for row in self.records if row[0] not in ids]

        with tempfile.TemporaryDirectory() as directory:
            collection = Collection()
            runtime = app.Runtime(
                app.RuntimeConfig(db_path=directory),
                object(),
                object(),
                collection,
            )
            runtime._lexical_snapshot = object()
            source_pdf = Path(directory) / "source_pdfs" / f"{version}.pdf"
            source_pdf.parent.mkdir()
            source_pdf.write_bytes(b"pdf")
            shared_pdf = source_pdf.with_name(f"{shared_version}.pdf")
            shared_pdf.write_bytes(b"pdf")
            third_pdf = source_pdf.with_name(f"{third_version}.pdf")
            third_pdf.write_bytes(b"pdf")

            self.assertEqual(
                app.document_inventory(runtime),
                [("a.pdf", 2), ("b.txt", 1), ("c.txt", 1)],
            )
            self.assertEqual(
                app.delete_document(["a.pdf", "c.txt"], False, runtime=runtime),
                "请先确认删除。",
            )
            self.assertEqual(collection.deleted, [])

            status = app.delete_document(["a.pdf", "c.txt"], True, runtime=runtime)

            self.assertIn("2 份文档（3 个文本块）", status)
            self.assertEqual(collection.deleted, ["a-1", "a-2", "c-1"])
            self.assertEqual(app.document_inventory(runtime), [("b.txt", 1)])
            self.assertFalse(source_pdf.exists())
            self.assertTrue(shared_pdf.exists())
            self.assertFalse(third_pdf.exists())
            self.assertIsNone(runtime._lexical_snapshot)

    def test_upload_file_preserves_pathlib_path(self):
        source = Path("folder") / "document.txt"
        runtime = object()
        with patch.object(app, "add_document_to_db", return_value="ok") as add:
            self.assertEqual(app.upload_file(source, runtime), "ok")
        add.assert_called_once_with(
            str(source),
            runtime=runtime,
            progress=None,
            replace_existing=False,
        )

    def test_upload_file_reports_parser_errors(self):
        with patch.object(app, "add_document_to_db", side_effect=ValueError("文件损坏")):
            self.assertEqual(app.upload_file("broken.pdf", object()), "添加失败：文件损坏")

    def test_quiz_json_is_parsed_and_scored(self):
        response = json.dumps(
            {"questions": [
                {
                    "question": f"问题 {index}",
                    "options": ["甲", "乙", "丙", "丁"],
                    "answer": "A",
                    "explanation": f"解析 {index}",
                }
                for index in range(1, 6)
            ]},
            ensure_ascii=False,
        )
        items = app.parse_quiz_items(response)

        self.assertEqual(len(items), 5)
        self.assertIn("得分：4 / 5", app.score_quiz(items, ["A. 甲"] * 4 + ["B. 乙"]))
        self.assertEqual(app.score_quiz(items, ["A. 甲"] * 4), "请完成全部 5 道题后再提交。")

    def test_evidence_panel_shows_exact_context_and_location(self):
        panel = app.format_evidence_panel(
            {
                "contexts": ["原文第一行\n原文第二行"],
                "context_metadatas": [
                    {"source": "paper.pdf", "page": 3, "type": "text"}
                ],
            }
        )

        self.assertIn("paper.pdf，第 3 页", panel)
        self.assertIn("> 原文第一行", panel)

    def test_model_service_can_be_configured_after_local_startup(self):
        class Collection:
            def count(self):
                return 1

        runtime = app.Runtime(
            app.RuntimeConfig(),
            None,
            object(),
            Collection(),
        )
        self.assertIn("设置", app.generate_mindmap(runtime))
        self.assertIn("设置", app.generate_quiz(runtime))

        with patch("openai.OpenAI") as openai_client:
            self.assertIn(
                "Base URL",
                app.configure_model_service("file:///tmp/model", "test-model", runtime=runtime),
            )
            self.assertIn(
                "API Key",
                app.configure_model_service(
                    "https://example.com/v1", "test-model", runtime=runtime
                ),
            )
        openai_client.assert_not_called()

        with patch("openai.OpenAI") as openai_client:
            openai_client.return_value.models.list.return_value.data = [
                type("Model", (), {"id": "test-model"})()
            ]
            status = app.configure_model_service(
                "https://example.com/v1",
                "test-model",
                "secret-value",
                runtime=runtime,
            )

        openai_client.assert_called_once_with(
            api_key="secret-value",
            base_url="https://example.com/v1",
        )
        self.assertIs(runtime.client, openai_client.return_value)
        self.assertEqual(runtime.config.llm_model, "test-model")
        self.assertNotIn("secret-value", status)
        openai_client.return_value.models.list.assert_called_once_with()

        previous_client = runtime.client
        with patch("openai.OpenAI") as openai_client:
            error = RuntimeError("unauthorized")
            error.status_code = 401
            openai_client.return_value.models.list.side_effect = error
            status = app.configure_model_service(
                "https://example.com/v1",
                "other-model",
                "invalid-value",
                runtime=runtime,
            )
        self.assertIn("API Key 无效", status)
        self.assertIs(runtime.client, previous_client)
        self.assertEqual(runtime.config.llm_model, "test-model")

        with patch("openai.OpenAI") as openai_client:
            status = app.configure_model_service(
                "http://localhost:11434/v1",
                "qwen3:4b-instruct",
                runtime=runtime,
            )
        openai_client.assert_called_once_with(
            api_key="local",
            base_url="http://localhost:11434/v1",
        )
        self.assertIn("qwen3:4b-instruct", status)

    def test_create_runtime_allows_missing_api_key(self):
        collection = object()
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("dotenv.load_dotenv"),
            patch("openai.OpenAI") as openai_client,
            patch("sentence_transformers.SentenceTransformer") as embedding_model,
            patch("chromadb.PersistentClient") as chroma_client,
        ):
            chroma_client.return_value.get_or_create_collection.return_value = collection
            runtime = app.create_runtime(app.RuntimeConfig(embedding_model="test-model"))

        openai_client.assert_not_called()
        embedding_model.assert_called_once_with("test-model")
        self.assertIsNone(runtime.client)
        self.assertIs(runtime.collection, collection)

    def test_generation_helpers_share_the_context_limit(self):
        class Collection:
            def count(self):
                return 6

            def get(self, **_kwargs):
                return {"documents": [f"chunk-{index}" for index in range(6)]}

        class Completions:
            def __init__(self):
                self.calls = []

            def create(self, **kwargs):
                self.calls.append(kwargs)
                message = type("Message", (), {"content": "ok"})()
                choice = type("Choice", (), {"message": message})()
                return type("Response", (), {"choices": [choice]})()

        completions = Completions()
        chat = type("Chat", (), {"completions": completions})()
        client = type("Client", (), {"chat": chat})()
        runtime = app.Runtime(app.RuntimeConfig(), client, object(), Collection())

        self.assertEqual(runtime.config.context_k, 4)
        self.assertEqual(app.generate_mindmap(runtime), "ok")
        self.assertEqual(app.generate_quiz(runtime), "ok")
        self.assertEqual(completions.calls[0]["messages"][1]["content"].count("chunk-"), 6)
        self.assertEqual(completions.calls[1]["messages"][1]["content"].count("chunk-"), 4)

    def test_context_budget_skips_oversized_chunks_without_cutting_them(self):
        runtime = app.Runtime(
            app.RuntimeConfig(llm_context_tokens=420, llm_max_tokens=100),
            None,
            None,
            None,
        )

        selected, prompt = app._pack_contexts(
            runtime,
            "system",
            "instruction",
            ["x" * 1200, "small evidence"],
        )

        self.assertEqual(selected, [1])
        self.assertNotIn("x" * 20, prompt)
        self.assertIn("【原文引用 1】\nsmall evidence", prompt)

    def test_truncated_generation_is_visible_and_truncated_quiz_is_rejected(self):
        message = type("Message", (), {"content": "partial"})()
        choice = type("Choice", (), {"message": message, "finish_reason": "length"})()
        response = type("Response", (), {"choices": [choice]})()
        completions = type("Completions", (), {"create": lambda self, **_kwargs: response})()
        chat = type("Chat", (), {"completions": completions})()
        runtime = app.Runtime(
            app.RuntimeConfig(),
            type("Client", (), {"chat": chat})(),
            None,
            None,
        )

        self.assertIn("尚未完成", app._complete_text(runtime, "system", "prompt"))
        with self.assertRaisesRegex(ValueError, "未载入测评"):
            app._complete_text(runtime, "system", "prompt", json_output=True)

    def test_deepseek_generation_disables_thinking_and_requests_json(self):
        from unittest.mock import Mock

        client = Mock()
        client.chat.completions.create.return_value.choices = [
            Mock(message=Mock(content='{"questions": []}'), finish_reason="stop")
        ]
        runtime = app.Runtime(app.RuntimeConfig(), client, None, None)

        app._complete_text(runtime, "system", "prompt", json_output=True)

        request = client.chat.completions.create.call_args.kwargs
        self.assertEqual(request["response_format"], {"type": "json_object"})
        self.assertEqual(request["extra_body"], {"thinking": {"type": "disabled"}})
        self.assertEqual(request["max_tokens"], 2048)

    def test_question_budget_keeps_prompt_and_evidence_panel_aligned(self):
        from unittest.mock import Mock

        collection = Mock()
        collection.count.return_value = 2
        collection.query.return_value = {
            "ids": [["oversized", "fits"]],
            "documents": [["overlong " * 4000, "Complete short evidence."]],
            "metadatas": [[{"source": "a.txt"}, {"source": "a.txt"}]],
        }
        embedding = Mock()
        embedding.encode.return_value.tolist.return_value = [1.0, 0.0]
        client = Mock()
        client.chat.completions.create.return_value.choices = [
            Mock(message=Mock(content="Partial answer"), finish_reason="length")
        ]
        runtime = app.Runtime(app.RuntimeConfig(), client, embedding, collection)
        result = app.query_knowledge("Explain this.", runtime=runtime)
        request = client.chat.completions.create.call_args.kwargs
        prompt = request["messages"][1]["content"]
        self.assertEqual(result["context_ids"], ["fits"])
        self.assertEqual(result["contexts"], ["Complete short evidence."])
        self.assertIn("【原文引用 1】", prompt)
        self.assertNotIn("overlong", prompt)
        self.assertIn("尚未完成", result["answer"])
        self.assertLessEqual(
            sum(app._estimated_tokens(message["content"]) for message in request["messages"])
            + request["max_tokens"] + 128,
            runtime.config.llm_context_tokens,
        )
        client.chat.completions.create.reset_mock()
        result = app.query_knowledge("中文" * 4000, runtime=runtime)
        self.assertIn("输入预算不足", result["answer"])
        self.assertEqual(result["contexts"], [])
        client.chat.completions.create.assert_not_called()

    def test_learning_generation_filters_real_collection(self):
        import chromadb
        from unittest.mock import Mock

        with tempfile.TemporaryDirectory() as directory:
            collection = chromadb.PersistentClient(path=directory).get_or_create_collection("scope-test")
            collection.add(
                ids=["a", "b"], documents=["SELECTED_EVIDENCE", "OTHER_EVIDENCE"],
                metadatas=[{"source": "a.pdf"}, {"source": "b.pdf"}],
                embeddings=[[1.0, 0.0], [0.0, 1.0]],
            )
            client = Mock()
            client.chat.completions.create.return_value.choices = [
                Mock(message=Mock(content="ok"))
            ]
            runtime = app.Runtime(app.RuntimeConfig(), client, object(), collection)
            for generate in (app.generate_mindmap, app.generate_quiz):
                for sources in (["a.pdf"], ["a.pdf", "b.pdf"], []):
                    self.assertEqual(generate(runtime, source_filter=sources), "ok")
                    prompt = client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
                    self.assertIn("SELECTED_EVIDENCE", prompt)
                    self.assertEqual("OTHER_EVIDENCE" in prompt, sources != ["a.pdf"])
                client.chat.completions.create.reset_mock()
                self.assertIn("没有可用内容", generate(runtime, source_filter=["deleted.pdf"]))
                client.chat.completions.create.assert_not_called()

    def test_outline_uses_each_main_section_instead_of_only_opening_chunks(self):
        from unittest.mock import Mock

        documents = [
            "INTRODUCTION " * 30,
            "INTRODUCTION CONTINUED " * 30,
            "METHODOLOGY " * 30,
            "RESULTS " * 30,
            "CONCLUSION " * 30,
            "REFERENCES " * 30,
            "APPENDIX " * 30,
        ]
        metadatas = [
            {"source": "paper.pdf", "chunk_index": 0, "headers": "H2: Introduction"},
            {"source": "paper.pdf", "chunk_index": 1, "headers": "H2: Introduction"},
            {"source": "paper.pdf", "chunk_index": 2, "headers": "H2: Methodology"},
            {"source": "paper.pdf", "chunk_index": 3, "headers": "H2: Results"},
            {"source": "paper.pdf", "chunk_index": 4, "headers": "H2: Conclusion"},
            {"source": "paper.pdf", "chunk_index": 5, "headers": "H2: References"},
            {"source": "paper.pdf", "chunk_index": 6, "headers": "H2: Appendix"},
        ]
        collection = Mock()
        collection.count.return_value = len(documents)
        collection.get.return_value = {
            "documents": documents,
            "metadatas": metadatas,
        }
        client = Mock()
        client.chat.completions.create.return_value.choices = [
            Mock(message=Mock(content="outline"), finish_reason="stop")
        ]
        runtime = app.Runtime(app.RuntimeConfig(), client, object(), collection)

        self.assertEqual(app.generate_mindmap(runtime, ["paper.pdf"]), "outline")

        request = client.chat.completions.create.call_args.kwargs
        prompt = request["messages"][1]["content"]
        for marker in ("INTRODUCTION", "METHODOLOGY", "RESULTS", "CONCLUSION"):
            self.assertIn(marker, prompt)
        self.assertNotIn("REFERENCES", prompt)
        self.assertNotIn("APPENDIX", prompt)

    def test_outline_filters_non_academic_sections_and_fits_every_section(self):
        documents = [
            "PAPER TITLE AND AUTHORS",
            "ABSTRACT " * 80,
            "INTRODUCTION " * 120,
            "METHODS " * 120,
            "PREDICTED CHANGES " * 120,
            "RESULTS " * 120,
            "AUTHOR CONTRIBUTIONS " * 120,
            "DATA AVAILABILITY " * 120,
            "CONCLUSION " * 120,
            "REFERENCES " * 120,
        ]
        metadatas = [
            {"source": "paper.pdf", "chunk_index": 0, "headers": "H1: Paper title"},
            {"source": "paper.pdf", "chunk_index": 1, "headers": "H1: Paper title"},
            {"source": "paper.pdf", "chunk_index": 2, "headers": "H2: 1 Introduction"},
            {"source": "paper.pdf", "chunk_index": 3, "headers": "H2: 2 Methods"},
            {
                "source": "paper.pdf",
                "chunk_index": 4,
                "headers": "H3: and the predicted changes",
            },
            {"source": "paper.pdf", "chunk_index": 5, "headers": "H2: 3 Results"},
            {
                "source": "paper.pdf",
                "chunk_index": 6,
                "headers": "H2: Author contributions",
            },
            {
                "source": "paper.pdf",
                "chunk_index": 7,
                "headers": "H2: Data availability",
            },
            {"source": "paper.pdf", "chunk_index": 8, "headers": "H2: Conclusion"},
            {"source": "paper.pdf", "chunk_index": 9, "headers": "H2: References"},
        ]

        candidates, labels = app._outline_section_candidates(documents, metadatas)
        joined_labels = " ".join(labels)
        self.assertEqual(len(candidates), 5)
        self.assertNotIn("and the predicted changes", joined_labels)
        self.assertNotIn("Author contributions", joined_labels)
        self.assertNotIn("Data availability", joined_labels)
        self.assertNotIn("References", joined_labels)
        self.assertTrue(candidates[0].startswith("ABSTRACT"))

        runtime = app.Runtime(
            app.RuntimeConfig(llm_context_tokens=1600, llm_max_tokens=512),
            None,
            None,
            None,
        )
        system = "system"
        instruction = "instruction"
        compacted = app._outline_candidates_within_budget(
            runtime, system, instruction, candidates, labels
        )
        selected, prompt = app._pack_contexts(
            runtime, system, instruction, compacted, labels=labels
        )
        self.assertEqual(len(selected), len(candidates))
        for label in labels:
            self.assertIn(label, prompt)

    def test_selected_papers_supply_both_named_sources_without_forcing_extra_papers(self):
        import chromadb
        from unittest.mock import Mock

        class Embedding:
            def encode(self, _question):
                return type("Vector", (list,), {"tolist": lambda self: list(self)})([1.0, 0.0])

        with tempfile.TemporaryDirectory() as directory:
            collection = chromadb.PersistentClient(path=directory).get_or_create_collection("multi-scope")
            collection.add(
                ids=["alpha1", "alpha2", "alpha3", "alpha4", "beta", "gamma"],
                documents=["Alpha method overview.", "Alpha background.", "Alpha results.",
                           "Alpha appendix.", "Beta method selects sentences.", "Gamma unrelated."],
                metadatas=[{"source": "alpha.pdf", "type": "text", "headers": "H1: Alpha"}] * 4
                + [{"source": "beta.pdf", "type": "text", "headers": "H1: Beta"},
                   {"source": "gamma.pdf", "type": "text", "headers": "H1: Gamma"}],
                embeddings=[[1.0, 0.0]] * 4 + [[0.0, 1.0], [0.0, 1.0]],
            )
            client = Mock()
            client.chat.completions.create.return_value.choices = [Mock(message=Mock(content="ok"))]
            runtime = app.Runtime(app.RuntimeConfig(), client, Embedding(), collection)
            sources = ["alpha.pdf", "beta.pdf", "gamma.pdf"]
            result = app.query_knowledge("Alpha 与 Beta 的方法分别是什么？", runtime=runtime, source_filter=sources)
            self.assertEqual({meta["source"] for meta in result["context_metadatas"]},
                             {"alpha.pdf", "beta.pdf"})
            self.assertIn("Beta method", client.chat.completions.create.call_args.kwargs["messages"][1]["content"])
            self.assertIn("问题词：beta", client.chat.completions.create.call_args.kwargs["messages"][1]["content"])

            result = app.query_knowledge("Alpha 的方法是什么？", runtime=runtime, source_filter=sources)
            self.assertEqual({meta["source"] for meta in result["context_metadatas"]}, {"alpha.pdf"})

            result = app.query_knowledge(
                "这两篇论文各自使用了什么方法？请分别说明。",
                runtime=runtime,
                source_filter=["alpha.pdf", "beta.pdf"],
            )
            self.assertEqual(
                {meta["source"] for meta in result["context_metadatas"]},
                {"alpha.pdf", "beta.pdf"},
            )


if __name__ == "__main__":
    unittest.main()
