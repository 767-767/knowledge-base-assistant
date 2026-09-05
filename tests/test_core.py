import hashlib
import os
import tempfile
import unittest
from unittest.mock import patch

import app
from sci_rag_reranking import RerankResult
from sci_rag_retrieval import DocumentRoute, RankedItem
from sci_rag_core import (
    Chunk,
    build_evidence_ledger,
    build_evidence_retry_prompt,
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
    normalize_for_match,
    parse_markdown_table,
    missing_pdf_formula_blocks,
    repair_latex_json_escapes,
    rerank_table_first,
    select_row_entity,
    split_to_chunks,
    supplement_answer_with_evidence,
    supplement_formula_with_evidence,
    table_numbers_from_question,
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


class CoreTests(unittest.TestCase):
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
        self.assertIn("逐一检查全部参考片段", app.SCIENTIFIC_SYSTEM_PROMPT)
        self.assertIn("限制问题要区分限制本身与示例现象", app.SCIENTIFIC_SYSTEM_PROMPT)
        self.assertIn("不能用“训练数据限制”等参考片段未出现的机制替代", app.SCIENTIFIC_SYSTEM_PROMPT)

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
        self.assertTrue(all(line.startswith("【片段 ") for line in ledger))

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

    def test_evidence_retry_prompt_is_bounded_and_gold_free(self):
        ledger = [
            "【片段 1，paper.pdf，Dataset】The annotation pipeline uses DeepSeek-R1.",
        ]
        validation = validate_answer_against_evidence(
            "标注管道如何构建？",
            "使用专业工具。",
            ledger,
        )
        prompt = build_evidence_retry_prompt(
            "标注管道如何构建？",
            "使用专业工具。",
            ledger,
            validation,
        )
        self.assertIn("DeepSeek-R1", prompt)
        self.assertIn("保留原答案中已有", prompt)
        self.assertNotIn("ground_truth", prompt)

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

    def test_wrapped_group_headers_are_repaired_before_lookup(self):
        markdown = (
            "|**Method**|**Formul**<br>validity|**ation design**<br>success rate|\n"
            "|---|---|---|\n|Qwen|75.5|68.3|"
        )
        tables, _body = extract_tables(markdown, "paper.pdf")
        headers, _rows = parse_markdown_table(tables[0].page_content)
        self.assertIn("Formulation design validity", headers)
        self.assertIn("Formulation design success rate", headers)

    def test_row_entity_ignores_column_name_and_normalizes_sup(self):
        question = "Table 2 中 DrugR* 的 Target property F1 score 是多少？"
        entity = select_row_entity(question, TABLE_2)
        self.assertEqual(entity, "DrugR*")
        filtered = filter_table_rows_by_entity(TABLE_2, entity)
        self.assertIn("0.3404", filtered)
        self.assertNotIn("0.2997", filtered)
        self.assertEqual(normalize_for_match("**DrugR**"), "drugr")
        self.assertEqual(normalize_for_match("**DrugR**<sup>_∗_</sup>"), "drugr*")

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
        self.assertEqual(table_numbers_from_question(question), (1, 2))
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
            {"source": "drugr.pdf", "type": "text", "page": 1},
            {"source": "drugr.pdf", "type": "text", "page": 2},
            {"source": "mgno.pdf", "type": "text", "page": 1},
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
    def test_app_import_has_no_runtime(self):
        self.assertIsNone(app._runtime)

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
                return RerankResult(ranked, 0.01, len(ranked), 0)

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
        self.assertTrue(all(meta.get("source") == "target.pdf" for meta in result["context_metadatas"]))

    def test_section_aliases_cover_source_local_method_terms(self):
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

            def upsert(self, **_kwargs):
                self.total += 1

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

    def test_formula_storage_keeps_existing_chunk_indices_and_ids(self):
        class Vector(list):
            def tolist(self):
                return list(self)

        class Embedding:
            def encode(self, _text):
                return Vector([0.1, 0.2])

        class Collection:
            def __init__(self):
                self.records = []

            def count(self):
                return len(self.records)

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
        ), patch.object(app, "file_sha256", return_value="document-hash"):
            app.add_document_to_db(handle.name, runtime=runtime)

        ids = [record["ids"][0] for record in collection.records]
        metas = [record["metadatas"][0] for record in collection.records]
        self.assertEqual([metas[0]["chunk_index"], metas[2]["chunk_index"]], [0, 1])
        self.assertNotIn("chunk_index", metas[1])
        self.assertEqual(
            ids[0],
            hashlib.sha256("document-hash:0:text:first".encode("utf-8")).hexdigest(),
        )
        self.assertEqual(
            ids[2],
            hashlib.sha256("document-hash:1:text:second".encode("utf-8")).hexdigest(),
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
                return RerankResult(ranked, 0.01, len(ranked), 0)

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


if __name__ == "__main__":
    unittest.main()
