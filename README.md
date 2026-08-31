# Sci-RAG

Scientific-paper RAG prototype with PDF/TXT/DOCX ingestion, canonical Markdown
table chunks, Chroma retrieval, DeepSeek generation, and optional RAGAS
evaluation.

## Quick start

1. Use the pinned dependencies in `requirements.txt` with a supported Python
   environment.
2. Copy `.env.example` to `.env` and set `DEEPSEEK_API_KEY` only when using
   generation or the Gradio UI.  Parsing and unit tests do not require an API
   key.
3. Run offline checks with `./venv/bin/python test_setup.py` and
   `./venv/bin/python -m unittest discover -s tests -v`.
4. Launch the UI with `./venv/bin/python app.py`.
5. Run the RAGAS evaluator only when external model calls are authorized:
   `./venv/bin/python evaluation/evaluate.py`.

`evaluation/evaluate.py` also accepts the multi-paper JSONL benchmark. Point
`SCI_RAG_DB_PATH` at an isolated, already indexed database and write reports
outside the repository, for example:

```bash
SCI_RAG_DB_PATH=/absolute/path/to/benchmark-chroma \
./venv/bin/python evaluation/evaluate.py \
  --testset evaluation/benchmark/cases.jsonl \
  --report-json /tmp/sci_rag_ragas.json \
  --report-md /tmp/sci_rag_ragas.md
```

This command regenerates answers and calls the external judge; it should not
be pointed at the production/local `chroma_db` for an audit run.

Importing `app.py` is side-effect free.  Models, ChromaDB, the OpenAI client,
and Gradio are initialized only by `create_runtime()`/the UI entrypoint.

For questions that explicitly name a table, row, and column (for example,
`Table 2` + `DrugR*` + `Target property F1 score`), the application parses the
Markdown table cell deterministically. This prevents a similarly worded Table
1 narrative from overriding the requested Table 2 value. A fresh database is
recommended when validating the new indexing metadata; the existing local
`chroma_db/` is intentionally ignored and is not rebuilt automatically.

## Phase 1 UI validation

The one-command validation launcher keeps the repository database untouched. It
creates a temporary empty ChromaDB, imports the default paper from
`../2602.08213v1.pdf`, and starts the normal UI:

```bash
bash scripts/launch_phase1_ui_test.sh
```

The launcher reads the local `.env` for `DEEPSEEK_API_KEY`. It does not copy the
key into the temporary directory. After the browser checks, press `Ctrl+C` in
the terminal; the temporary database is removed automatically.

To test only compatibility with the existing local index, use:

```bash
bash scripts/launch_phase1_ui_test.sh --existing
```

To use a different PDF in the fresh-index test:

```bash
bash scripts/launch_phase1_ui_test.sh /absolute/path/to/paper.pdf
```

Acceptance checks for the default paper:

- the upload, chat, outline, and quiz tabs are present;
- Table 2 + `DrugR*` + Overall Optimization Score returns `0.2060`;
- Table 2 + `DrugR*` + Target property F1 score returns `0.3404`;
- Table 1 + `DrugR` + Overall Optimization Score returns `0.2712`;
- the answer does not substitute Table 1 for an explicit Table 2 question;
- a normal narrative question, outline generation, quiz generation, and a
  second document upload still complete normally.

## Multi-paper benchmark (Phase 2)

The benchmark manifest is in `evaluation/benchmark/`. It currently contains
five papers and 53 cases (11 existing DrugR cases plus independently curated
cases for four additional papers). Each external PDF is represented by its
SHA-256 without storing the PDF in Git. Validate the manifest offline with:

```bash
./venv/bin/python evaluation/validate_benchmark.py
```

To verify the seed PDF and the four new PDFs against recorded hashes, pass both
external directories (the flag may be repeated):

```bash
./venv/bin/python evaluation/validate_benchmark.py \
  --papers-dir /Users/qinleqi/Desktop \
  --papers-dir /Users/qinleqi/Desktop/sci-rag-benchmark-papers \
  --verify-files --require-complete
```

Add future PDFs outside the repository, then add their metadata and cases to the
manifest/JSONL files before implementing or comparing new retrieval methods.

The current Phase 2 parser regression suite is offline and does not rebuild the
database:

```bash
./venv/bin/python -m unittest discover -s tests -v
```

The suite covers caption placement, grouped/unit table headers, PDF markup
boundaries, and layout-table false positives. A local PDF smoke check confirms
that the same ingestion path can recover table numbers and deterministic cells;
it does not prove retrieval or answer-generation quality.

For a no-API, no-Chroma multi-paper retrieval baseline, run:

```bash
./venv/bin/python evaluation/benchmark_retrieval.py \
  --papers-dir /Users/qinleqi/Desktop \
  --papers-dir /Users/qinleqi/Desktop/sci-rag-benchmark-papers \
  --top-k 1,3,5,10
```

This BM25-lite diagnostic ranks one global in-memory index and reports document,
reference-context, page, table-number, and deterministic required-fact coverage
proxies. Add `--show-failures` to print every case that is not fully covered and
its missing facts. Cross-language fact matches are accepted only through
case-level aliases that the validator checks against human gold contexts. It is
the fixed comparison baseline for Hybrid/RRF; it does not call DeepSeek or prove
answer correctness.

For a local-only Hybrid/RRF comparison (the embedding model must already be
cached), use `--retriever hybrid` with `HF_HUB_OFFLINE=1`. The diagnostic does
not change the app's default dense retrieval path or the existing ChromaDB.

## Experimental Hybrid runtime

The application still defaults to the original Chroma dense retrieval. To test
the experimental BM25 + dense Reciprocal Rank Fusion path for one process, run:

```bash
SCI_RAG_RETRIEVAL_MODE=hybrid ./venv/bin/python app.py
```

The first Hybrid question reads the current collection once and builds an
in-memory BM25 snapshot. Later questions reuse it, and a document upload through
this runtime invalidates it. `SCI_RAG_HYBRID_CANDIDATE_K` controls candidates
from each ranking (default `50`), `SCI_RAG_HYBRID_RRF_K` controls the RRF
constant (default `60`), and `SCI_RAG_CONTEXT_K` still caps generation context.
No new dependency or database rebuild is required.

For an opt-in multi-paper source-routing control, set
`SCI_RAG_DOCUMENT_ROUTING=true`. The runtime builds one lexical profile per
uploaded `source` and applies a Chroma `where={"source": ...}` filter only when
the question contains a unique high-signal identifier belonging to one source
(for example, `DrugR` or `AlphaFold3`). Ambiguous or cross-paper questions are
not routed and keep the global search. In Hybrid mode the same source filter is
also applied to the BM25 side. When a question is a bounded composite/list
question, same-section expansion may use that unique routed source; without a
unique route, multi-source collections still skip this expansion. The default
is `false`; this is a controlled multi-paper experiment, not a correctness
guarantee or a replacement for document-level evaluation.

For an opt-in parent/window context control, set
`SCI_RAG_PARENT_WINDOW=true`. The ranked top-k slots and anchor IDs stay fixed;
only the first two eligible text anchors may include an immediately adjacent
same-source, same-page text chunk. Tables, reference sections, cross-page
neighbors, and neighbors already selected in top-k are skipped. Returned
metadata records every contributing chunk ID so the expanded prompt context is
auditable. The default is `false`.

This control requires `page` and `chunk_index` metadata written by current
ingestion. Older databases without those fields safely keep the original
context unchanged; do not infer adjacency from Chroma result order. On the
five-paper retrieval benchmark, parent/window improved @10 full fact coverage
from 43/53 to 47/53 without a case regression, while adding about 1,170
characters per question on average. Generated-answer quality and API latency
still require a separate isolated-database A/B.

Explicit `Table N` handling runs after fusion: all structured table chunks are
still checked, another table cannot replace the requested one, and a resolvable
row/column question still uses deterministic cell lookup without calling the
generation model. Generic quantity phrases such as “how many samples” do not
trigger an all-table scan unless the question explicitly refers to a table.
Hybrid alone remains an opt-in retrieval experiment and is not the default.

## Experimental local cross-encoder

An additional local-only reranker can be enabled after caching the pinned model
revision. The download is about 1.1 GB and is stored in the Hugging Face cache,
not this repository:

```bash
./venv/bin/hf download BAAI/bge-reranker-base \
  config.json model.safetensors sentencepiece.bpe.model \
  special_tokens_map.json tokenizer.json tokenizer_config.json \
  --revision 2cfc18c9415c912f9d8155881c133215df768a70 --max-workers 4
```

Run the UI with Hybrid plus the reranker:

```bash
HF_HUB_OFFLINE=1 \
SCI_RAG_RETRIEVAL_MODE=hybrid \
SCI_RAG_RERANKER_MODEL=BAAI/bge-reranker-base \
SCI_RAG_RERANKER_REVISION=2cfc18c9415c912f9d8155881c133215df768a70 \
./venv/bin/python app.py
```

The model is loaded only during explicit runtime creation and always uses
`local_files_only=True`; a missing cache fails instead of downloading. The
cross-encoder ranking is conservatively fused with the original Hybrid order,
then the existing explicit-table filtering and deterministic cell lookup run as
before. Dense remains the default, and Hybrid without the reranker is unchanged.

On the fixed 5-paper/53-case offline benchmark, Hybrid + cross-encoder + RRF
raised top-10 full required-fact coverage from `0.547` to `0.698`; fact macro and
micro reached `0.785/0.776`. CPU reranking averaged `2.73 s` per 50 candidates
with a `3.31 s` p95 and about `2.20 GB` process peak RSS on the test machine.
These are retrieval-context proxies, not generated-answer accuracy. See
`PHASE2_RERANKER_HANDOFF.md` for the gate, regressions, and limitations.

## Offline answer completeness audit (Phase 3)

The retrieval benchmark checks whether required facts reached the context. The
separate answer audit checks whether a saved answer explicitly contains those
same manually declared facts; it does not call a model and does not decide
semantic truth, citation correctness, or causal logic.

Save one JSON object per line outside the repository:

```json
{"case_id":"drugr-09","answer":"...","mode":"hybrid","latency_seconds":2.1}
```

Audit the 11-paper seed set with:

```bash
./venv/bin/python evaluation/answer_audit.py \
  --testset evaluation/test_questions.json \
  --answers /tmp/sci_rag_answers.jsonl \
  --require-all \
  --json-out /tmp/sci_rag_answer_audit.json
```

Use `evaluation/benchmark/cases.jsonl` as `--testset` for the 53-case
multi-paper benchmark. The output reports answer fact macro/micro coverage,
full/partial/zero rates, and missing facts per case. This is an answer
completeness smoke check, not a replacement for human review or RAGAS.
The audit loader resolves pointer-style cases through the sibling
`evaluation/benchmark/manifest.json`, so the 11 DrugR cases are expanded from
`evaluation/test_questions.json` automatically.

To compare two fixed-question runs (for example Dense versus Hybrid+Rerank),
use the offline A/B tool. It rejects different case-ID sets so an apparent
aggregate gain cannot come from evaluating an easier subset:

```bash
./venv/bin/python evaluation/compare_answer_runs.py \
  --testset evaluation/benchmark/cases.jsonl \
  --baseline /tmp/sci_rag_answers_dense.jsonl \
  --candidate /tmp/sci_rag_answers_hybrid_reranker.jsonl \
  --baseline-name dense \
  --candidate-name hybrid-reranker \
  --require-all \
  --json-out /tmp/sci_rag_answer_compare.json
```

The report contains per-case fact-coverage deltas, missing facts, status
transitions, and aggregate macro/micro/full/partial/zero changes. It is still
lexical completeness evidence only; it does not judge semantic correctness,
table units, citations, or faithfulness.

After answer collection, generate a human-review template outside the
repository:

```bash
./venv/bin/python evaluation/review_answers.py \
  --testset evaluation/benchmark/cases.jsonl \
  --answers /tmp/sci_rag_answers_hybrid_reranker.jsonl \
  --require-all \
  --template-out /tmp/sci_rag_answer_review.jsonl
```

Fill `judgment` with `correct`, `partial`, `incorrect`, or `unanswerable`, and
label each applicable `table_number`, `units`, `formula`, and `citation` field
as `correct`, `incorrect`, `not_applicable`, or `uncertain`. Then validate and
summarize the completed file:

```bash
./venv/bin/python evaluation/review_answers.py \
  --testset evaluation/benchmark/cases.jsonl \
  --answers /tmp/sci_rag_answers_hybrid_reranker.jsonl \
  --reviews /tmp/sci_rag_answer_review_filled.jsonl \
  --require-all \
  --json-out /tmp/sci_rag_human_review.json
```

Human labels remain separate from lexical fact coverage; this report is the
place to record semantic, table, unit, formula, and citation judgments before
any optional RAGAS run.

Before interpreting an existing RAGAS JSON report, run the offline preflight:

```bash
./venv/bin/python evaluation/ragas_preflight.py \
  --report-json evaluation/evaluation_report.json \
  --testset evaluation/test_questions.json \
  --require-complete
```

Add `--require-trace` for a future report that must preserve both the complete
generation context trace and the evaluated top-N prefix. The preflight checks
case/score consistency and reports missing trace, reference-context, or model
metadata as warnings/errors. It cannot prove from a saved report alone that a
RAGAS metric used `ground_truth`, that generation and evaluation saw identical
contexts, or that answers are semantically correct.

Current validation has 145 offline tests passing. On the existing
104-chunk DrugR database, all 11 seed questions reached full required-fact
coverage in one Dense run and one Hybrid+Rerank run; this remains a
single-paper generation check, not a multi-paper or RAGAS generalization
result. The pre-alias multi-paper retrieval baseline reported Hybrid+CE+RRF
@10 full required-fact context coverage of 0.698 across 53 cases, with 16
cases incomplete. After PDF-verified alias/markup corrections (without
changing ranking), the same result is 0.717 with 15 cases incomplete. See
`PHASE3_ANSWER_COMPLETENESS_HANDOFF.md` and
`PHASE4_RETRIEVAL_FAILURE_AUDIT.md` for scope and temporary JSONL paths.

Phase 4.0 has completed an offline root-cause audit of those 16 incomplete
cases. The audit separates gold-fact surface/normalization gaps, Hybrid
top-50 candidate misses, cross-encoder demotions, and final RRF fusion
dilution; it does not yet change the default retriever. See
`PHASE4_RETRIEVAL_FAILURE_AUDIT.md` before running a five-paper generation
benchmark.

Phase 4.0 also compared CE-only with the current CE-plus-original-rank RRF
under the same candidate pool. Both reached 0.717 full fact coverage at
`@10`, but they recovered different cases; neither has replaced the default
fusion path. An offline weighted-RRF sweep (CE weights 2, 4, and 8) also
found no strict overall improvement: weight 2 raised fact macro/micro to
0.800/0.789 but kept full coverage at 0.717 and lowered page hit to 0.881;
weights 4 and 8 lowered full coverage to 0.698. The default fusion therefore
remains unchanged.

The same-section expansion safety check found that unbounded expansion can
pollute a multi-paper top-10; it is now bounded to a single-source context,
requires an existing section anchor, caps additions at six chunks, and is
skipped for multi-source collections. This preserves the single-paper UI path
without claiming a multi-paper retrieval gain.

The candidate-k=80 control increased fact macro/micro only to 0.800/0.789
while leaving full coverage at 0.717; rerank mean/P95 rose to 4.453/5.231s
with peak RSS about 2.685GB. The default candidate-k therefore remains 50.

The Phase 4.2 provenance audit is included in every offline retrieval JSON
report. For each required fact it records the matched chunk's benchmark
document, source, page, section and chunk type, and flags matches found only
in another paper, outside the annotated source pages, under a References
section, in a figure/caption chunk, or without page metadata. These fields are
diagnostic only: they do not filter results or change the default ranking.
Use the per-case `provenance` object to separate lexical coverage from
target-document evidence; missing provenance metadata is reported as
`unknown`, not silently trusted.

In the alias-corrected Hybrid + cross-encoder + equal-RRF control run, the
diagnostic found 115 matched required facts at `@10`; six facts were matched
only outside their annotated source pages, with no fact matched only in another
paper or in a References section. This is an evidence-location diagnostic,
not an improvement to the `0.794/0.782/0.717` retrieval proxy scores.

## Phase 5.4 generation benchmark result

With explicit user authorization, the five-paper/53-case benchmark was run
through the same Gradio query path in a temporary Chroma database (479 chunks;
the original `chroma_db` was not changed). Dense and Hybrid+Rerank each covered
53/53 cases. The saved answers remain outside the repository under `/tmp`.

The deterministic answer-fact audit was `0.5818/0.5411` macro/micro for Dense
and `0.6211/0.6027` for Hybrid+Rerank; full coverage was `0.4528` and `0.4906`.
Hybrid improved lexical coverage on 10 cases and regressed 5, while mean
latency increased from about `1.60s` to `4.32s`.

The independent semantic pass classified Dense as `22 correct / 9 partial /
22 incorrect`, and Hybrid+Rerank as `22 / 13 / 18`; both therefore had a
41.51% fully-correct rate in this internal review. The old generation batch
still contains table-cell mapping errors; the deterministic guard added in
Phase 5.6 covers the verified table patterns, while units, multi-fact
composition, and false “资料未提供” refusals remain unresolved.
The full evidence and per-case rationale are in
`PHASE5_GENERATION_BENCHMARK_REPORT.md`; versioned review labels are in
`evaluation/benchmark/reviews_dense_53.jsonl` and
`evaluation/benchmark/reviews_hybrid_reranker_53.jsonl`.
The subsequent RAGAS 0.4.3 reports are kept outside Git under `/tmp`; its
three configured metrics declare no `reference` input, so their scores do not
use `ground_truth` and must not be read as correctness.

The post-benchmark table regression guard now resolves entities in any table
column (including PDF tables with blank merged leading cells), preserves
`<br>` value variants, and carries spanning `Test(C&L)`/`Test(Other)` markers
when the same row is repeated under multiple settings. The guard is
table-agnostic and is covered by the offline suite; it does not rebuild the
existing database.

### Phase 5.7 regression evidence

The repaired build was opened on `127.0.0.1:7861` against a temporary
five-paper Chroma index (479 chunks). The UI showed the index count and the
chat tab opened normally. Through the same Gradio chat endpoint, deterministic
checks passed for both Table 2 settings, Darcy rough L2/H1, GPT-4o RAG/FT,
MgNO baseline/six-layer values, and the previously failing DrugR Table 2
`DrugR*` questions. The temporary server was stopped and the production
`chroma_db/` was not touched.

The repaired 53-case generation rerun is kept outside Git under
`/private/tmp/sci_rag_*_phase56.*`. Dense fact macro/micro coverage is
`0.6903/0.6438` (full `56.60%`, mean `1.716s`); Hybrid+Rerank is
`0.7201/0.6918` (full `62.26%`, mean `4.219s`). These are lexical answer
checks, not semantic accuracy. RAGAS was not rerun to completion because its
first task exceeded the 180-second timeout; the historical RAGAS reports must
not be mixed with this new batch.

### Phase 5.9 query-decomposition control

Composite questions can be retrieved with the original query plus bounded
punctuation/conjunction clauses by setting
`SCI_RAG_QUERY_DECOMPOSITION=true`; the default remains `false`. The same
control is available to the offline benchmark as `--query-decomposition`.
On the fixed five-paper/53-case routing + Hybrid + cross-encoder benchmark it
changed @10 fact macro/micro/full from `0.805/0.796/0.717` to
`0.843/0.830/0.774`, while target-document and Table N hit stayed at
`1.000/0.944`; page hit fell from `0.929` to `0.905`. It is therefore an
opt-in A/B path, not a new default. See
`PHASE5_9_QUERY_DECOMPOSITION_HANDOFF.md` and the external JSON report
`/private/tmp/sci_rag_benchmark_query_decomposition_phase59.json`.

### Phase 6.0 retrieval convergence

The Phase 5.9 decomposition implementation incorrectly treated an otherwise
unsplittable question as two variants when stripping only its final question
mark. That duplicate-query path is now removed: extra variants are produced
only after a real clause split. Table intent is also restricted to an explicit
table number or a deictic reference such as `下表`/`该表格中`; research topics
such as “scientific table representation learning” no longer scan every table.

The offline benchmark can now add the same deterministic application table
path with `--structured-table-guard`. When document routing has already chosen
a source, the web table scan uses that source too, preventing a same-numbered
table in another uploaded paper from winning the row/column lookup.

With routing + corrected query decomposition + the structured table guard,
the fixed five-paper/53-case @10 result is fact macro/micro/full
`0.881/0.871/0.811` (43/53 full), with target-document/page/Table-N hit
`1.000/0.929/1.000`. Relative to Phase 5.9, only `scidqa-09` and
`table-llm-10` changed, both from zero to full; no case regressed. This remains
retrieval-context evidence, not generated-answer correctness, and no runtime
experimental switch was made default.

An offline `--adjacent-context` negative control improved four incomplete
cases but regressed three previously full cases and reduced page hit to
`0.881`. It is not connected to the web path. See
`PHASE6_RETRIEVAL_CONVERGENCE_HANDOFF.md` for the exact reports, remaining ten
failures, and the next parent/window-context experiment.

The subsequent non-displacing `--parent-window` control keeps the original
top-10 ranking and enriches only the first two text anchors. It reached @10
fact macro/micro/full `0.936/0.932/0.887` (47/53 full) with unchanged
target-document/page/Table-N hit `1.000/0.929/1.000`; four cases improved and
none regressed. Across 53 cases it added 60,388 characters. This is effective
retrieval-context evidence, not proof that the generator uses the added text
correctly. The final DeepSeek generation A/B completed all 53 pairs with no API
errors. Automatic answer fact coverage changed from `0.8459/0.8151` macro/micro
to `0.9088/0.8836`, with full coverage `40/53` to `45/53`; internal human review
changed from `40 correct / 11 partial / 2 incorrect` to `43 / 8 / 2`. Four
changes were genuine semantic improvements; one was only a lexical-alias
artifact. A figure/OCR sample-count question remained wrong in both runs, so
the switch remains opt-in and disabled by default.

### Phase 6.2 experimental spatial figure evidence

Born-digital PDFs can optionally preserve short positioned text blocks above a
recognized `Figure N` caption by setting `SCI_RAG_SPATIAL_FIGURE_EVIDENCE=true`
before ingestion. Each experimental `type=figure` chunk stores normalized x/y
ranges, page, figure kind, and figure number. An explicit `Figure N`/`图N`
question scans the exact figure inside the existing document route; ordinary
questions exclude figure chunks from dense, BM25, routing, and cross-encoder
candidates so the extra labels cannot perturb normal retrieval.

This is not OCR or multimodal image understanding. Images are still neither
embedded nor persisted, and image-only/Extended Data panels often have no
usable text blocks. Enabling the switch changes newly ingested chunks, so test
it only with a fresh isolated database; it does not retrofit an existing DB.
The equivalent offline control is `--spatial-figure-evidence`.

Across the five fixed PDFs the opt-in parser produced 23 figure chunks (502
total versus the 479-chunk control corpus). The isolated retrieval path retained
the Phase 6.1 @10 fact macro/micro/full result `0.936/0.932/0.887` and
target-document/page/Table-N hit `1.000/0.929/1.000`. For the prior AlphaFold 3
Figure 1 failure, the exact coordinate evidence keeps protein–RNA `n=25`,
protein–dsDNA `n=38`, and CASP15 RNA `n=8` in separate horizontal groups; a
targeted DeepSeek run answered all three correctly. This single case is a
regression check, not evidence of general multimodal capability, so the switch
remains disabled by default.

### Phase 6.3 generation stability

The reusable `evaluation/generation_stability.py` runner repeats the complete
53-case benchmark against one isolated database and one fixed retrieval
configuration. It records every answer and context trace outside Git and
supports safe resume/retry by `(repeat, case_id)`.

Two full repeats (106/106 API calls successful) produced identical context IDs
and metadata for all 53 cases. Exact answer text matched after whitespace
normalization in 16/53 cases, while only two cases changed deterministic fact
status (`mgno-02` full→partial and `af3-02` partial→full). Both repeats had
answer fact macro/micro/full around `0.80/0.774/0.717`. This separates stable
retrieval from variable generation wording; the lexical audit is not semantic
answer accuracy and does not justify changing the default temperature or adding
a second API call. See `PHASE6_3_GENERATION_STABILITY_HANDOFF.md`.

The evidence-only answer diagnostic is available as a separate offline check:

```bash
./venv/bin/python evaluation/validate_answer_evidence.py \
  --answers /private/tmp/sci_rag_phase63_stability_53x2.jsonl \
  --json-out /private/tmp/sci_rag_phase63_evidence_validation.json
```

It rebuilds the same literal evidence ledger visible to generation and emits
`review`, `not_applicable`, or `ok` signals. It never reads `ground_truth` or
`required_facts`, cannot establish semantic correctness, and deliberately
defers explicit table cells, spatial figure groups, and formula/operator
semantics to their specialized guards. To expose the diagnostic in a running
instance, set `SCI_RAG_ANSWER_VALIDATION=true`; the default is `false`. A
review notice is advisory only—the runtime does not rewrite or automatically
retry the answer. The current 106-row audit produced 36 structured-table,
4 spatial-figure, 6 formula/operator, 58 ordinary `ok` rows, and 2 review rows
(both `drugr-09`, the same omitted-process-marker signal across two repeats).

Phase 6.5 adds a bounded `build_evidence_retry_prompt()` helper for a possible
second pass. A historical `drugr-09` A/B showed no improvement because that
run's fixed retrieval context lacked the explicit `4,855` passage; repeating
generation cannot recover missing evidence. The historical trace predates the
new runtime/source provenance fields, so it is not proof that the current code
still misses the passage. A current single-case check retrieves the passage and
does not need retry. Automatic retry therefore remains disabled. See
`PHASE6_5_RETRY_GATE_HANDOFF.md`.

`evaluation/audit_generation_trace.py` provides an offline provenance check for
future repeated-generation traces. New rows from
`evaluation/generation_stability.py` include a secret-free runtime configuration
and source fingerprint, so context stability is not inferred from feature
switches alone. Historical traces without these fields are reported as
provenance-incomplete and must not be merged with later runs as if they used
identical code and model settings.

### Phase 6.7 candidate-pool audit

The retrieval benchmark now applies table/figure/parent-window controls
independently for every requested k. This prevents an @10 metric from changing
merely because @50 was included in the same command. Under the fixed five-paper
configuration (Hybrid + the pinned cross-encoder + routing + query
decomposition + parent/window + structured table/figure guards), the corrected
offline comparison was:

| context k | fact macro | fact micro | full cases | reference context | source page |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 0.936 | 0.932 | 47/53 (0.887) | 0.846 | 0.929 |
| 50 | 0.995 | 0.993 | 52/53 (0.981) | 0.899 | 1.000 |

Five of the six non-full @10 cases are recovered by the larger candidate
prefix; `mgno-02` still misses the formula surface `3×3` even at @50. Thus the
reranker candidate pool can contain useful tail evidence, but blindly passing
50 contexts to DeepSeek is not an acceptable default (about 2.8 seconds per
reranked query on CPU and roughly 2.5 GB peak RSS in this run). The unresolved
formula case requires representation/query handling, not another answer retry.
The full report is outside Git at
`/private/tmp/sci_rag_phase66_candidate_pool_10_50_fixed.json`.

### Phase 6.8 formula-evidence guard

`SCI_RAG_FORMULA_EVIDENCE` is an opt-in lexical guard for questions that explicitly ask for an
equation, formula, convolution-kernel size, or discretized-system form. It promotes a small set of
semantically related formula-bearing text blocks, caps noisy symbol matches from PDF extraction, and
never acts as a symbolic solver. Candidates are restricted to an existing unique source route (or a
single-source collection); an unqualified multi-paper query receives no formula injection. The
default is `false` because the multi-paper `mgno-02` case still needs source disambiguation or formula
representation normalization. The A/B report is outside Git at
`/private/tmp/sci_rag_phase68_formula_guard_scoped_v2.json`.

### Phase 7 direction audit

The current repository has no general multimodal, tool-calling, or Graph-RAG subsystem. PDF ingestion
deliberately uses `write_images=False`/`embed_images=False`; the optional figure path reads only
born-digital text-layer coordinates. The application has one OpenAI-compatible chat call but no tool
registry or executor, and no graph extraction/storage code. `PHASE7_DIRECTION_AUDIT.md` records the
evidence, entry gates, risks, and acceptance metrics. The recommended order is a small allow-listed
deterministic table/arithmetic tool experiment, conditional OCR/VLM only for an image-only failure set,
and Graph-RAG only after a demonstrated cross-document multi-hop need. The app already has a restricted
deterministic table-cell lookup path; no model-driven tool registry or executor is present, so any new
tool work should target a measured arithmetic/unit-conversion gap rather than duplicate table lookup.
No direction changes the default path yet.

### Phase 7.1 current-generation validation

The current Hybrid path was run end-to-end on all 53 benchmark questions against the isolated
502-chunk database with the pinned `BAAI/bge-reranker-base` revision, document routing, query
decomposition, parent/window and spatial text evidence enabled. All 53 calls completed without
errors. The deterministic answer audit reported required-fact macro/micro coverage of `0.9261/0.8973`
and complete coverage for `46/53` cases. This is lexical coverage only, not semantic correctness or
RAGAS quality. Provenance is complete for all rows; because this validation used one repeat, it does
not estimate answer-text stability. A limited `context_k=50` comparison recovered most candidate-pool
edge cases but did not consistently make the model state every tool/threshold, so the web default
remains `context_k=10`. Details and exact commands are in `PHASE7_1_CURRENT_GENERATION_HANDOFF.md`.

The benchmark manifest now declares safe equivalent surfaces for common formula/stride/translation
forms (`3 × 3`, Chinese “步长为 2”, “扩散模块”, and “闭合构象”). These aliases only prevent format
false negatives; they do not relax the gold facts or establish semantic correctness. No RAGAS run or
default configuration change was made.

### Phase 7.2 deterministic-tool gate

The 53-case benchmark contains no unit-conversion, arithmetic, or external-lookup case that would
justify a general tool-calling agent. The existing allow-listed table-cell path remains sufficient for
the structured questions. A generic table-caption unit safeguard now preserves shared scales such as
`×10⁻²` in deterministic answers and returned traces; the full offline suite is 145/145. Tool calling
is deferred until a separately curated set has at least 20 real operation questions and at least five
stable baseline failures that a local allow-listed operation can repair. See
`PHASE7_2_TOOL_AUDIT.md`.

### Phase 7.3 multimodal / Graph-RAG gate

The current 53-case set has no image-only failure set or demonstrated cross-document multi-hop need.
The existing born-digital figure-coordinate path is therefore retained, while OCR/VLM and Graph-RAG
are deferred. `PHASE7_3_DIRECTION_GATE.md` records the entry conditions: at least 10 manually checked
image-only/complex-figure cases for multimodal work, or at least five stable multi-hop failures for a
provenance-first graph experiment. Neither direction changes the default path at this stage.

### Phase 8 integration handoff

The latest integration regression is complete: 145/145 offline tests, compilation, and
`git diff --check` pass. The current Hybrid+fixed-reranker generation run completed 53/53 cases
without errors, with complete provenance for every row. The result remains a lexical diagnostic
(`0.9261/0.8973` fact macro/micro and `46/53` full), not a semantic accuracy or generalization claim.
`PHASE8_INTEGRATION_HANDOFF.md` lists the exact review commands and stop conditions. No push was
performed.

### Phase 9 semantic answer review

The Phase 7.1 Hybrid + fixed-reranker answers were reviewed against the 53 benchmark gold
contexts. The review is an internal single-reviewer/agent pass, not an independent double-blind
annotation or a replacement for RAGAS. It classified `43 correct / 7 partial / 3 incorrect`.
The lexical audit was higher (`0.9261/0.8973` fact macro/micro and `46/53` full), demonstrating
that string-level fact coverage can overstate semantic correctness: three answers contained a
correct number but then contradicted the requested table, and three were unsupported refusals.

The non-correct cases are recorded with rationale in `PHASE9_SEMANTIC_REVIEW_HANDOFF.md`; the
answer JSONL and review labels remain outside Git under `/private/tmp`. The next experiment is a
targeted offline diagnosis of `scidqa-07`, `mgno-03`, and `mgno-04` (retrieval tail versus context
compression versus refusal), followed by controlled prompt/context A/B for the partial cases.
No default context-k increase or new OCR/VLM, tool-agent, or Graph-RAG subsystem is justified by
this review alone.

### Phase 10 targeted generation repair

The generation prompt now prevents a contradictory refusal after a table value has already been
found and requires checking later algorithm/appendix fragments before refusing. Table chunks also
show parser metadata such as `[表格，Table 2]` in the generation label. In a controlled rerun,
SciDQA `scidqa-03` and `scidqa-04` both answered the requested Table 2 values without the prior
contradiction.

The existing opt-in `SCI_RAG_FORMULA_EVIDENCE=true` gate now covers explicit multigrid algorithm
questions about initialization, residual smoothing, restriction/prolongation, stride, and cycle
types. With the same isolated 502-chunk database and `context_k=10`, `mgno-03` and `mgno-04`
both produced full required-fact answers; this remains a two-case A/B, not a full-benchmark
semantic result. The switch is still `false` by default, and `context_k` remains 10. See
`PHASE10_TARGETED_REPAIR_HANDOFF.md` for commands, outputs, and the remaining `scidqa-07`
candidate-pool failure.

### Phase 11 routed method-section repair

The bounded same-section expansion now accepts a unique source selected by the
existing conservative document router. It still refuses to expand an
ambiguous multi-paper query, keeps the source boundary, caps additions at six
chunks, and only activates for composite/list-style questions. The section
query aliases include experimental setup/configuration terms, so a method
question such as “SciDQA 的四种实验配置是什么？” can recover the nearby
appendix list without changing the normal top-k ranking or global context cap.

On the fixed 502-chunk isolated database, the earlier `scidqa-07` `context_k=10`
run had only a truncated Experimental Setup lead-in and produced an unsupported
refusal. With the route-aware section expansion, the same fixed Hybrid + pinned
reranker configuration returned the B.1 list containing closed-book,
title-abs, RAG, and full-text; the answer audit was `1/1` full and manual
inspection found no cross-paper context. This is a targeted single-case
generation check, not a 53-case semantic accuracy claim. Offline regression is
148/148 tests; the route-aware behavior is covered by benchmark and runtime
tests. See `PHASE11_METHOD_EVIDENCE_HANDOFF.md`.

### Phase 12 full-generation gate

The fixed 502-chunk isolated database was run twice through the current
Hybrid+reranker path (106/106 DeepSeek calls, zero API errors). The two
repeats both reached `47/53` full lexical answer-fact coverage; context and
metadata were stable for `50/53` cases, while normalized answer text was
identical for `19/53`. `scidqa-07` stayed full in both repeats, but one other
method case changed fact status between repeats, so this remains a generation
stability diagnostic rather than a semantic accuracy claim. The default
retrieval mode, context cap, and opt-in evidence switches remain unchanged.
See `PHASE12_FULL_GENERATION_GATE_HANDOFF.md` for exact outputs and stop
conditions.

### Phase 7.3 multimodal / Graph-RAG gate

The current 53-case set has no image-only failure set or demonstrated cross-document multi-hop need.
The existing born-digital figure-coordinate path is therefore retained, while OCR/VLM and Graph-RAG
are deferred. `PHASE7_3_DIRECTION_GATE.md` records the entry conditions: at least 10 manually checked
image-only/complex-figure cases for multimodal work, or at least five stable multi-hop failures for a
provenance-first graph experiment. Neither direction changes the default path at this stage.
