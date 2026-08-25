import os
import ssl
import random
import re
import tempfile
import shutil
from dotenv import load_dotenv
load_dotenv()
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
ssl._create_default_https_context = ssl._create_unverified_context

import gradio as gr
from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)
from langchain_community.document_loaders import TextLoader, Docx2txtLoader
from sentence_transformers import SentenceTransformer
import chromadb
import uuid
from openai import OpenAI

# ---------- 初始化 ----------
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    raise ValueError("请在 .env 文件中设置 DEEPSEEK_API_KEY")

client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com/v1")
embedding_model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(
    name="knowledge_base",
    metadata={"hnsw:space": "cosine"}
)

# ---------- 文档处理 ----------
# GFM 表格分隔行正则：匹配 "| --- | --- |" 结构
_TABLE_SEP_RE = re.compile(r"\|(?:\s*:?-{3,}:?\s*\|)+")
# 备用正则：暴力提取 "Table N ..." 段落
_TABLE_CAPTION_RE = re.compile(r"\bTable\s*\d+.*?(?=\n\n|\Z)", re.DOTALL | re.MULTILINE)
# 若设为 True，GFM 表格与暴力提取结果同时入库
_ALWAYS_BRUTE_FORCE = False


def extract_tables(markdown_text, source):
    """
    提取 Markdown 中的表格，返回 List[Document]。
    优先匹配标准 GFM 表格（| --- |），若无则用正则暴力提取 "Table N" 段落。
    每个表格块 metadata 包含 type="table"、source、table_caption。
    """
    tables = []
    lines = markdown_text.split("\n")

    # 方案 A：标准 GFM 表格逐行解析
    i, n = 0, len(lines)
    while i < n:
        if not _TABLE_SEP_RE.fullmatch(lines[i].strip()):
            i += 1
            continue
        header_line, header_idx = "", i - 1
        if header_idx >= 0:
            prev = lines[header_idx].strip()
            if prev.startswith("|") and prev.endswith("|") and not _TABLE_SEP_RE.fullmatch(prev):
                header_line = lines[header_idx]
            else:
                header_idx = i
        rows, j = [], i + 1
        while j < n:
            cand = lines[j].strip()
            if not (cand.startswith("|") and cand.endswith("|")):
                break
            if _TABLE_SEP_RE.fullmatch(cand):
                break
            rows.append(lines[j])
            j += 1
        if not rows:
            i += 1
            continue
        caption = ""
        scan = header_idx - 1
        for _ in range(3):
            if scan < 0:
                break
            cand = lines[scan].strip()
            if cand and re.search(r"table|表\s*\d|表\d", cand, re.IGNORECASE):
                caption = cand
                break
            scan -= 1
        tables.append(Document(
            page_content="\n".join([header_line, lines[i]] + rows),
            metadata={"type": "table", "source": source,
                      "table_caption": caption or "未命名表格"},
        ))
        i = j

    # 方案 B：备用暴力提取
    if not tables or _ALWAYS_BRUTE_FORCE:
        existing = "\n".join(t.page_content for t in tables)
        for m in _TABLE_CAPTION_RE.finditer(markdown_text):
            block = m.group(0).strip()
            if not block or (tables and block in existing):
                continue
            first_line = block.split("\n", 1)[0]
            tables.append(Document(
                page_content=block,
                metadata={"type": "table", "source": source,
                          "table_caption": first_line},
            ))
        if tables and not any(_TABLE_SEP_RE.fullmatch(ln.strip()) for ln in lines):
            print(f"[表格提取] 未发现 GFM 表格，已用备用正则暴力提取 {len(tables)} 个 'Table N' 段落。")

    return tables


def _split_to_chunks(documents, source):
    """
    将文档切分为最终存储块。
    - 表格块（type="table"）保持完整，不切分。
    - 非表格文本：优先按标题层级切分，超长块回退为段落/句子级切分。
    """
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "H1"), ("##", "H2"), ("###", "H3"), ("####", "H4")],
        strip_headers=False,
    )
    fallback_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=128,
        length_function=len,
        separators=["\n\n", "\n", "。", "；"],
    )

    final_chunks = []
    for doc in documents:
        table_chunks = extract_tables(doc.page_content, source)

        for hc in header_splitter.split_text(doc.page_content):
            meta = {"source": source, "type": "text"}
            if isinstance(hc, Document):
                header_parts = [f"{lv}: {hc.metadata[lv]}"
                                for lv in ("H1", "H2", "H3", "H4") if lv in hc.metadata]
                meta["headers"] = " > ".join(header_parts)
                content = hc.page_content
            elif isinstance(hc, (list, tuple)):
                meta["headers"] = hc[0] or ""
                content = hc[1]
            else:
                content = hc
            if len(content) > 1024:
                for sc in fallback_splitter.split_text(content):
                    final_chunks.append(Document(page_content=sc, metadata=dict(meta)))
            else:
                final_chunks.append(Document(page_content=content, metadata=meta))

        final_chunks.extend(table_chunks)

    return final_chunks


def load_and_split_document(file_path):
    """
    文档加载管线：
      - PDF：用 pymupdf4llm 转 Markdown，保留表格结构和 LaTeX 公式。
      - 图片提取为占位符 [Image: xxx.png]。
      - 调用 extract_tables 和 _split_to_chunks 完成切分。
    """
    source = os.path.basename(file_path)
    suffix = os.path.splitext(file_path)[1].lower()

    if suffix == ".pdf":
        try:
            import pymupdf4llm
            import pymupdf
        except ImportError:
            raise ImportError("缺少 pymupdf4llm 依赖。请执行: pip install pymupdf4llm pillow")

        doc = pymupdf.open(file_path)
        markdown_text = pymupdf4llm.to_markdown(doc)

        # 调试信息：打印 Markdown 前 3000 字符及 GFM 表格数量
        print("=" * 70)
        print(f"[DEBUG] pymupdf4llm 转换结果前 3000 字符（{source}）：")
        print(markdown_text[:3000])
        gfm_count = sum(1 for ln in markdown_text.split("\n") if _TABLE_SEP_RE.fullmatch(ln.strip()))
        print(f"[DEBUG] 是否含 GFM 表格分隔行：{gfm_count > 0}（共 {gfm_count} 个）")
        print("=" * 70)

        image_dir = None
        images_saved = 0
        for page_index in range(len(doc)):
            page = doc[page_index]
            for block_index, block in enumerate(page.get_text("blocks")):
                if block[6] != 1:
                    continue
                try:
                    img_info = doc.extract_image(block[7])
                    if image_dir is None:
                        image_dir = tempfile.mkdtemp(prefix="sci_rag_images_")
                    fname = f"page_{page_index + 1}_img_{block_index + 1}.{img_info.get('ext', 'png')}"
                    with open(os.path.join(image_dir, fname), "wb") as f:
                        f.write(img_info["image"])
                    markdown_text += f"\n\n[Image: {fname}]\n"
                    images_saved += 1
                except Exception:
                    continue
        doc.close()
        if image_dir is not None and images_saved == 0:
            shutil.rmtree(image_dir, ignore_errors=True)
            image_dir = None

        documents = [Document(page_content=markdown_text, metadata={"source": source})]
        try:
            return _split_to_chunks(documents, source)
        finally:
            if image_dir:
                shutil.rmtree(image_dir, ignore_errors=True)

    elif suffix == ".txt":
        documents = TextLoader(file_path, encoding="utf-8").load()
    elif suffix == ".docx":
        documents = Docx2txtLoader(file_path).load()
    else:
        raise ValueError("不支持的文件类型（仅支持 .pdf / .txt / .docx）")

    for d in documents:
        d.metadata["source"] = source
    return _split_to_chunks(documents, source)


def add_document_to_db(file_path):
    chunks = load_and_split_document(file_path)
    for chunk in chunks:
        chunk_id = str(uuid.uuid4())
        text = chunk.page_content
        metadata = dict(chunk.metadata)
        metadata.setdefault("source", os.path.basename(file_path))
        embedding = embedding_model.encode(text).tolist()
        collection.add(
            ids=[chunk_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[metadata]
        )
    return f"✅ 成功添加 {len(chunks)} 个文本块到知识库，当前共 {collection.count()} 个。"


def upload_file(file):
    if file is None:
        return "请选择一个文件"
    return add_document_to_db(file.name)


# ---------- 核心问答 ----------
_TABLE_QUESTION_KEYWORDS = ["table", "表", "数值", "多少", "样本量", "比率", "n="]
_ENTITY_RE = re.compile(r"[A-Za-z0-9_*+\-]+(?:[-\s][A-Za-z0-9_*+\-]+)*")

SCIENTIFIC_SYSTEM_PROMPT = """你是一个面向科学论文的严谨学术问答引擎（Sci-RAG），职责是从给定的参考片段中抽取事实、数值与实验方法论。必须遵守以下规则：

【强制规则 1：数值必须原样引用并指明出处】
- 若参考文本中存在具体数值（如 300K、2.5MPa、p<0.05、N=42），回答时必须原样引用该数值，不得四舍五入、不得改写、不得推算。
- 引用数值后必须指明出处，格式为：根据参考片段 [X] 所示（X 为片段的编号）。

【强制规则 2：趋势判断必须有明确对比依据】
- 若用户问题涉及趋势判断（如升高、降低、催化活性增强、显著优于等），必须确认参考文本中有明确的对比依据。
- 若参考文本未提供该趋势的明确对比依据，必须原话回复："资料未提供该趋势的明确依据，无法推测。"

【强制规则 3：实验步骤按时间顺序重组】
- 若回答涉及实验步骤或方法流程，请按"第一、第二、第三"的逻辑重组叙述，不得调换核心操作的时间顺序，不得省略中间步骤。

【强制规则 4：实体联合约束】
- 若用户问题同时指定了表格编号和行/列实体名称，回答时必须将两者视为联合约束条件。
- 若同一实体名称出现在多个表格中，优先采用用户指定编号的表格中的数据。
- 若在指定表格中找不到该实体名称，回复："在 Table [编号] 中未找到 [实体名称] 的条目。"
- 严禁跨表取数——禁止使用 Table 1 的数据回答关于 Table 2 的问题。

【其他要求】
- 表格以 Markdown 形式给出，数值类问题请直接依据表格行列作答。
- [Image: xxx.png] 是图片占位标记，若问题涉及图片内容，请说明"该图内容未纳入文本检索范围"。
- 若参考片段无法回答该问题，请如实说明"资料未提供相关信息"，严禁编造。"""


def _filter_table_rows_by_entity(content, entity):
    """
    行级过滤：只保留包含指定实体名的数据行及表头。
    用于在表格块内部进一步缩小范围，避免多行干扰。
    """
    lines = content.split("\n")
    sep_idx = None
    for i, ln in enumerate(lines):
        if _TABLE_SEP_RE.fullmatch(ln.strip()):
            sep_idx = i
            break
    if sep_idx is not None:
        header, data = lines[:sep_idx + 1], lines[sep_idx + 1:]
    else:
        header, data = lines[:1], lines[1:]
    entity_low = entity.lower()
    matched_rows = [ln for ln in data if entity_low in ln.lower()]
    if not matched_rows:
        return None
    return "\n".join(header + matched_rows)


def _rerank_table_first(question, texts, metas):
    """
    重排序：
      - 非表格类提问：保持原序。
      - 表格类提问：将 type="table" 的块置顶。
      - 若用户指定了 Table N，只保留命中的表格块，排除其他表格块。
      - 若提取到实体名，进一步行级过滤。
    返回 (order, note)。
    """
    q = question.lower()
    if not any(kw in q for kw in _TABLE_QUESTION_KEYWORDS):
        return list(range(len(texts))), ""

    table_idx = [i for i, m in enumerate(metas) if m.get("type") == "table"]
    note = ""

    m = re.search(r"table\s*(\d+)", q)
    if m:
        table_num = m.group(1)
        table_pat = re.compile(rf"Table\s*{table_num}", re.IGNORECASE)
        matched = []
        for i in table_idx:
            caption = str(metas[i].get("table_caption", ""))
            headers = str(metas[i].get("headers", ""))
            if table_pat.search(caption) or table_pat.search(texts[i]) or table_pat.search(headers):
                matched.append(i)

        if matched:
            table_idx = matched
            entities = _ENTITY_RE.findall(question)
            candidates = [e for e in entities
                          if not re.fullmatch(r"table\s*\d+", e, re.IGNORECASE) and not e.isdigit()]
            entity = max(candidates, key=len) if candidates else ""
            if entity:
                for i in table_idx:
                    filtered = _filter_table_rows_by_entity(texts[i], entity)
                    if filtered is not None:
                        texts[i] = filtered
        else:
            note = f"未找到 Table {table_num} 的精确匹配，以下是知识库中所有表格数据供参考。"

    other_idx = [i for i, m in enumerate(metas) if m.get("type") != "table"]
    return table_idx + other_idx, note


def query_knowledge(message, history=None, return_contexts=True):
    """
    核心问答入口（Gradio 聊天界面与 RAGAS 评估脚本共用）。

    参数：
      message         用户问题
      history         Gradio 传入的历史消息（本实现不使用，保留以兼容 ChatInterface）
      return_contexts 为 True 时返回 dict：{"answer": 原始回答文本（不含参考来源页脚）,
                      "contexts": 实际进入提示词的检索上下文列表（重排序/过滤后，
                      与【片段 X】编号一一对应）}，供 RAGAS 等评估框架使用；
                      为 False 时返回带参考来源页脚的纯文本回答（UI 展示用）。
    """
    if not message:
        if return_contexts:
            return {"answer": "请输入一个问题。", "contexts": []}
        return "请输入一个问题。"
    if collection.count() == 0:
        if return_contexts:
            return {"answer": "📚 知识库为空，请先上传文档。", "contexts": []}
        return "📚 知识库为空，请先上传文档。"

    question_embedding = embedding_model.encode(message).tolist()
    results = collection.query(query_embeddings=[question_embedding], n_results=60)
    retrieved_texts = results['documents'][0] if results['documents'] else []
    if not retrieved_texts:
        if return_contexts:
            return {"answer": "未找到相关内容，请换个问法。", "contexts": []}
        return "未找到相关内容，请换个问法。"
    retrieved_metas = results['metadatas'][0] if results.get('metadatas') else []

    order, note = _rerank_table_first(message, retrieved_texts, retrieved_metas)
    ordered_texts = [retrieved_texts[i] for i in order]
    ordered_metas = [retrieved_metas[i] for i in order]

    # 二次过滤：若用户指定了 Table N，只保留 caption 中包含该编号的表格块
    table_filter_note = ""
    num_match = re.search(r"Table\s*(\d+)", message, re.IGNORECASE)
    if num_match:
        table_num = num_match.group(1)
        table_pat = re.compile(rf"Table\s*{table_num}(?!\d)", re.IGNORECASE)
        keep_idx = [
            i for i, m in enumerate(ordered_metas)
            if m.get("type") != "table" or table_pat.search(str(m.get("table_caption", "")))
        ]
        if any(ordered_metas[i].get("type") == "table" for i in keep_idx):
            ordered_texts = [ordered_texts[i] for i in keep_idx]
            ordered_metas = [ordered_metas[i] for i in keep_idx]
            table_filter_note = f"已根据您的要求只检索 Table {table_num} 的数据。"

    context_parts = []
    for idx, (t, m) in enumerate(zip(ordered_texts, ordered_metas), start=1):
        label = f"【片段 {idx}】[表格]" if m.get("type") == "table" else f"【片段 {idx}】"
        context_parts.append(f"{label}\n{t}")
    context = "\n\n---\n\n".join(context_parts)
    if note:
        context = f"【检索提示】{note}\n\n" + context
    if table_filter_note:
        context = f"【检索提示】{table_filter_note}\n\n" + context

    user_prompt = f"【参考资料】\n{context}\n\n【问题】\n{message}"

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": SCIENTIFIC_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=2048
        )
        answer = response.choices[0].message.content
        if return_contexts:
            # 评估模式：返回原始 answer（不带参考来源页脚，避免干扰指标计算）
            # + 检索上下文列表（ordered_texts 即最终进入提示词的上下文，
            # 已含表格重排序与行级实体过滤结果）
            return {"answer": answer, "contexts": ordered_texts}
        sources = [
            f"- {m.get('source', '未知')}{'（表格）' if m.get('type') == 'table' else ''}"
            for m in ordered_metas
        ]
        # 去重但保留顺序（防止显示大量重复的文件名）
        unique_sources = list(dict.fromkeys(sources))
        return answer + "\n\n📌 **参考来源：**\n" + "\n".join(unique_sources)
    except Exception as e:
        if return_contexts:
            return {"answer": f"❌ 调用出错：{str(e)}", "contexts": ordered_texts}
        return f"❌ 调用出错：{str(e)}"


# ---------- 功能 1：生成学习大纲 ----------
def generate_mindmap():
    if collection.count() == 0:
        return "📚 知识库为空，请先上传文档。"
    all_chunks = collection.get()
    if not all_chunks or not all_chunks['documents']:
        return "无法读取文档内容。"
    context = "\n\n".join(all_chunks['documents'][:15])
    system_prompt = "你是一位顶级学术助教。请根据提供的课程资料，生成一份层级清晰、结构完整的学习大纲。"
    user_prompt = f"""请基于以下资料生成一份Markdown格式的层级大纲（使用 # ## ### - 表示层级），不要包含任何开场白或结尾总结，直接输出大纲结构。

资料内容：
{context}
"""
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=2000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ 生成大纲失败：{str(e)}"


# ---------- 功能 2：智能出题 ----------
def generate_quiz():
    if collection.count() == 0:
        return "📚 知识库为空，请先上传文档。"
    all_chunks = collection.get()
    if not all_chunks or not all_chunks['documents']:
        return "无法读取文档内容。"
    sample_chunks = random.sample(all_chunks['documents'], min(8, len(all_chunks['documents'])))
    context = "\n\n".join(sample_chunks)
    system_prompt = "你是一个严谨的大学教师。请根据资料出5道单项选择题，用于考察学生对知识的掌握程度。"
    user_prompt = f"""请根据以下资料，生成5道单项选择题。
输出格式要求（严格遵守）：
第1题：[题目内容]
A. [选项A] B. [选项B] C. [选项C] D. [选项D]
答案：X
解析：[详细解释为什么选这个]

资料内容：
{context}
"""
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.4,
            max_tokens=2000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ 出题失败：{str(e)}"


# ---------- Gradio 界面 ----------
def chat_respond(message, history):
    """Gradio 聊天包装：UI 始终返回纯文本回答。
    query_knowledge 的 return_contexts 默认为 True（返回 dict，供评估脚本使用），
    聊天界面需要显式关闭以获得带参考来源页脚的字符串形式。"""
    return query_knowledge(message, history, return_contexts=False)


with gr.Blocks(title="AI 大学生学习工作台", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎓 AI 大学生学习工作台")
    gr.Markdown("上传你的课件或论文，用 AI 帮你学！")

    with gr.Tab("📤 上传文档"):
        file_input = gr.File(label="选择文档", file_types=[".pdf", ".txt", ".docx"])
        upload_output = gr.Textbox(label="上传状态", lines=3)
        upload_button = gr.Button("添加到知识库")
        upload_button.click(upload_file, inputs=file_input, outputs=upload_output)
        gr.Markdown(f"**当前知识库文本块数：** {collection.count()}")

    with gr.Tab("💬 智能问答"):
        gr.ChatInterface(
            fn=chat_respond,
            title="📖 基于文档的问答",
            description="输入问题，AI会从已上传的文档中检索答案。",
            chatbot=gr.Chatbot(height=450),
            textbox=gr.Textbox(placeholder="例如：这篇论文的核心创新点是什么？", scale=7)
        )

    with gr.Tab("🧠 生成学习大纲"):
        gr.Markdown("### 一键生成层级学习大纲（自动转为脑图结构）")
        btn_mindmap = gr.Button("🚀 生成大纲与脑图")
        output_mindmap = gr.Markdown(label="📋 大纲内容", value="点击上方按钮生成...")
        btn_mindmap.click(generate_mindmap, inputs=[], outputs=output_mindmap)

    with gr.Tab("📝 智能出题"):
        gr.Markdown("### 基于当前知识库自动生成练习题（含解析）")
        btn_quiz = gr.Button("📝 生成5道选择题")
        output_quiz = gr.Markdown(label="📋 题目与解析", value="点击上方按钮生成...")
        btn_quiz.click(generate_quiz, inputs=[], outputs=output_quiz)

# ---------- 启动 ----------
if __name__ == "__main__":
    demo.launch()