import os
import ssl
import random
import re                # 新增：表格提取 / Table N 关键词正则
import tempfile          # 新增：图片占位符临时目录
import shutil            # 新增：临时目录清理
from dotenv import load_dotenv
load_dotenv()
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
ssl._create_default_https_context = ssl._create_unverified_context

import gradio as gr
from langchain_core.documents import Document   # 新增：独立表格块用 Document 承载
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,                 # 新增：标题级切分
)
# 注意：不再使用 PyPDFLoader（乱码根源），PDF 解析改用 pymupdf4llm（函数内延迟导入）
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

# ---------- 文档处理（Sci-RAG 改造：pymupdf4llm + 表格抽块 + 标题级切分） ----------

# GFM 表格分隔行正则：整行形如 "| --- | --- |"（允许 :---: 对齐语法）。
# 每个单元格必须是 3 个以上连字符，防止表头/数据行里的单个 "-" 被误判
_TABLE_SEP_RE = re.compile(r"\|(?:\s*:?-{3,}:?\s*\|)+")

# 备用正则（修改 2）：暴力提取 "Table N ..." 连续段落，直到空行或文末
_TABLE_CAPTION_RE = re.compile(r"\bTable\s*\d+.*?(?=\n\n|\Z)", re.DOTALL | re.MULTILINE)

# 若 PDF 里"部分表格转成了 GFM、部分没有"（混合情况），把下面开关改为 True，
# GFM 表格与暴力提取的 Table N 段落会同时入库（重复内容自动跳过）
_ALWAYS_BRUTE_FORCE = False


def extract_tables(markdown_text, source):
    """
    修改 2：表格提取函数（GFM 正则优先 + "Table N" 暴力备选）。

    方案 A：若存在标准 GFM 表格（| --- | 分隔行），逐行解析：
            分隔行上一行是表头，下面连续的管道行是数据行，整表作为独立块；
    方案 B：若没有任何 GFM 表格（pymupdf4llm 对复杂表格转换失败时），
            用 r'\\bTable\\s*\\d+.*?(?=\\n\\n|\\Z)' 暴力提取所有包含
            "Table" 的连续段落，确保表格内容不会被漏掉。

    返回 List[Document]，每个表格块 metadata 强制带
    {"type": "table", "source": source, "table_caption": 标题}。
    """
    tables = []
    lines = markdown_text.split("\n")

    # ---- 方案 A：标准 GFM 表格逐行解析 ----
    i, n = 0, len(lines)
    while i < n:
        if not _TABLE_SEP_RE.fullmatch(lines[i].strip()):
            i += 1
            continue
        # 分隔行的上一行即表头行
        header_line, header_idx = "", i - 1
        if header_idx >= 0:
            prev = lines[header_idx].strip()
            if prev.startswith("|") and prev.endswith("|") and not _TABLE_SEP_RE.fullmatch(prev):
                header_line = lines[header_idx]
            else:
                header_idx = i  # 无表头的畸形表格
        # 向下连续取数据行（以 | 开头、以 | 结尾，且不是分隔行）
        rows, j = [], i + 1
        while j < n:
            cand = lines[j].strip()
            if not (cand.startswith("|") and cand.endswith("|")):
                break  # 遇到非表格行，数据行结束
            if _TABLE_SEP_RE.fullmatch(cand):
                break  # 遇到下一个表格的分隔行
            rows.append(lines[j])
            j += 1
        if not rows:
            i += 1
            continue  # 只有表头没有数据的空表格（论文模板），跳过
        # 向上找表格标题（Table N / 表 N），用于溯源
        caption = ""
        scan = header_idx - 1
        for _ in range(3):
            if scan < 0:
                break
            cand = lines[scan].strip()
            if cand:
                if re.search(r"table|表\s*\d|表\d", cand, re.IGNORECASE):
                    caption = cand
                break
            scan -= 1
        tables.append(Document(
            page_content="\n".join([header_line, lines[i]] + rows),
            metadata={"type": "table", "source": source,
                      "table_caption": caption or "未命名表格"},
        ))
        i = j

    # ---- 方案 B：备用暴力提取（无 GFM 表格时启用，或开关强制开启） ----
    if not tables or _ALWAYS_BRUTE_FORCE:
        existing = "\n".join(t.page_content for t in tables)
        for m in _TABLE_CAPTION_RE.finditer(markdown_text):
            block = m.group(0).strip()
            if not block:
                continue
            if tables and block in existing:
                continue  # 该段落已被 GFM 方案覆盖，跳过
            first_line = block.split("\n", 1)[0]
            tables.append(Document(
                page_content=block,
                metadata={"type": "table", "source": source,
                          "table_caption": first_line},
            ))
        if tables and not any(_TABLE_SEP_RE.fullmatch(ln.strip())
                              for ln in lines):
            print(f"[表格提取] 未发现 GFM 表格，已用备用正则暴力提取 "
                  f"{len(tables)} 个 'Table N' 段落。")

    return tables


def _split_to_chunks(documents, source):
    """
    修改 3 所在位置：最终分块循环。

    - 表格块（metadata.type == "table"）：跳过 RecursiveCharacterTextSplitter，
      整个表格文本作为一个 chunk 直接存入向量库，防止长表格被切碎；
    - 非表格文本：MarkdownHeaderTextSplitter 优先（按 #/##/###/#### 标题切分），
      超过 1024 字符的块回退 RecursiveCharacterTextSplitter
      (chunk_size=1024, chunk_overlap=128, separators=["\\n\\n", "\\n", "。", "；"])。
    """
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "H1"), ("##", "H2"), ("###", "H3"), ("####", "H4")],
        strip_headers=False,  # 块内保留标题行，embedding 时带上章节语义
    )
    fallback_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,          # 扩大窗口以容纳长内容
        chunk_overlap=128,        # 相邻块重叠，避免句间语义断裂
        length_function=len,
        # 故意不含英文句点/逗号与 "|"，防止切断表格行列结构
        separators=["\n\n", "\n", "。", "；"],
    )

    final_chunks = []
    for doc in documents:
        # 1) 表格抽块：独立 Document，metadata 带 type="table"（修改 2）
        table_chunks = extract_tables(doc.page_content, source)

        # 2) 非表格正文：标题级切分优先
        for hc in header_splitter.split_text(doc.page_content):
            meta = {"source": source, "type": "text"}
            if isinstance(hc, Document):
                # 各级标题在 hc.metadata 的 H1~H4 键里，归一化成路径字符串
                header_parts = [f"{lv}: {hc.metadata[lv]}"
                                for lv in ("H1", "H2", "H3", "H4")
                                if lv in hc.metadata]
                meta["headers"] = " > ".join(header_parts)
                content = hc.page_content
            elif isinstance(hc, (list, tuple)):
                # 旧版 langchain 返回 (headers, content) 元组
                meta["headers"] = hc[0] or ""
                content = hc[1]
            else:
                content = hc
            # 3) 超长正文回退到段落/句子级切分
            if len(content) > 1024:
                for sc in fallback_splitter.split_text(content):
                    final_chunks.append(Document(page_content=sc, metadata=dict(meta)))
            else:
                final_chunks.append(Document(page_content=content, metadata=meta))

        # 4) 修改 3：表格块不再经过 RecursiveCharacterTextSplitter，
        #    整表作为一个 chunk 直接入库（长表格也不切碎）
        final_chunks.extend(table_chunks)

    return final_chunks


def load_and_split_document(file_path):
    """
    Sci-RAG 文档处理管线：
      1) PDF 用 pymupdf4llm 转 Markdown（保留 | --- | 表格结构与 LaTeX 公式）；
      2) 提取无法解析的图片为 PNG，正文标注 [Image: xxx.png] 占位符；
      3) extract_tables 表格抽块（修改 2）+ 标题级切分（修改 3）。
    """
    source = os.path.basename(file_path)
    suffix = os.path.splitext(file_path)[1].lower()

    if suffix == ".pdf":
        # 延迟导入：pymupdf4llm 未安装时，txt/docx 文档仍然可用
        try:
            import pymupdf4llm
            import pymupdf  # pymupdf4llm 底层依赖，用于图片提取
        except ImportError:
            raise ImportError(
                "缺少 pymupdf4llm 依赖。请执行: pip install pymupdf4llm pillow")

        doc = pymupdf.open(file_path)
        markdown_text = pymupdf4llm.to_markdown(doc)

        # ---------- 修改 1：调试打印（解析完 PDF 之后、文本切分之前） ----------
        print("=" * 70)
        print(f"[DEBUG] pymupdf4llm 转换结果前 3000 字符（{source}）：")
        print(markdown_text[:3000])
        gfm_count = sum(1 for ln in markdown_text.split("\n")
                        if _TABLE_SEP_RE.fullmatch(ln.strip()))
        print(f"[DEBUG] 是否含 GFM 表格分隔行 (| --- |)：{gfm_count > 0}"
              f"（共 {gfm_count} 个）")
        print("=" * 70)

        # 提取图片占位符（无法解析的图/复杂公式原样保留为图片）
        image_dir = None
        images_saved = 0
        for page_index in range(len(doc)):
            page = doc[page_index]
            # page.get_text("blocks") 返回块列表，类型 1 表示"图片块"
            for block_index, block in enumerate(page.get_text("blocks")):
                if block[6] != 1:  # 仅处理图片块，跳过文本块
                    continue
                try:
                    img_info = doc.extract_image(block[7])  # 图片 xref
                    if image_dir is None:
                        image_dir = tempfile.mkdtemp(prefix="sci_rag_images_")
                    fname = f"page_{page_index + 1}_img_{block_index + 1}." \
                            f"{img_info.get('ext', 'png')}"
                    with open(os.path.join(image_dir, fname), "wb") as f:
                        f.write(img_info["image"])
                    # 在 Markdown 文本中追加图片占位符标记
                    markdown_text += f"\n\n[Image: {fname}]\n"
                    images_saved += 1
                except Exception:
                    continue  # 个别图片提取失败不影响整体流程
        doc.close()
        if image_dir is not None and images_saved == 0:
            shutil.rmtree(image_dir, ignore_errors=True)
            image_dir = None

        documents = [Document(page_content=markdown_text, metadata={"source": source})]

        try:
            return _split_to_chunks(documents, source)
        finally:
            # 占位符文本已存入块内容，临时图片目录用完即清理
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
        # 完整保留块元数据：source / headers / type(=table|text) / table_caption，
        # 供检索后的重排序按 type 区分表格块
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

# ---------- 核心问答（Sci-RAG 改造：n_results=6 + 表格优先重排序 + 严谨 Prompt） ----------

# 修改 4：数值/表格类提问关键词（命中即触发表格块置顶）
_TABLE_QUESTION_KEYWORDS = ["table", "表", "数值", "多少", "样本量", "比率", "n="]

# 修改 6：实体名提取正则（行级过滤用）。匹配 ASCII 实体记号——
# 字母/数字/下划线/*/+/-，允许空格或连字符连接多段（如 "DrugR*"、
# "CO2 methanation"、"Ni-Fe"）；中文会被自然截断。
_ENTITY_RE = re.compile(r"[A-Za-z0-9_*+\-]+(?:[-\s][A-Za-z0-9_*+\-]+)*")

# 科学严谨模式 System Prompt（防幻觉三规则）
SCIENTIFIC_SYSTEM_PROMPT = """你是一个面向科学论文的严谨学术问答引擎（Sci-RAG），职责是从给定的参考片段中抽取事实、数值与实验方法论。你必须恪守以下三条强制规则：

【强制规则 1：数值必须原样引用并指明出处】
- 若参考文本中存在具体数值（如 300K、2.5MPa、p<0.05、N=42），回答时必须原样引用该数值，不得四舍五入、不得改写、不得推算。
- 引用数值后必须指明出处，格式为：根据参考片段 [X] 所示（X 为片段的编号）。

【强制规则 2：趋势判断必须有明确对比依据】
- 若用户问题涉及趋势判断（如升高、降低、催化活性增强、显著优于等），必须确认参考文本中有明确的对比依据（如两个条件下的数值对比、统计检验结果）。
- 若参考文本未提供该趋势的明确对比依据，必须原话回复："资料未提供该趋势的明确依据，无法推测。"严禁臆测或编造趋势。

【强制规则 3：实验步骤按时间顺序重组】
- 若回答涉及实验步骤或方法流程，请按"第一、第二、第三"的逻辑重组叙述，不得调换核心操作的时间顺序，不得省略中间步骤。
- 实验方法回答必须落到参考文本中出现的具体名词（如算法名称、温度条件、溶剂），不得泛泛而谈。

【强制规则 4：精确匹配表号与行名】
- 如果用户问题中明确指定了表格编号（如 "Table 2"）和行名（如 "DrugR*"），你必须首先在参考片段中找到同时包含该表号和该行名的表格行，提取该行的数值。
- 如果找不到完全匹配的行，请如实告知"未找到完全匹配的条目"，不要用其他表格的数据替代。

【强制规则 5：实体联合约束】
如果用户问题中同时指定了（1）一个具体的表格编号和（2）一个具体的行/列实体名称，
回答时必须将这两者视为"联合约束条件"：
- 若同一实体名称出现在多个表格中，必须优先采用用户指定编号的那个表格中的数据；
- 若在指定表格中找不到该实体名称，则回复："在 Table [编号] 中未找到 [实体名称] 的条目。"；
- 严禁跨表取数——禁止使用 Table 1 的数据来回答关于 Table 2 的问题。

【其他要求】
- 参考片段中的表格以 Markdown 表格形式给出，回答数值类问题时请直接依据表格行列作答。
- 参考片段中的 [Image: xxx.png] 是论文插图的占位标记，若问题涉及图片内容，请说明"该图内容未纳入文本检索范围"。
- 若参考片段无法回答该问题，请如实说明"资料未提供相关信息"，严禁编造。"""


def _filter_table_rows_by_entity(content, entity):
    """
    修改 6：行级精确过滤（配合 _rerank_table_first 的 if matched 分支）。

    对命中的 Table N 表格块逐行检查（不区分大小写），只保留包含实体名的
    数据行 + 表头行；GFM 表格同时保留 | --- | 分隔行，维持 Markdown 语法。

    返回:
        过滤后的表格文本；
        若没有任何数据行包含该实体，返回 None，由调用方回退为原块内容
        （宁可多给上下文，也不误删信息）。
    """
    lines = content.split("\n")
    sep_idx = None
    for i, ln in enumerate(lines):
        if _TABLE_SEP_RE.fullmatch(ln.strip()):
            sep_idx = i
            break
    if sep_idx is not None:
        # 标准 GFM 表格：表头行 + 分隔行归入表头，其余为数据行
        header, data = lines[:sep_idx + 1], lines[sep_idx + 1:]
    else:
        # 备用暴力提取的表格段落（无 | --- | 结构）：首行 "Table N ..." 作表头
        header, data = lines[:1], lines[1:]
    entity_low = entity.lower()
    matched_rows = [ln for ln in data if entity_low in ln.lower()]
    if not matched_rows:
        return None
    return "\n".join(header + matched_rows)


def _rerank_table_first(question, texts, metas):
    r"""
    修改 4 核心：轻量级重排序（不引入额外模型）。

    规则：
    1) 提问包含 ["Table", "表", "数值", "多少"] 等关键词时，
       把 type="table" 的块强行置顶到上下文最前面（即使向量相似度得分不是最高）；
    2) 若用户明确问了 "Table N"（如 Table 2），用正则
       re.search(rf'Table\s*{N}', 文本, re.IGNORECASE) 同时检查
       metadata 的 table_caption 字段与块内容（Markdown 表格源码），
       兼容 "**Table 2**" 加粗标记、多空格等格式差异（headers 字段保留为
       保守第三处检查），任一命中即视为命中；命中后只保留命中的表格块，
       其他 type="table" 块（如 Table 1 的块）从最终 order 中彻底排除，
       避免大模型同时看到多张表格而取错数（非表格文本块不受影响，保持原序）；
    2b) 命中 Table N 后做行级精确过滤（修改 6）：用 _ENTITY_RE 从问题中
        提取实体名（如 "DrugR*"），对命中的表格块逐行查找（不区分大小写），
        只保留包含该实体的数据行 + 表头行，原地替换 texts 中该块的内容；
        实体提取不到、或没有任何行命中时，保持原块内容不变（回退）；
    3) 若指定了 "Table N" 但没有任何表格块命中，不丢弃表格块——
       保留全部 type="table" 的块置顶，并返回提示语 note，
       由调用方（query_knowledge）加在上下文开头；
    4) 未指定 Table 编号（table_num 为 None）：所有表格块置顶（原有行为不变）；
    5) 非数值/表格类提问：保持向量检索原序，不做干预。

    参数:
        question: 用户提问原文
        texts   : 检索到的文本块列表（与 metas 一一对应）
        metas   : 对应的 metadata 字典列表

    返回:
        (order, note)
        order: 重排序后的下标列表（表格块在前，非表格块随后）
        note : 上下文开头提示语；无特殊情况时为空字符串

    副作用:
        规则 2b 命中时，texts 中对应表格块的内容被原地替换为行级过滤后的
        精简表格。query_knowledge 在本函数返回后才按 order 重建上下文，
        因此能自然读到过滤后的内容，无需改动调用方。
    """
    q = question.lower()
    # 非数值/表格类提问：不干预
    if not any(kw in q for kw in _TABLE_QUESTION_KEYWORDS):
        return list(range(len(texts))), ""

    table_idx = [i for i, m in enumerate(metas) if m.get("type") == "table"]

    note = ""
    # 明确问了 "Table N"？提取表号（table_num 为 None 时走规则 4：全部表格置顶）
    m = re.search(r"table\s*(\d+)", q)
    if m:
        table_num = m.group(1)
        # 修改 5：匹配 "Table N" 改用正则 re.search(..., re.IGNORECASE)，
        # 不再做子串比对。\s* 兼容 pymupdf4llm 加粗标题 "**Table 2**"、
        # 全角/多空格等格式差异——论文转 Markdown 后标题常带 "**" 包裹，
        # 子串比对容易失配。检查顺序：
        #   ① metadata.table_caption：表格标题只存在这里（块内容仅含 | 行，
        #      几乎不含 "Table N" 字样），是最主要的命中来源；
        #   ② 块内容（Markdown 表格源码）；
        #   ③ metadata.headers（章节路径，保守第三处）。
        # 任一命中即视为命中。
        table_pat = re.compile(rf"Table\s*{table_num}", re.IGNORECASE)
        matched = []
        for i in table_idx:
            caption = str(metas[i].get("table_caption", ""))
            headers = str(metas[i].get("headers", ""))
            if (table_pat.search(caption)
                    or table_pat.search(texts[i])
                    or table_pat.search(headers)):
                matched.append(i)
        if matched:
            # 只保留命中的表格块，忽略其他表格块（如 Table 1 的块）——
            # 其他表格块不得进入最终 order，否则大模型会同时看到多张表格而混淆
            table_idx = matched

            # 修改 6：行级精确过滤——从问题中提取实体名（如 DrugR*），
            # 对命中的表格块逐行查找（不区分大小写），只保留包含该实体的
            # 数据行 + 表头行，原地替换块内容：上下文更短、目标数值更突出，
            # 避免整张表格（含多个变体行）干扰 LLM 取数。
            entities = _ENTITY_RE.findall(question)
            # 排除表号本身（"Table 2"）与纯数字，剩余取最长者作为实体名
            candidates = [e for e in entities
                          if not re.fullmatch(r"table\s*\d+", e, re.IGNORECASE)
                          and not e.isdigit()]
            entity = max(candidates, key=len) if candidates else ""
            if entity:
                for i in table_idx:
                    filtered = _filter_table_rows_by_entity(texts[i], entity)
                    if filtered is not None:
                        texts[i] = filtered  # 原地替换为过滤后的表格内容
        else:
            # 没有任何块命中 Table N：不丢弃表格块——保留全部 type="table" 的块，
            # 并生成提示语，由 query_knowledge 加在上下文开头
            note = (f"未找到 Table {table_num} 的精确匹配，"
                    f"以下是知识库中所有表格数据供参考。")

    # 最终 order：表格块置顶，其余非表格块保持原相对顺序。
    # 关键保证：matched 非空时 table_idx 已被替换为 matched，而 other_idx
    # 只含非表格块——未被命中的 type="table" 块既不在 table_idx、也不在
    # other_idx 中，被彻底排除在上下文之外；matched 为空时 table_idx 保持
    # 为全部表格块（原有回退行为）。
    other_idx = [i for i, m in enumerate(metas) if m.get("type") != "table"]
    return table_idx + other_idx, note


def query_knowledge(message, history):
    if not message:
        return "请输入一个问题。"
    if collection.count() == 0:
        return "📚 知识库为空，请先上传文档。"

    question_embedding = embedding_model.encode(message).tolist()
    # 修改 4：召回数量 3 -> 6，给重排序留出候选空间
    results = collection.query(query_embeddings=[question_embedding], n_results=60)
    # --- 调试：打印检索到的前 15 个块的 type 和 caption ---
    print("===== 检索到的候选块（前15） =====")
    for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
        typ = meta.get('type', '无')
        cap = meta.get('table_caption', '')[:60]
        print(f"块{i}: type={typ}, caption={cap}")
    print("====================================")
    retrieved_texts = results['documents'][0] if results['documents'] else []
    if not retrieved_texts:
        return "未找到相关内容，请换个问法。"
    # 注意：ChromaDB 的 metadata 是嵌套列表，取第 0 层
    retrieved_metas = results['metadatas'][0] if results.get('metadatas') else []

    # ---------- 修改 4：表格优先重排序 ----------
    order, note = _rerank_table_first(message, retrieved_texts, retrieved_metas)
    ordered_texts = [retrieved_texts[i] for i in order]
    ordered_metas = [retrieved_metas[i] for i in order]

    # ---------- 修改 8：Table N 二次过滤（caption 级精确过滤） ----------
    # 在重排序结果之上再加一道保险：问题明确指定表格编号时，只保留
    # table_caption 含 "Table N"/"TableN" 的表格块（Table\s*N 正则同时覆盖
    # 带/不带空格两种写法，IGNORECASE 兼容大小写；(?!\d) 防止 "Table 1"
    # 误匹配 "Table 12"）+ 全部非表格文本块；
    # 过滤后若一个表格块都不剩，则回退到全部块（防止无答案）。
    table_filter_note = ""
    num_match = re.search(r"Table\s*(\d+)", message, re.IGNORECASE)
    if num_match:
        table_num = num_match.group(1)
        table_pat = re.compile(rf"Table\s*{table_num}(?!\d)", re.IGNORECASE)
        keep_idx = [
            i for i, m in enumerate(ordered_metas)
            if m.get("type") != "table"
            or table_pat.search(str(m.get("table_caption", "")))
        ]
        if any(ordered_metas[i].get("type") == "table" for i in keep_idx):
            # 过滤后仍有表格块：应用过滤，并提示只检索了 Table N 的数据
            ordered_texts = [ordered_texts[i] for i in keep_idx]
            ordered_metas = [ordered_metas[i] for i in keep_idx]
            table_filter_note = f"已根据您的要求只检索 Table {table_num} 的数据。"
        # else: 过滤后无表格块 → 保持原 ordered 列表不变（回退到全部块）

    # 拼接上下文：片段编号供 Prompt 规则 1 的"根据参考片段 [X] 所示"引用
    context_parts = []
    for idx, (t, m) in enumerate(zip(ordered_texts, ordered_metas), start=1):
        label = f"【片段 {idx}】[表格]" if m.get("type") == "table" else f"【片段 {idx}】"
        context_parts.append(f"{label}\n{t}")
    context = "\n\n---\n\n".join(context_parts)
    # 指定 Table N 但无精确命中时，在上下文开头附加检索提示
    if note:
        context = f"【检索提示】{note}\n\n" + context
    # 修改 8：二次过滤生效时，提示用户只检索了 Table N 的数据
    # （与上面的 note 互斥：note 仅当 matched 为空，此时二次过滤必然回退）
    if table_filter_note:
        context = f"【检索提示】{table_filter_note}\n\n" + context

    user_prompt = f"【参考资料】\n{context}\n\n【问题】\n{message}"

    # ===== 调试打印：发送给大模型的完整上下文 =====
    print("\n" + "="*60)
    print("【发送给大模型的完整上下文】")
    print("="*60)
    print(user_prompt)
    print("="*60 + "\n")

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
        sources = [
            f"- {m.get('source', '未知')}{'（表格）' if m.get('type') == 'table' else ''}"
            for m in ordered_metas
        ]
        return answer + "\n\n📌 **参考来源：**\n" + "\n".join(sources)
    except Exception as e:
        return f"❌ 调用出错：{str(e)}"

# ---------- 🆕 功能1：生成学习大纲/脑图 ----------
def generate_mindmap():
    if collection.count() == 0:
        return "📚 知识库为空，请先上传文档。"
    
    # 获取所有文档的片段（取前20个不同来源的块，防止token超限）
    all_chunks = collection.get()
    if not all_chunks or not all_chunks['documents']:
        return "无法读取文档内容。"
    
    # 简单去重，拼接上下文
    context = "\n\n".join(all_chunks['documents'][:15])  # 取前15块
    
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

# ---------- 🆕 功能2：智能出题 ----------
def generate_quiz():
    if collection.count() == 0:
        return "📚 知识库为空，请先上传文档。"
    
    all_chunks = collection.get()
    if not all_chunks or not all_chunks['documents']:
        return "无法读取文档内容。"
    
    # 随机抽取5-8个块作为出题素材
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
            temperature=0.4,  # 稍微提高一点随机性让题目更有趣
            max_tokens=2000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ 出题失败：{str(e)}"

# ---------- Gradio 集成界面 (全新升级) ----------
with gr.Blocks(title="AI 大学生学习工作台", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎓 AI 大学生学习工作台")
    gr.Markdown("上传你的课件或论文，用 AI 帮你学！")
    
    # 第一栏：上传文档（通用）
    with gr.Tab("📤 上传文档"):
        file_input = gr.File(label="选择文档", file_types=[".pdf", ".txt", ".docx"])
        upload_output = gr.Textbox(label="上传状态", lines=3)
        upload_button = gr.Button("添加到知识库")
        upload_button.click(upload_file, inputs=file_input, outputs=upload_output)
        # 显示当前知识库状态
        gr.Markdown(f"**当前知识库文本块数：** {collection.count()}")

    # 第二栏：智能问答（原有升级）
    with gr.Tab("💬 智能问答"):
        gr.ChatInterface(
            fn=query_knowledge,
            title="📖 基于文档的问答",
            description="输入问题，AI会从已上传的文档中检索答案。",
            chatbot=gr.Chatbot(height=450),
            textbox=gr.Textbox(placeholder="例如：这篇论文的核心创新点是什么？", scale=7)
        )

    # 第三栏：🆕 生成大纲/脑图
    with gr.Tab("🧠 生成学习大纲"):
        gr.Markdown("### 一键生成层级学习大纲（自动转为脑图结构）")
        btn_mindmap = gr.Button("🚀 生成大纲与脑图")
        output_mindmap = gr.Markdown(label="📋 大纲内容", value="点击上方按钮生成...")
        btn_mindmap.click(generate_mindmap, inputs=[], outputs=output_mindmap)

    # 第四栏：🆕 智能出题
    with gr.Tab("📝 智能出题"):
        gr.Markdown("### 基于当前知识库自动生成练习题（含解析）")
        btn_quiz = gr.Button("📝 生成5道选择题")
        output_quiz = gr.Markdown(label="📋 题目与解析", value="点击上方按钮生成...")
        btn_quiz.click(generate_quiz, inputs=[], outputs=output_quiz)

# ---------- 启动 ----------
if __name__ == "__main__":
    demo.launch()