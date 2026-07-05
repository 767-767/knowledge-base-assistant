import os
import ssl
import random
from dotenv import load_dotenv
load_dotenv()
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
ssl._create_default_https_context = ssl._create_unverified_context

import gradio as gr
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
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

# ---------- 文档处理 (保持不变) ----------
def load_and_split_document(file_path):
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path, encoding="utf-8")
    elif file_path.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError("不支持的文件类型")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512, chunk_overlap=64, length_function=len,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
    )
    return text_splitter.split_documents(documents)

def add_document_to_db(file_path):
    chunks = load_and_split_document(file_path)
    for chunk in chunks:
        chunk_id = str(uuid.uuid4())
        text = chunk.page_content
        metadata = {"source": os.path.basename(file_path)}
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

# ---------- 核心问答 (保持不变) ----------
def respond(message, history):
    if not message:
        return "请输入一个问题。"
    if collection.count() == 0:
        return "📚 知识库为空，请先上传文档。"
    
    question_embedding = embedding_model.encode(message).tolist()
    results = collection.query(query_embeddings=[question_embedding], n_results=3)
    retrieved_texts = results['documents'][0] if results['documents'] else []
    if not retrieved_texts:
        return "未找到相关内容，请换个问法。"
    
    context = "\n\n---\n\n".join(retrieved_texts)
    system_prompt = "你是一个专业的学习助手。请根据参考资料回答问题，并注明来源。"
    user_prompt = f"【参考资料】\n{context}\n\n【问题】\n{message}"
    
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=2048
        )
        answer = response.choices[0].message.content
        sources = [f"- {meta.get('source', '未知')}" for meta in results['metadatas'][0]]
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
            fn=respond,
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