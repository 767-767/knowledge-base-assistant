"""Offline dependency smoke check.

Model downloads and API/database initialization belong to the application
runtime and are intentionally not performed by this script.
"""

from importlib.metadata import version


for package in (
    "langchain-text-splitters",
    "chromadb",
    "sentence-transformers",
    "openai",
    "gradio",
    "python-dotenv",
    "python-docx",
    "pymupdf4llm",
    "pymupdf",
):
    print(f"{package}: {version(package)}")
print("离线依赖检查完成；未加载模型、未调用 API、未创建数据库。")
