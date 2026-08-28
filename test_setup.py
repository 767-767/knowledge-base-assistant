"""Offline dependency smoke check.

Model downloads and API/database initialization belong to the application
runtime and are intentionally not performed by this script.
"""

from importlib.metadata import version


for package in ("chromadb", "sentence-transformers", "pymupdf4llm", "python-docx", "ragas"):
    print(f"{package}: {version(package)}")
print("离线依赖检查完成；未加载模型、未调用 API、未创建数据库。")
