import chromadb
from sentence_transformers import SentenceTransformer

print("ChromaDB version:", chromadb.__version__)
model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
print("✓ 嵌入模型加载成功")
print("环境准备完成！")
