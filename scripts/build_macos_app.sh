#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -x "$PROJECT_DIR/venv/bin/python" ]]; then
  PROJECT_PYTHON="$PROJECT_DIR/venv/bin/python"
else
  PROJECT_PYTHON="$PROJECT_DIR/.venv/bin/python"
fi

: "${SCI_RAG_LLAMA_SERVER:?请设置 SCI_RAG_LLAMA_SERVER 为 llama-server 路径}"
: "${SCI_RAG_LOCAL_MODEL:?请设置 SCI_RAG_LOCAL_MODEL 为 Qwen GGUF 路径}"
: "${SCI_RAG_EMBEDDING_MODEL:?请设置 SCI_RAG_EMBEDDING_MODEL 为 BGE 模型目录}"

"$PROJECT_PYTHON" -c "import PyInstaller" 2>/dev/null || {
  echo "请先运行：$PROJECT_PYTHON -m pip install pyinstaller" >&2
  exit 1
}

SITE_PACKAGES="$("$PROJECT_PYTHON" -c 'import site; print(site.getsitepackages()[0])')"
STAGED_MODEL="$PROJECT_DIR/build/assets/qwen3-4b-instruct-q4_k_m.gguf"
mkdir -p "$(dirname "$STAGED_MODEL")"
cp -c "$SCI_RAG_LOCAL_MODEL" "$STAGED_MODEL"

cd "$PROJECT_DIR"
PYINSTALLER_CONFIG_DIR="${PYINSTALLER_CONFIG_DIR:-/private/tmp/scirag-pyinstaller}" \
  "$PROJECT_PYTHON" -m PyInstaller \
  --noconfirm \
  --name Sci-RAG \
  --windowed \
  --target-arch arm64 \
  --osx-bundle-identifier com.scirag.desktop \
  --collect-data chromadb \
  --hidden-import chromadb.api.rust \
  --hidden-import chromadb.telemetry.product.posthog \
  --hidden-import chromadb_rust_bindings \
  --add-data "$SITE_PACKAGES/gradio:gradio" \
  --add-data "$SITE_PACKAGES/groovy:groovy" \
  --add-data "$SITE_PACKAGES/safehttpx:safehttpx" \
  --add-binary "$SCI_RAG_LLAMA_SERVER:runtime" \
  --add-data "$STAGED_MODEL:models" \
  --add-data "$SCI_RAG_EMBEDDING_MODEL:models/bge-small-zh-v1.5" \
  desktop.py

echo "已生成：$PROJECT_DIR/dist/Sci-RAG.app"
