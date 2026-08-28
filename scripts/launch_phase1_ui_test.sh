#!/usr/bin/env bash

# Launch an isolated Phase 1 UI validation environment.
#
# Default mode creates a temporary empty ChromaDB, indexes the supplied PDF,
# and starts the normal Gradio UI.  The repository's existing chroma_db is
# never modified.  The optional --existing mode copies the current database
# into the temporary directory without re-indexing.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="$PROJECT_ROOT/venv/bin/python"
DEFAULT_PDF="$PROJECT_ROOT/../2602.08213v1.pdf"
MODE="fresh"
PDF_PATH="$DEFAULT_PDF"

usage() {
  cat <<'EOF'
用法：
  bash scripts/launch_phase1_ui_test.sh [PDF路径]
  bash scripts/launch_phase1_ui_test.sh --existing

默认模式：
  在 /tmp 创建临时空 ChromaDB，导入指定 PDF，然后启动 Gradio。

--existing 模式：
  复制项目当前 chroma_db 到 /tmp 后启动，不重新导入 PDF。
EOF
}

case "${1:-}" in
  "") ;;
  --help|-h)
    usage
    exit 0
    ;;
  --existing)
    MODE="existing"
    ;;
  --*)
    echo "未知参数：$1" >&2
    usage >&2
    exit 2
    ;;
  *)
    PDF_PATH="$1"
    ;;
esac

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "找不到项目虚拟环境：$PYTHON_BIN" >&2
  echo "请先在项目根目录创建 venv 并安装 requirements.txt。" >&2
  exit 1
fi

if [[ ! -f "$PROJECT_ROOT/.env" ]]; then
  echo "找不到 $PROJECT_ROOT/.env；请从 .env.example 创建并填写 DEEPSEEK_API_KEY。" >&2
  exit 1
fi

if ! grep -Eq '^DEEPSEEK_API_KEY=[^[:space:]]+' "$PROJECT_ROOT/.env"; then
  echo "$PROJECT_ROOT/.env 中没有可用的 DEEPSEEK_API_KEY。" >&2
  exit 1
fi

if [[ "$MODE" == "fresh" && ! -f "$PDF_PATH" ]]; then
  echo "找不到 PDF：$PDF_PATH" >&2
  echo "也可以显式传入 PDF 路径：bash scripts/launch_phase1_ui_test.sh /path/to/paper.pdf" >&2
  exit 1
fi

if [[ "$MODE" == "existing" && ! -d "$PROJECT_ROOT/chroma_db" ]]; then
  echo "找不到现有数据库：$PROJECT_ROOT/chroma_db" >&2
  exit 1
fi

RUN_DIR="$(mktemp -d /tmp/sci-rag-phase1-ui.XXXXXX)"
cleanup() {
  rm -rf "$RUN_DIR"
}
trap cleanup EXIT INT TERM

DB_PATH="$RUN_DIR/chroma_db"
if [[ "$MODE" == "existing" ]]; then
  cp -a "$PROJECT_ROOT/chroma_db" "$DB_PATH"
  echo "已复制现有 ChromaDB 到临时目录：$DB_PATH"
else
  echo "正在使用临时空数据库：$DB_PATH"
  echo "正在导入：$PDF_PATH"
fi

cd "$PROJECT_ROOT"
export SCI_RAG_DB_PATH="$DB_PATH"
export PYTHONUNBUFFERED=1

if [[ "$MODE" == "fresh" ]]; then
  "$PYTHON_BIN" - "$PDF_PATH" <<'PY'
import sys

import app

pdf_path = sys.argv[1]
runtime = app.create_runtime()
message = app.add_document_to_db(pdf_path, runtime=runtime)
print(message)
print(f"临时数据库块数：{runtime.collection.count()}")
PY
fi

echo
echo "Phase 1 临时网页即将启动。"
echo "验证完成后，在终端按 Ctrl+C 停止；临时数据库会自动清理。"
echo "原始项目数据库和 .env 不会被修改。"
echo

"$PYTHON_BIN" - <<'PY'
import app

runtime = app.create_runtime()
print(f"当前测试数据库块数：{runtime.collection.count()}")
app.build_demo(runtime).launch()
PY
