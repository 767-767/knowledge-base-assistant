#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -x "$PROJECT_DIR/venv/bin/python" ]]; then
  PYTHON="$PROJECT_DIR/venv/bin/python"
elif [[ -x "$PROJECT_DIR/.venv/bin/python" ]]; then
  PYTHON="$PROJECT_DIR/.venv/bin/python"
else
  command -v python3 >/dev/null || {
    echo "需要 Python 3 才能启动 Sci-RAG。" >&2
    exit 1
  }
  echo "首次运行：正在创建本地 Python 环境……"
  python3 -m venv "$PROJECT_DIR/.venv"
  PYTHON="$PROJECT_DIR/.venv/bin/python"
fi

if ! "$PYTHON" "$PROJECT_DIR/test_setup.py" >/dev/null 2>&1; then
  echo "正在安装或补全依赖……"
  "$PYTHON" -m pip install --timeout 60 --resume-retries 10 \
    -r "$PROJECT_DIR/requirements.txt"
fi

cd "$PROJECT_DIR"
exec "$PYTHON" app.py
