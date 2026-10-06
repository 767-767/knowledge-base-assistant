# 第三方组件说明

本文记录源码版及当前 macOS 内部应用包直接使用的主要第三方组件。项目源码采用 GNU AGPL 3.0；本清单用于保留第三方归属和说明，不代表当前内部应用包已经完成正式二进制发行所需的全部准备。

## 内置模型与推理程序

| 组件 | 当前内容 | 许可证 | 来源 |
|---|---|---|---|
| Qwen3 4B Instruct 2507 | `qwen3-4b-instruct-q4_k_m.gguf` | Apache License 2.0 | <https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507> |
| BAAI BGE small zh v1.5 | 中文嵌入模型 | MIT | <https://huggingface.co/BAAI/bge-small-zh-v1.5> |
| llama.cpp / llama-server | 本地 GGUF 推理服务 | MIT | <https://github.com/ggml-org/llama.cpp> |
| Ollama | 当前内部构建所用 `llama-server` 二进制的本机来源 | MIT | <https://github.com/ollama/ollama> |

当前 GGUF 自带元数据将模型标为 `Qwen3 4B Instruct 2507`、`apache-2.0`，并链接到上述 Qwen 许可证。当前 `llama-server --version` 报告 `0.3.0-dev (build 1, commit 0f3a71be1)`；正式分发前应改为来源、版本和构建方式均可复现的二进制。

## 直接 Python 依赖

| 组件 | 当前版本 | 声明的许可证 |
|---|---:|---|
| langchain-text-splitters | 1.1.2 | MIT |
| ChromaDB | 1.5.9 | Apache License 2.0（上游仓库） |
| sentence-transformers | 6.0.0 | Apache License 2.0 |
| OpenAI Python | 1.109.1 | Apache License 2.0 |
| Gradio | 6.25.0 | Apache License 2.0 |
| python-dotenv | 1.2.3 | BSD 3-Clause |
| python-docx | 1.2.0 | MIT |
| PyMuPDF4LLM | 1.28.2 | GNU AGPL 3.0 或 Artifex 商业许可证 |
| PyMuPDF | 1.28.2 | GNU AGPL 3.0 或 Artifex 商业许可证 |

应用包还包含上述库的传递依赖。许多 Python 包的许可证文本随各自的 `.dist-info` 目录进入应用资源，但正式分发前仍需完成一次完整的传递依赖和 NOTICE 文件核对。

## 当前分发边界

PyMuPDF 官方文档说明，PyMuPDF、MuPDF 与 PyMuPDF4LLM 采用 AGPL/商业双许可证；官方 FAQ 还明确指出，在商业产品的 RAG 数据管道中解析 PDF 也适用这一许可选择。本项目选择免费的 AGPL-3.0 源码分发路线，GitHub 源码必须与项目 [`LICENSE`](LICENSE) 和本说明一同提供。

当前 GitHub 源码不包含 `.app`、模型、虚拟环境或本地资料库。若以后对外提供包含 PyMuPDF 的二进制应用，还需要同步提供与该二进制对应的完整源代码和构建材料，并完成最终传递依赖声明。现有 `.app` 尚未满足后续发行准备项，因此继续只作本机验收产物。
