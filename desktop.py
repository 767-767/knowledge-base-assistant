"""Launch 文档学习工作台 with its packaged local model server."""

from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import secrets
import signal
import socket
import subprocess
import sys
import threading
import time
from urllib.error import URLError
from urllib.request import Request, urlopen


DISPLAY_NAME = "文档学习工作台"
DATA_DIR_NAME = "Sci-RAG"
MODEL_NAME = "qwen3:4b-instruct"


def resource_root() -> Path:
    return Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))


def packaged_model_paths() -> tuple[Path, Path]:
    root = resource_root()
    executable = "llama-server.exe" if os.name == "nt" else "llama-server"
    return (
        Path(os.getenv("SCI_RAG_LLAMA_SERVER", root / "runtime" / executable)),
        Path(
            os.getenv(
                "SCI_RAG_LOCAL_MODEL",
                root / "models" / "qwen3-4b-instruct-q4_k_m.gguf",
            )
        ),
    )


def application_data_dir() -> Path:
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / DATA_DIR_NAME
    if os.name == "nt":
        return Path(os.getenv("LOCALAPPDATA", Path.home())) / DATA_DIR_NAME
    return Path(os.getenv("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "sci-rag"


def _startup_window():
    import tkinter as tk

    window = tk.Tk()
    window.title(DISPLAY_NAME)
    window.resizable(False, False)
    tk.Label(
        window,
        text=f"{DISPLAY_NAME}正在启动",
        font=("Helvetica", 20, "bold"),
    ).pack(
        padx=48, pady=(28, 10)
    )
    tk.Label(
        window,
        text="正在加载本地模型，首次打开通常需要 30 至 60 秒。",
        font=("Helvetica", 13),
    ).pack(padx=48, pady=(0, 28))
    window.update_idletasks()
    x = (window.winfo_screenwidth() - window.winfo_width()) // 2
    y = (window.winfo_screenheight() - window.winfo_height()) // 2
    window.geometry(f"+{x}+{y}")
    window.update()
    return window


def _free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _wait_until_ready(
    process: subprocess.Popen[bytes], base_url: str, api_key: str
) -> None:
    request = Request(
        f"{base_url}/health",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("本地模型服务启动失败。")
        try:
            with urlopen(request, timeout=0.5) as response:
                if response.status == 200:
                    return
        except (URLError, TimeoutError):
            time.sleep(0.1)
    raise TimeoutError("本地模型加载超过 120 秒。")


@contextmanager
def local_model_server(server: Path, model: Path):
    if not server.is_file():
        raise FileNotFoundError(f"缺少本地推理程序：{server}")
    if not model.is_file():
        raise FileNotFoundError(f"缺少本地模型：{model}")

    port = _free_port()
    api_key = secrets.token_urlsafe(24)
    base_url = f"http://127.0.0.1:{port}"
    environment = os.environ.copy()
    environment["LLAMA_API_KEY"] = api_key
    process = subprocess.Popen(
        [
            str(server),
            "--model",
            str(model),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--ctx-size",
            "8192",
            "--parallel",
            "1",
            "--alias",
            MODEL_NAME,
            "--cors-origins",
            "localhost",
            "--offline",
            "--no-webui",
            "--no-slots",
            "--log-disable",
        ],
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    try:
        _wait_until_ready(process, base_url, api_key)
        yield f"{base_url}/v1", api_key
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def main() -> None:
    startup = _startup_window() if getattr(sys, "frozen", False) else None
    try:
        server, model = packaged_model_paths()
        data_dir = application_data_dir()
        data_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("SCI_RAG_DB_PATH", str(data_dir / "chroma_db"))

        embedding_model = resource_root() / "models" / "bge-small-zh-v1.5"
        if embedding_model.is_dir():
            os.environ.setdefault("SCI_RAG_EMBEDDING_MODEL", str(embedding_model))
            os.environ.setdefault("HF_HUB_OFFLINE", "1")

        with local_model_server(server, model) as (base_url, api_key):
            os.environ["LLM_BASE_URL"] = base_url
            os.environ["LLM_MODEL"] = MODEL_NAME
            os.environ["LLM_API_KEY"] = api_key

            import gradio as gr

            from app import APP_CSS, app_theme, build_demo, create_runtime

            shutdown_requested = threading.Event()

            def request_shutdown() -> None:
                threading.Timer(0.5, shutdown_requested.set).start()

            runtime = create_runtime()
            if startup is not None:
                startup.destroy()
                startup = None
            demo = build_demo(
                runtime,
                managed_local_model=True,
                on_exit=request_shutdown,
            )
            signal.signal(signal.SIGINT, lambda *_args: shutdown_requested.set())
            signal.signal(signal.SIGTERM, lambda *_args: shutdown_requested.set())
            demo.launch(
                inbrowser=True,
                prevent_thread_lock=True,
                server_name="127.0.0.1",
                theme=app_theme(gr),
                css=APP_CSS,
            )
            shutdown_requested.wait()
            demo.close()
    except Exception as exc:
        if startup is not None:
            startup.destroy()
        if getattr(sys, "frozen", False):
            from tkinter import messagebox

            messagebox.showerror(f"{DISPLAY_NAME}无法启动", str(exc))
        raise


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    main()
