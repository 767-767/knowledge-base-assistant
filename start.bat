@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"

if exist "venv\Scripts\python.exe" (
    set "PYTHON=venv\Scripts\python.exe"
    goto run
)
if exist ".venv\Scripts\python.exe" (
    set "PYTHON=.venv\Scripts\python.exe"
    goto run
)

where py >nul 2>&1
if not errorlevel 1 (
    py -3 -m venv .venv
) else (
    where python >nul 2>&1 || (
        echo 需要 Python 3 才能启动个人知识库助手。
        exit /b 1
    )
    python -m venv .venv
)
if errorlevel 1 exit /b 1
set "PYTHON=.venv\Scripts\python.exe"

:run
"%PYTHON%" -c "import operator, sys; raise SystemExit(not operator.ge(sys.version_info, (3, 10)))"
if errorlevel 1 (
    echo 需要 Python 3.10 或更高版本。
    exit /b 1
)

"%PYTHON%" test_setup.py >nul 2>&1
if errorlevel 1 (
    echo 正在安装或补全依赖……
    "%PYTHON%" -m pip install --timeout 60 -r requirements.txt
    if errorlevel 1 exit /b 1
)

"%PYTHON%" app.py
