@echo off
chcp 65001 >nul
title RAG 知識庫問答系統 (Google Gemini + FAISS)

echo ============================================================
echo   RAG (Retrieval-Augmented Generation) 知識庫問答系統
echo ============================================================
echo.

REM 檢查是否存在 .env 設定檔
if not exist ".env" (
    echo [提醒] 尚未偵測到 .env 檔案！
    echo [提醒] 請確認已在 .env 中設定 GOOGLE_API_KEY。
    echo.
)

cd /d "%~dp0"

REM 檢查是否有安裝 uv
where uv >nul 2>nul
if %ERRORLEVEL% equ 0 goto USE_UV

REM 若無 uv，檢查是否有安裝 python
where python >nul 2>nul
if %ERRORLEVEL% neq 0 goto NO_PYTHON

REM 使用標準 Python
if not exist ".venv" (
    echo [資訊] 正在建立虛擬環境 .venv
    python -m venv .venv
    call .venv\Scripts\activate.bat
    echo [資訊] 正在安裝相依套件
    python -m pip install --upgrade pip
    pip install langchain langchain-community langchain-google-genai langchain-text-splitters faiss-cpu pypdf python-dotenv
) else (
    call .venv\Scripts\activate.bat
)

echo.
echo [執行] 正在啟動 RAG 系統 (main.py)
echo.
python main.py
goto END

:USE_UV
echo [資訊] 偵測到 uv 套件管理工具，正在檢查並同步套件
call uv sync
echo.
echo [執行] 正在啟動 RAG 系統 (main.py)
echo.
call uv run python main.py
goto END

:NO_PYTHON
echo [錯誤] 系統未安裝 Python 或 uv，請先安裝 Python (>=3.10) 或 uv。
goto END

:END
echo.
echo ============================================================
echo 程式執行結束。
echo ============================================================
pause
