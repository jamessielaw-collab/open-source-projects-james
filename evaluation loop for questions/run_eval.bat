@echo off
cd /d "%~dp0"
if not defined EVAL_LLM_PROVIDER set EVAL_LLM_PROVIDER=ollama
if not defined EVAL_AGENT_A_MODEL set EVAL_AGENT_A_MODEL=qwen2.5:3b
if not defined EVAL_AGENT_B_MODEL set EVAL_AGENT_B_MODEL=qwen2.5:3b
if /I "%EVAL_LLM_PROVIDER%"=="minimax" if not defined MINIMAX_API_KEY if not exist ".env" (
    echo MINIMAX_API_KEY is not set.
    echo.
    echo Option 1: create a .env file next to run_eval.bat containing:
    echo MINIMAX_API_KEY=your-minimax-key-here
    echo.
    echo Option 2: run from PowerShell:
    echo $env:MINIMAX_API_KEY="your-minimax-key-here"
    echo python eval_loop.py
    echo.
    pause
    exit /b 1
)
python eval_loop.py
pause
