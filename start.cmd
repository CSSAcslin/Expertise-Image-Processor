@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" run_web.py
) else (
  python run_web.py
)
if errorlevel 1 pause
