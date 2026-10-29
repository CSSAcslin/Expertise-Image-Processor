@echo off
setlocal
cd /d "%~dp0"
if defined EIP_PYTHON (
  if not exist "%EIP_PYTHON%" (
    echo Configured Python does not exist: "%EIP_PYTHON%"
    exit /b 1
  )
) else (
  if exist ".venv\Scripts\python.exe" (
    set "EIP_PYTHON=%~dp0.venv\Scripts\python.exe"
  ) else (
    set "EIP_PYTHON=python"
  )
)
echo Python: "%EIP_PYTHON%"
"%EIP_PYTHON%" run_web.py %*
set "EIP_EXIT_CODE=%ERRORLEVEL%"
if not "%EIP_EXIT_CODE%"=="0" pause
exit /b %EIP_EXIT_CODE%
