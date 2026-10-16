@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Please complete the Python setup in README.md first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m stockbot.dashboard
if errorlevel 1 pause
