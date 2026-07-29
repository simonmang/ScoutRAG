@echo off
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_dashboard.ps1"
if errorlevel 1 (
  echo.
  echo ScoutRAG konnte nicht gestartet werden.
  pause
)
