@echo off
rem Double-click to start the AI Handwritten Math Solver and open it in your browser.
title AI Handwritten Math Solver
cd /d "%~dp0"
set "PY=%USERPROFILE%\.venvs\handwritten-math-solver\Scripts\python.exe"
set "URL=http://localhost:8601"
set "HEALTH=$(try { (Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 %URL%/_stcore/health).StatusCode -eq 200 } catch { $false })"
set "TF_CPP_MIN_LOG_LEVEL=2"

if not exist "%PY%" (
    echo Python environment not found:
    echo   %PY%
    echo Create it with the steps under "Run it" in README.md.
    pause
    exit /b 1
)

rem Already running from an earlier double-click? If no file has changed since it started, just open the
rem page again. If the code or model is newer, close that old copy so the new version loads.
set "STATE=none"
for /f "usebackq delims=" %%s in (`powershell -NoProfile -Command "$c = Get-NetTCPConnection -LocalPort 8601 -State Listen -ErrorAction SilentlyContinue; if (-not $c) { 'none'; exit }; $p = Get-CimInstance Win32_Process -Filter ('ProcessId = ' + $c[0].OwningProcess); if ($p.CommandLine -notlike '*streamlit*') { 'busy'; exit }; $newest = (Get-ChildItem *.py, models\* | Sort-Object LastWriteTime -Descending | Select-Object -First 1).LastWriteTime; if ($newest -gt $p.CreationDate) { Stop-Process -Id $p.ProcessId -Force; 'stopped' } else { 'current' }"`) do set "STATE=%%s"
if "%STATE%"=="current" (
    start "" "%URL%"
    exit /b 0
)
if "%STATE%"=="busy" (
    echo Another program is already using port 8601. Close it and try again.
    pause
    exit /b 1
)
if "%STATE%"=="stopped" echo An older copy of the app was still running. It has been closed so the new version starts.

rem Open the browser as soon as the app answers. This window keeps the app running.
start "" /b powershell -NoProfile -Command "for ($i = 0; $i -lt 240; $i++) { if (%HEALTH%) { Start-Process '%URL%'; break }; Start-Sleep -Milliseconds 500 }"

echo Starting the AI Handwritten Math Solver. Your browser opens by itself when it is ready.
echo Keep this window open while you use the app. Close it to stop the app.
echo.
"%PY%" -m streamlit run app.py --server.port 8601 --server.headless true --browser.gatherUsageStats false
if errorlevel 1 pause
