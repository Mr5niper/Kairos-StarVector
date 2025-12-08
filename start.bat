:: start.bat
:: One-click launcher (safe & minimal). Put in project root (same folder as requirements.txt and gui\streamlit_app.py)

@echo off
setlocal
cls

REM --- Move to project root (this file's folder) ---
cd /d "%~dp0"

REM --- Find Python ---
where py >nul 2>&1
if %ERRORLEVEL%==0 (
  set "PY_CMD=py -3"
) else (
  where python >nul 2>&1
  if %ERRORLEVEL%==0 (
    set "PY_CMD=python"
  ) else (
    echo [start] ERROR: Python 3 not found on PATH. Install Python first.
    pause
    exit /b 1
  )
)

REM --- Create venv if missing ---
if not exist "venv\Scripts\python.exe" (
  echo [start] Creating virtualenv...
  %PY_CMD% -m venv venv
  if errorlevel 1 (
    echo [start] ERROR: Could not create virtualenv.
    pause
    exit /b 1
  )
)

REM --- Use venv Python ---
set "VENV_PY=%CD%\venv\Scripts\python.exe"

REM --- Install/upgrade dependencies ---
echo [start] Installing dependencies...
"%VENV_PY%" -m pip install --upgrade pip
if errorlevel 1 goto pipfail

"%VENV_PY%" -m pip install -r requirements.txt
if errorlevel 1 goto pipfail

REM --- Make local package importable ---
"%VENV_PY%" -m pip install -e .
REM If editable install fails, we will fall back to PYTHONPATH
if errorlevel 1 (
  echo [start] WARNING: Editable install failed, using PYTHONPATH fallback.
  set "PYTHONPATH=%CD%"
)

REM --- Ensure features folder exists and has a headlines CSV ---
if not exist "features" mkdir features
if not exist "features\news_headlines.csv" (
  echo [start] No news_headlines.csv found; creating a tiny placeholder...
  >features\news_headlines.csv echo Date,Title
  >>features\news_headlines.csv echo 2020-01-02,Markets mixed amid uncertainty
)

REM --- Try to build the conditional features cache (won't stop GUI if it fails) ---
echo [start] Building conditional features cache (astro+event)...
"%VENV_PY%" scripts\build_features.py

REM --- Launch Streamlit via python -m (no reliance on PATH) ---
echo [start] Launching GUI...
"%VENV_PY%" -m streamlit run gui\streamlit_app.py
if errorlevel 1 goto streamlitfail

goto end

:pipfail
echo [start] ERROR: Failed to install requirements. Check your internet and try again.
pause
exit /b 1

:streamlitfail
echo [start] ERROR: Streamlit failed to start.
echo Try running manually to see full error:
echo "%VENV_PY%" -m streamlit run gui\streamlit_app.py
pause
exit /b 1

:end
endlocal