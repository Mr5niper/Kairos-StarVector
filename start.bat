@echo off
setlocal enabledelayedexpansion

:: ==========================================================================
::  Kairos StarVector  --  run from source
::  Strictly requires Python 3.13.12
::
::  Sets up the venv if needed, installs the pinned dependencies, and starts
::  the app. Use this for development, or if you would rather not build an
::  exe at all: it starts faster than the one-file build, because nothing
::  has to be unpacked.
::
::  Usage:
::     START.bat          Start the app.
::     START.bat ml       Also install the machine-learning extras first.
::     START.bat reset    Delete the venv and rebuild it from scratch.
:: ==========================================================================

set "REQUIRED_PYTHON_VERSION=3.13.12"
set "PYTHON_DOWNLOAD_URL=https://www.python.org/downloads/release/python-31312/"
set "PY=py -3.13"

set "MODE=run"
if /i "%~1"=="ml" set "MODE=ml"
if /i "%~1"=="reset" set "MODE=reset"

cd /d "%~dp0"

echo ==========================================================================
echo   Kairos StarVector
echo ==========================================================================

:: --- Python check --------------------------------------------------------
%PY% --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python 3.13 not found via the py launcher.
    echo         Install Python %REQUIRED_PYTHON_VERSION% from:
    echo         %PYTHON_DOWNLOAD_URL%
    start "" "%PYTHON_DOWNLOAD_URL%"
    goto :error
)

for /f "tokens=2 delims= " %%v in ('%PY% --version 2^>^&1') do set "FOUND=%%v"
if not "!FOUND!"=="%REQUIRED_PYTHON_VERSION%" (
    echo [WARN] Found Python !FOUND!, expected %REQUIRED_PYTHON_VERSION%.
    echo        The pinned wheels target 3.13. Continuing, but if the install
    echo        fails, this is why.
    echo.
)

:: --- Reset if asked -----------------------------------------------------
if /i "!MODE!"=="reset" (
    if exist ".\venv" (
        echo [INFO] Removing the existing venv...
        rmdir /s /q ".\venv"
    )
    echo [INFO] Reset done. Run START.bat again.
    goto :end
)

:: --- Venv ---------------------------------------------------------------
set "FRESH=0"
if not exist ".\venv\Scripts\python.exe" (
    echo [INFO] Creating virtual environment...
    %PY% -m venv .\venv
    if errorlevel 1 (
        echo [ERROR] Could not create the virtual environment.
        goto :error
    )
    set "FRESH=1"
)

call ".\venv\Scripts\activate.bat"
if not defined VIRTUAL_ENV (
    echo [ERROR] Could not activate the virtual environment.
    echo         Try: START.bat reset
    goto :error
)

:: --- Dependencies -------------------------------------------------------
:: Only installed on a fresh venv or when the marker is missing, so normal
:: startup does not wait on pip every single time.
if "!FRESH!"=="1" goto :install
if not exist ".\venv\.kairos-deps-ok" goto :install
goto :skipinstall

:install
echo [INFO] Installing pinned dependencies. First run only.
python -m pip install --upgrade pip >nul 2>&1
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Dependency install failed. Check your connection.
    goto :error
)
echo ok > ".\venv\.kairos-deps-ok"
echo [INFO] Dependencies installed.

:skipinstall

if /i "!MODE!"=="ml" (
    echo [INFO] Installing machine-learning extras. This is a large download.
    python -m pip install -r requirements-ml.txt
    if errorlevel 1 (
        echo [ERROR] Machine-learning extras failed to install.
        goto :error
    )
)

:: --- Launch -------------------------------------------------------------
echo.
echo [INFO] Starting. Your browser should open in a few seconds.
echo [INFO] Keep this window open; closing it stops the server.
echo.
python kairos_app.py
if errorlevel 1 (
    echo.
    echo [ERROR] The app exited with an error. The traceback is above.
    goto :error
)
goto :end

:error
echo.
echo [FAILURE] Startup did not complete.
pause
exit /b 1

:end
echo.
pause
endlocal
exit /b 0
