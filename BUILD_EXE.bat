@echo off
setlocal enabledelayedexpansion

:: ==========================================================================
::  Kairos StarVector  --  EXE Builder
::  Strictly requires Python 3.13.12
::
::  Usage:
::     BUILD_EXE.bat           Standard build. Chart, alignment wave,
::                             statistics and cycle analysis. Roughly
::                             350-450 MB, opens in a few seconds.
::
::     BUILD_EXE.bat full      Also bundles PyTorch, transformers and
::                             LightGBM for the Forecast models tab.
::                             Roughly 1.5-2.5 GB and takes about a minute
::                             to unpack on every launch, because one-file
::                             mode re-extracts the whole archive each time
::                             it starts. Only worth it if you actually run
::                             the neural benchmark. Otherwise leave it out
::                             and run that from source instead.
::
::  Everything installs from requirements.txt with exact == versions, so
::  the same build inputs produce the same build every time.
:: ==========================================================================

set "APP_NAME=Kairos-StarVector"
set "REQUIRED_PYTHON_VERSION=3.13.12"
set "PYTHON_DOWNLOAD_URL=https://www.python.org/downloads/release/python-31312/"
set "PY=py -3.13"
set "SPEC_FILE=Kairos-StarVector.spec"

:: --- Parse the optional build mode ---------------------------------------
set "BUILD_MODE=light"
if /i "%~1"=="full" set "BUILD_MODE=full"
if /i "%~1"=="-full" set "BUILD_MODE=full"
if /i "%~1"=="--full" set "BUILD_MODE=full"

cd /d "%~dp0"

echo ==========================================================================
echo   Building %APP_NAME%   (mode: !BUILD_MODE!)
echo ==========================================================================
echo.

:: ==========================================================================
:: Pre-flight: verify Python via the py launcher
:: ==========================================================================
:: The py launcher lives in C:\Windows and is found even when the 'python'
:: on PATH is a different version, which is the usual reason a build script
:: silently uses the wrong interpreter.
echo [INFO] Checking Python version...

%PY% --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo [ERROR] Python 3.13 was not found via the py launcher.
    echo         This build script requires Python %REQUIRED_PYTHON_VERSION%.
    echo         Tried: %PY%
    echo.
    echo         Install it from:
    echo         %PYTHON_DOWNLOAD_URL%
    echo.
    echo [NOTE]  During installation, enable the py launcher option.
    start "" "%PYTHON_DOWNLOAD_URL%"
    goto :error
)

for /f "tokens=2 delims= " %%v in ('%PY% --version 2^>^&1') do set "CURRENT_PYTHON_VERSION=%%v"

echo [INFO] Current Python version:  !CURRENT_PYTHON_VERSION!
echo [INFO] Required Python version: %REQUIRED_PYTHON_VERSION%

if not "!CURRENT_PYTHON_VERSION!"=="%REQUIRED_PYTHON_VERSION%" (
    echo.
    echo [ERROR] Incorrect Python version detected.
    echo         The py launcher resolved !CURRENT_PYTHON_VERSION! instead of
    echo         %REQUIRED_PYTHON_VERSION%.
    echo.
    echo         The pinned wheels in requirements.txt are the CPython 3.13
    echo         Windows builds. A different minor version will fall back to
    echo         building from source and will need a C compiler.
    echo.
    echo         Install the correct version from:
    echo         %PYTHON_DOWNLOAD_URL%
    echo.
    start "" "%PYTHON_DOWNLOAD_URL%"
    goto :error
)

:: ==========================================================================
:: Sanity check: required files present
:: ==========================================================================
echo [INFO] Verifying project files...
set "MISSING=0"
for %%F in (kairos_app.py requirements.txt %SPEC_FILE% gui\app.py kairos\astro.py configs\default.yaml) do (
    if not exist "%%F" (
        echo [ERROR] Missing required file: %%F
        set "MISSING=1"
    )
)
if "!MISSING!"=="1" (
    echo.
    echo [ERROR] Project files are incomplete. Extract the full archive,
    echo         keeping the folder structure intact, and run this again.
    goto :error
)
echo [INFO] All required files present.

:: ==========================================================================
:: STEP 1 - Virtual environment
:: ==========================================================================
echo.
echo [STEP 1/5] Creating virtual environment in '.\venv'...

if not exist ".\venv\Scripts\python.exe" (
    %PY% -m venv .\venv
    if errorlevel 1 (
        echo [ERROR] Failed to create the virtual environment.
        goto :error
    )
    echo [INFO] Virtual environment created.
) else (
    echo [INFO] Virtual environment already exists. Reusing it.
)

:: ==========================================================================
:: STEP 2 - Activate
:: ==========================================================================
echo.
echo [STEP 2/5] Activating virtual environment...
call ".\venv\Scripts\activate.bat"

if not defined VIRTUAL_ENV (
    echo [ERROR] Failed to activate the virtual environment.
    echo         Expected: .\venv\Scripts\activate.bat
    goto :error
)
echo [INFO] Active: %VIRTUAL_ENV%

:: ==========================================================================
:: STEP 3 - Dependencies
:: ==========================================================================
echo.
echo [STEP 3/5] Installing pinned dependencies...
python -m pip install --upgrade pip >nul 2>&1

:: --only-binary :all: refuses to build anything from source. Every version
:: in requirements.txt publishes a CPython 3.13 Windows wheel, so if pip
:: ever wants to compile something, the pin has drifted and that should be
:: an error rather than a twenty-minute detour into a missing compiler.
echo [INFO] Installing from requirements.txt (exact versions, wheels only)...
python -m pip install --only-binary :all: -r requirements.txt
if errorlevel 1 (
    echo.
    echo [WARN] Wheel-only install failed. Retrying without that restriction.
    echo        If this succeeds, something in requirements.txt no longer has
    echo        a 3.13 Windows wheel and is being compiled locally.
    python -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Failed to install dependencies.
        echo         Check your internet connection and the errors above.
        goto :error
    )
)

if /i "!BUILD_MODE!"=="full" (
    echo.
    echo [INFO] FULL build: installing machine-learning extras.
    echo [INFO] This downloads roughly 2.5 GB and will take a while.
    python -m pip install -r requirements-ml.txt
    if errorlevel 1 (
        echo [ERROR] Failed to install the machine-learning extras.
        goto :error
    )
)

echo [INFO] Dependencies installed.

:: ==========================================================================
:: STEP 4 - Import self-test
:: ==========================================================================
:: Catching a broken environment here is far cheaper than discovering it
:: after a twenty-minute PyInstaller run.
echo.
echo [STEP 4/5] Verifying the environment before building...
python -c "import numpy,pandas,scipy,plotly,streamlit,yfinance,ephem,yaml; import kairos; from kairos import astro,waves,gann,market,charting,paths; print('[INFO] Imports OK - kairos', kairos.__version__)"
if errorlevel 1 (
    echo [ERROR] The environment failed its import check. Not building.
    goto :error
)

python -c "from kairos import astro; import pandas as pd; d=pd.DatetimeIndex(['2026-08-18']); l=astro.longitudes(d); print('[INFO] Ephemeris OK - Sun at', astro.dms(l['SUN'].iloc[0]))"
if errorlevel 1 (
    echo [ERROR] The ephemeris self-test failed. Not building.
    goto :error
)

:: ==========================================================================
:: STEP 5 - Build
:: ==========================================================================
echo.
echo [STEP 5/5] Building the one-file executable...
echo [INFO] This takes several minutes. PyInstaller is quiet in places; that
echo        is normal, not a hang.
echo.

if /i "!BUILD_MODE!"=="full" (
    set "KAIROS_BUILD_FULL=1"
) else (
    set "KAIROS_BUILD_FULL=0"
)

pyinstaller --clean --noconfirm "%SPEC_FILE%"

if errorlevel 1 (
    echo.
    echo [ERROR] PyInstaller build failed. Scroll up for the actual error.
    echo.
    echo         Common causes:
    echo           - Antivirus locked a file in build\ or dist\. Add this
    echo             folder to your exclusions and try again.
    echo           - Not enough free disk space. A light build needs about
    echo             2 GB free while working; a full build needs about 8 GB.
    echo           - A stale venv. Delete the venv folder and rerun.
    goto :error
)

if not exist "dist\%APP_NAME%.exe" (
    echo [ERROR] The build reported success but dist\%APP_NAME%.exe is missing.
    goto :error
)

for %%A in ("dist\%APP_NAME%.exe") do set "EXE_SIZE=%%~zA"
set /a EXE_MB=!EXE_SIZE! / 1048576

echo.
echo ==========================================================================
echo   [SUCCESS] Build complete.
echo ==========================================================================
echo   Executable : dist\%APP_NAME%.exe
echo   Size       : !EXE_MB! MB
echo   Mode       : !BUILD_MODE!
echo.
echo   Double-click the exe to start. It opens a console window that shows
echo   the local address, then your browser. Keep the console open; closing
echo   it stops the server.
echo.
echo   One-file executables unpack themselves to a temp folder on every
echo   launch, so the first few seconds of startup are that extraction,
echo   not the program hanging.
echo ==========================================================================
goto :end

:error
echo.
echo [FAILURE] The build process did not complete.
echo.
pause
exit /b 1

:end
echo.
pause
endlocal
exit /b 0
