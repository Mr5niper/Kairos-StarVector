@echo off
setlocal enabledelayedexpansion

:: ==========================================================================
::  Kairos StarVector  --  EXE Builder
::  Strictly requires Python 3.13.12
::
::  Usage:
::     BUILD_EXE.bat           Standard build. Chart, sky grid, statistics
::                             and cycle analysis. Roughly 350-450 MB.
::
::     BUILD_EXE.bat full      Also bundles PyTorch, transformers and
::                             LightGBM for the Forecast models tab.
::                             Roughly 1.5-2.5 GB and about a minute to
::                             unpack on every launch, because one-file mode
::                             re-extracts the whole archive each time it
::                             starts.
::
::  No .spec file is used or kept. PyInstaller regenerates a spec on every
::  run from the command line below, so a committed one is a build artifact
::  pretending to be source: edit the bat and the spec silently wins, edit
::  the spec and the next build overwrites it. Every option lives here, and
::  the generated spec is deleted afterwards and git-ignored.
:: ==========================================================================

set "APP_NAME=Kairos-StarVector"
set "REQUIRED_PYTHON_VERSION=3.13.12"
set "PYTHON_DOWNLOAD_URL=https://www.python.org/downloads/release/python-31312/"
set "PY=py -3.13"
set "ENTRY=kairos_app.py"

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
    start "" "%PYTHON_DOWNLOAD_URL%"
    goto :error
)

:: ==========================================================================
:: Sanity check: required files present
:: ==========================================================================
echo [INFO] Verifying project files...
set "MISSING=0"
for %%F in (%ENTRY% requirements.txt gui\app.py kairos\astro.py configs\default.yaml) do (
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

:: --- Stale files from an older version ------------------------------------
:: Extracting an archive over an existing checkout overwrites matching files
:: and adds new ones, but never deletes files the archive does not contain.
:: That matters rather than just being untidy: the whole gui and
:: stock_forecast folders are bundled as data, so leftovers are compiled into
:: the exe, and stock_forecast\dataset.py still imports skyfield, which
:: stopped being a dependency in 6.0.0.
set "STALE=0"
for %%F in (
    gui\gann_app.py
    gui\streamlit_app.py
    stock_forecast\dataset.py
    stock_forecast\gann_grid.py
    scripts\run_optuna.py
    scripts\fetch_news_yf.py
) do (
    if exist "%%F" (
        echo [WARN] Stale file from a previous version: %%F
        set "STALE=1"
    )
)
if "!STALE!"=="1" (
    echo.
    echo [WARN] Those files were removed in 6.0.0 and are superseded.
    echo        Delete them and run this again for a clean build.
    echo.
    choice /C YN /M "Continue anyway"
    if errorlevel 2 goto :error
)

:: ==========================================================================
:: STEP 1 - Virtual environment
:: ==========================================================================
echo.
echo [STEP 1/6] Creating virtual environment in '.\venv'...

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
echo [STEP 2/6] Activating virtual environment...
call ".\venv\Scripts\activate.bat"

if not defined VIRTUAL_ENV (
    echo [ERROR] Failed to activate the virtual environment.
    goto :error
)
echo [INFO] Active: %VIRTUAL_ENV%

:: ==========================================================================
:: STEP 3 - Dependencies
:: ==========================================================================
echo.
echo [STEP 3/6] Installing pinned dependencies...
python -m pip install --upgrade pip >nul 2>&1

:: --only-binary :all: refuses to build anything from source. Every version
:: in requirements.txt publishes a CPython 3.13 Windows wheel, so if pip ever
:: wants to compile something the pin has drifted, and that should be an
:: error rather than a twenty-minute detour into a missing compiler.
echo [INFO] Installing from requirements.txt (exact versions, wheels only)...
python -m pip install --only-binary :all: -r requirements.txt
if errorlevel 1 (
    echo.
    echo [WARN] Wheel-only install failed. Retrying without that restriction.
    python -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Failed to install dependencies.
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
:: STEP 4 - Environment self-test
:: ==========================================================================
:: Catching a broken environment here is far cheaper than discovering it
:: after a twenty-minute PyInstaller run.
echo.
echo [STEP 4/6] Verifying the environment before building...
python -c "import numpy,pandas,scipy,plotly,streamlit,yfinance,ephem,yaml; import kairos; from kairos import astro,waves,gann,market,charting,paths,calibrate,skygrid; print('[INFO] Imports OK - kairos', kairos.__version__)"
if errorlevel 1 (
    echo [ERROR] The environment failed its import check. Not building.
    goto :error
)

python -c "from kairos import astro; import pandas as pd; d=pd.DatetimeIndex(['2026-08-18']); l=astro.longitudes(d); print('[INFO] Ephemeris OK - Sun at', astro.dms(l['SUN'].iloc[0]))"
if errorlevel 1 (
    echo [ERROR] The ephemeris self-test failed. Not building.
    goto :error
)

:: Validate the icon before PyInstaller touches it. A PNG renamed to .ico is
:: the usual mistake, and PyInstaller's own failure for it is a struct
:: unpacking error deep in its resource writer that never mentions the icon.
set "ICON_ARG="
if exist "icon.ico" (
    python -c "import sys; sys.exit(0 if open('icon.ico','rb').read(4)==b'\x00\x00\x01\x00' else 1)"
    if errorlevel 1 (
        echo [ERROR] icon.ico is not a valid icon file - probably a renamed PNG.
        echo         Convert it properly, or delete it to build without an icon.
        goto :error
    )
    set "ICON_ARG=--icon icon.ico"
    echo [INFO] Icon OK: icon.ico
) else (
    echo [INFO] No icon.ico found; building without an icon.
)

set "VERSION_ARG="
if exist "version.txt" set "VERSION_ARG=--version-file version.txt"

:: ==========================================================================
:: STEP 5 - Assemble the PyInstaller command
:: ==========================================================================
:: Built up in pieces rather than one enormous line, because a single command
:: spanning forty caret-continued lines breaks the moment one trailing space
:: sneaks in after a caret.
echo.
echo [STEP 5/6] Assembling build options...

set "ARGS=--clean --noconfirm --onefile --noupx --console --name %APP_NAME%"
set "ARGS=!ARGS! !ICON_ARG! !VERSION_ARG!"

:: Bundled resources. The GUI script is data rather than an analysed import,
:: so kairos is added both as data and as collected submodules: the frozen
:: importer needs the modules, and paths.py resolves the source either way.
set "ARGS=!ARGS! --add-data "gui;gui""
set "ARGS=!ARGS! --add-data "kairos;kairos""
set "ARGS=!ARGS! --add-data "configs;configs""
set "ARGS=!ARGS! --add-data "stock_forecast;stock_forecast""
if exist "README.md" set "ARGS=!ARGS! --add-data "README.md;.""

:: Streamlit's compiled frontend, package metadata and dynamically imported
:: server stack are invisible to the import scanner and must be collected.
set "ARGS=!ARGS! --collect-all streamlit --collect-all altair --collect-all pydeck"
set "ARGS=!ARGS! --collect-all narwhals --collect-all uvicorn --collect-all starlette"
set "ARGS=!ARGS! --collect-all plotly --collect-all pyarrow"
set "ARGS=!ARGS! --collect-all yfinance --collect-all curl_cffi --collect-all ephem"
set "ARGS=!ARGS! --collect-submodules kairos"

:: Uvicorn resolves its protocol classes from strings at runtime.
set "ARGS=!ARGS! --hidden-import uvicorn.protocols.http.httptools_impl"
set "ARGS=!ARGS! --hidden-import uvicorn.protocols.http.h11_impl"
set "ARGS=!ARGS! --hidden-import uvicorn.protocols.websockets.websockets_impl"
set "ARGS=!ARGS! --hidden-import uvicorn.loops.asyncio"
set "ARGS=!ARGS! --hidden-import uvicorn.lifespan.on --hidden-import uvicorn.lifespan.off"
set "ARGS=!ARGS! --hidden-import httptools --hidden-import websockets"
set "ARGS=!ARGS! --hidden-import websockets.legacy --hidden-import multipart"
set "ARGS=!ARGS! --hidden-import itsdangerous --hidden-import encodings.idna"
set "ARGS=!ARGS! --hidden-import pandas._libs.tslibs.timedeltas"

if /i "!BUILD_MODE!"=="full" (
    echo [INFO] FULL build: including the machine-learning stack.
    set "ARGS=!ARGS! --collect-all torch --collect-all lightgbm"
    set "ARGS=!ARGS! --collect-all statsmodels --collect-all sklearn"
    set "ARGS=!ARGS! --collect-submodules stock_forecast"
    set "ARGS=!ARGS! --exclude-module tkinter --exclude-module matplotlib"
    set "ARGS=!ARGS! --exclude-module PyQt5 --exclude-module PyQt6"
    set "ARGS=!ARGS! --exclude-module PySide2 --exclude-module PySide6"
    set "ARGS=!ARGS! --exclude-module IPython --exclude-module jupyter"
    set "ARGS=!ARGS! --exclude-module notebook --exclude-module pytest"
) else (
    echo [INFO] LIGHT build: machine-learning stack excluded.
    :: Excluded rather than merely absent, so a stray import cannot drag two
    :: gigabytes in if these happen to be installed in the venv.
    set "ARGS=!ARGS! --exclude-module torch --exclude-module torchvision"
    set "ARGS=!ARGS! --exclude-module torchaudio --exclude-module transformers"
    set "ARGS=!ARGS! --exclude-module sentence_transformers"
    set "ARGS=!ARGS! --exclude-module tokenizers --exclude-module safetensors"
    set "ARGS=!ARGS! --exclude-module lightgbm --exclude-module optuna"
    set "ARGS=!ARGS! --exclude-module sklearn --exclude-module scikit_learn"
    set "ARGS=!ARGS! --exclude-module statsmodels --exclude-module tensorflow"
    set "ARGS=!ARGS! --exclude-module keras --exclude-module onnxruntime"
    set "ARGS=!ARGS! --exclude-module tkinter --exclude-module matplotlib"
    set "ARGS=!ARGS! --exclude-module PyQt5 --exclude-module PyQt6"
    set "ARGS=!ARGS! --exclude-module PySide2 --exclude-module PySide6"
    set "ARGS=!ARGS! --exclude-module IPython --exclude-module jupyter"
    set "ARGS=!ARGS! --exclude-module notebook --exclude-module pytest"
    set "ARGS=!ARGS! --exclude-module sqlalchemy --exclude-module alembic"
)

:: ==========================================================================
:: STEP 6 - Build
:: ==========================================================================
echo.
echo [STEP 6/6] Building the one-file executable...
echo [INFO] This takes several minutes. PyInstaller is quiet in places; that
echo        is normal, not a hang.
echo.
echo [INFO] Two warnings about 'pydeck.widget' and 'pyarrow.tests.parquet'
echo        are expected and harmless. Those subpackages bridge to ipywidgets
echo        and pytest, neither of which ships here.
echo.

pyinstaller !ARGS! %ENTRY%

if errorlevel 1 (
    echo.
    echo [ERROR] PyInstaller build failed. Scroll up for the actual error.
    echo.
    echo         Common causes:
    echo           - Antivirus locked a file in build\ or dist\. Add this
    echo             folder to your exclusions and try again.
    echo           - Not enough free disk space. A light build needs about
    echo             2 GB while working; a full build needs about 8 GB.
    echo           - A stale venv. Delete the venv folder and rerun.
    goto :error
)

if not exist "dist\%APP_NAME%.exe" (
    echo [ERROR] The build reported success but dist\%APP_NAME%.exe is missing.
    goto :error
)

:: The spec PyInstaller just generated is an artifact, not source. Remove it
:: so it cannot be committed or drift out of step with this script.
if exist "%APP_NAME%.spec" (
    del /q "%APP_NAME%.spec"
    echo [INFO] Removed the generated %APP_NAME%.spec
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
echo   Double-click the exe to start. It opens a console window showing the
echo   local address, then your browser. Keep the console open; closing it
echo   stops the server.
echo.
echo   One-file executables unpack to a temp folder on every launch, so the
echo   first few seconds of startup are that extraction, not a hang.
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
