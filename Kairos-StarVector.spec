# -*- mode: python ; coding: utf-8 -*-
"""
Kairos-StarVector.spec
======================
PyInstaller build recipe. Driven by BUILD_EXE.bat; you should not normally
need to run it directly.

    pyinstaller --clean --noconfirm Kairos-StarVector.spec

Set KAIROS_BUILD_FULL=1 in the environment to include the machine-learning
stack. BUILD_EXE.bat does this for you when passed the `full` argument.

Why a spec file rather than a long command line
-----------------------------------------------
Streamlit is awkward to freeze. It is a web server that reads its own
package metadata at import time, loads a tree of static assets from disk,
and since version 1.61 sits on Uvicorn, which resolves its protocol
implementations from strings at runtime. None of that is visible to
PyInstaller's import scanner. Getting it right needs half a dozen
collect_all calls and a list of hidden imports, which is unreadable as
command-line flags and impossible to comment.

The three failure modes this file exists to prevent:

  * "No module named streamlit.web.bootstrap" - the package was collected
    as bytecode but its metadata was not, so importlib.metadata.version()
    raised during startup.
  * A blank white page in the browser - the server started but the
    frontend static assets were never bundled.
  * An endless cascade of windows on launch - a spawned subprocess
    re-executed the bootloader. kairos_app.py calls freeze_support() first
    to stop that; this file keeps the process count down by excluding the
    libraries that fork.
"""
import os
import sys

from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules

BUILD_FULL = os.environ.get("KAIROS_BUILD_FULL", "0").strip().lower() in ("1", "true", "yes")

APP_NAME = "Kairos-StarVector"
ENTRY = "kairos_app.py"

datas = []
binaries = []
hiddenimports = []


# Subpackages that exist only to bridge to a library we deliberately do not
# ship. Without filtering, collect_submodules tries to import each one,
# fails, and logs a warning:
#
#   WARNING: Failed to collect submodules for 'plotly.matplotlylib' because
#   importing 'plotly.matplotlylib' raised: No module named 'matplotlib'
#
#   WARNING: Failed to collect submodules for 'pydeck.widget' because
#   importing 'pydeck.widget' raised: No module named 'ipywidgets'
#
# Both are harmless - the parent package still collects fine and neither
# bridge is reachable from this app. plotly.matplotlylib converts matplotlib
# figures to Plotly ones; pydeck.widget is the Jupyter notebook renderer,
# and Streamlit's st.pydeck_chart uses the deck.gl JSON API instead.
#
# They are filtered anyway, for one reason: a build log that cries wolf
# trains you to skim past warnings, and the next one might matter. Filtering
# by name means the module is never imported at all, so nothing is logged.
SKIP_SUBMODULES = {
    "plotly": ("plotly.matplotlylib",),
    "pydeck": ("pydeck.widget",),
    "transformers": ("transformers.models.deprecated",),
}


def add_all(package: str, required: bool = True) -> None:
    """
    Pull in a package's data files, extension modules, submodules and
    metadata. Missing optional packages are skipped rather than aborting
    the build, which is what lets one spec file serve both the light and
    the full build.
    """
    skip = SKIP_SUBMODULES.get(package, ())

    def keep(name: str) -> bool:
        return not any(name == s or name.startswith(s + ".") for s in skip)

    try:
        pkg_datas, pkg_binaries, pkg_hidden = collect_all(
            package, filter_submodules=keep, on_error="ignore",
        )
    except Exception as exc:
        if required:
            raise SystemExit(f"[spec] Required package '{package}' could not be "
                             f"collected: {exc}")
        print(f"[spec] optional package '{package}' not present, skipping")
        return
    datas.extend(pkg_datas)
    binaries.extend(pkg_binaries)
    hiddenimports.extend(pkg_hidden)
    note = f" (skipped {', '.join(skip)})" if skip else ""
    print(f"[spec] collected {package}: {len(pkg_datas)} data, "
          f"{len(pkg_binaries)} binaries, {len(pkg_hidden)} hidden imports{note}")


# --- Streamlit and its frontend -----------------------------------------
# collect_all is what brings in the compiled frontend under
# streamlit/static plus the .dist-info directory that
# importlib.metadata.version("streamlit") looks for at import time.
add_all("streamlit")
add_all("altair")
add_all("pydeck")
add_all("narwhals")

# Uvicorn and Starlette. Uvicorn picks its HTTP and WebSocket protocol
# classes by importing module paths held in strings, so the scanner never
# sees them.
add_all("uvicorn")
add_all("starlette")
hiddenimports += [
    "uvicorn.protocols.http.httptools_impl",
    "uvicorn.protocols.http.h11_impl",
    "uvicorn.protocols.websockets.websockets_impl",
    "uvicorn.loops.asyncio",
    "uvicorn.lifespan.on",
    "uvicorn.lifespan.off",
    "httptools",
    "websockets",
    "websockets.legacy",
    "multipart",
    "itsdangerous",
]

# --- Charting -----------------------------------------------------------
add_all("plotly")

# --- Data -------------------------------------------------------------
# pyarrow backs Streamlit's dataframe serialisation. Large, and not
# optional however much you might want it to be.
add_all("pyarrow")
add_all("yfinance")
add_all("curl_cffi")

# --- Astronomy --------------------------------------------------------
add_all("ephem")

# --- This project -----------------------------------------------------
# The GUI script is data, not an analysed import, so nothing it imports is
# discovered automatically. The kairos package is therefore added twice, on
# purpose: as collected submodules so the frozen importer can serve it, and
# as source files so the path in kairos/paths.py resolves either way.
hiddenimports += collect_submodules("kairos")
datas += [
    ("gui", "gui"),
    ("kairos", "kairos"),
    ("configs", "configs"),
    ("README.md", "."),
]

hiddenimports += [
    "kairos", "kairos.astro", "kairos.waves", "kairos.gann",
    "kairos.market", "kairos.charting", "kairos.paths",
    "scipy.special._cdflib",
    "scipy._lib.array_api_compat.numpy.fft",
    "pandas._libs.tslibs.timedeltas",
    "encodings.idna",
]

# --- Optional machine-learning stack ----------------------------------
if BUILD_FULL:
    print("[spec] FULL build: including the machine-learning stack")
    for pkg in ("torch", "lightgbm", "statsmodels", "sklearn",
                "transformers", "sentence_transformers", "optuna", "feedparser"):
        add_all(pkg, required=False)
    hiddenimports += collect_submodules("stock_forecast")
    datas += [("stock_forecast", "stock_forecast")]
    excludes = ["tkinter", "matplotlib", "PyQt5", "PyQt6", "PySide2", "PySide6",
                "IPython", "jupyter", "notebook", "pytest", "test", "tests"]
else:
    print("[spec] LIGHT build: machine-learning stack excluded")
    # These are excluded rather than merely absent so that if they happen
    # to be installed in the build venv, a stray import cannot silently
    # drag two gigabytes into the bundle.
    excludes = [
        "torch", "torchvision", "torchaudio",
        "transformers", "sentence_transformers", "tokenizers", "safetensors",
        "lightgbm", "optuna", "sklearn", "scikit_learn", "statsmodels",
        "tensorflow", "keras", "onnxruntime",
        "tkinter", "matplotlib", "PyQt5", "PyQt6", "PySide2", "PySide6",
        "IPython", "jupyter", "notebook", "pytest", "test", "tests",
        "sqlalchemy", "alembic",
    ]
    # The benchmark source still ships so it can be run from source later;
    # it is data here, never imported, so nothing it needs is pulled in.
    datas += [("stock_forecast", "stock_forecast")]

# De-duplicate. collect_all across overlapping packages repeats entries,
# and duplicates in a one-file build inflate the archive noticeably.
datas = sorted(set(datas))
binaries = sorted(set(binaries))
hiddenimports = sorted(set(hiddenimports))

print(f"[spec] TOTAL: {len(datas)} data entries, {len(binaries)} binaries, "
      f"{len(hiddenimports)} hidden imports, {len(excludes)} exclusions")

ICON = "icon.ico" if os.path.exists("icon.ico") else None
VERSION_FILE = "version.txt" if os.path.exists("version.txt") else None

a = Analysis(
    [ENTRY],
    pathex=[os.path.abspath(".")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name=APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    # UPX is off deliberately. It corrupts some compiled extension modules,
    # and Windows Defender flags UPX-packed executables far more often than
    # unpacked ones. A larger file that runs is worth more than a smaller
    # one that gets quarantined.
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    # Console kept on purpose. Streamlit logs the local URL and any startup
    # error there; with --windowed a failed launch shows the user nothing
    # at all and looks like the program simply did not open.
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=ICON,
    version=VERSION_FILE,
)
