"""
kairos.paths
============
Where things live, whether running from source or from a frozen one-file
executable.

PyInstaller's one-file mode unpacks the bundle into a temporary directory
that is deleted on exit. Read-only resources therefore have to be looked
up under `sys._MEIPASS`, while anything the app writes — caches, exports,
config the user edits — has to go somewhere persistent next to the .exe.
Getting these two mixed up is the classic reason a frozen app "works
until you close it".
"""
from __future__ import annotations

import os
import sys


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def bundle_dir() -> str:
    """
    Read-only resources bundled at build time.

    Frozen one-file: the temp extraction directory.
    Frozen one-dir:  the directory holding the executable.
    Source:          the project root.
    """
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        return str(meipass)
    if is_frozen():
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def app_dir() -> str:
    """
    Writable directory that survives between runs: alongside the .exe when
    frozen, the project root otherwise.
    """
    if is_frozen():
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _ensure(path: str) -> str:
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass
    return path


def data_dir() -> str:
    return _ensure(os.path.join(app_dir(), "data"))


def cache_dir() -> str:
    return _ensure(os.path.join(data_dir(), "cache"))


def artifacts_dir() -> str:
    return _ensure(os.path.join(app_dir(), "artifacts"))


def features_dir() -> str:
    return _ensure(os.path.join(app_dir(), "features"))


def exports_dir() -> str:
    return _ensure(os.path.join(app_dir(), "exports"))


def resource(*parts: str) -> str:
    """
    Locate a bundled read-only resource, preferring the writable app
    directory so a user can override a bundled config by dropping their
    own copy next to the executable.
    """
    local = os.path.join(app_dir(), *parts)
    if os.path.exists(local):
        return local
    return os.path.join(bundle_dir(), *parts)


def config_path(name: str = "default.yaml") -> str:
    return resource("configs", name)


def app_script() -> str:
    """The Streamlit script the launcher hands to the bootstrap."""
    return resource("gui", "app.py")
