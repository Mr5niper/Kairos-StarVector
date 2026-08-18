"""
Kairos StarVector - launcher
============================
Single entry point for both `python kairos_app.py` and the frozen
executable built by BUILD_EXE.bat.

Why a launcher instead of `streamlit run gui/app.py`
----------------------------------------------------
Streamlit is a server, not a script. Inside a PyInstaller bundle there is
no `streamlit` console script on PATH and no interpreter to re-invoke, so
the usual command line cannot work. The app has to be started in-process
through Streamlit's own bootstrap, with the script path pointing into the
unpacked bundle.

Three things here exist purely because of the frozen case:

  * `multiprocessing.freeze_support()` must run first. Without it, any
    library that spawns a process causes the executable to relaunch itself
    from the top, and you get an endless cascade of windows.
  * The file watcher is disabled. It would otherwise try to stat frozen
    modules that have no real path on disk.
  * The port is probed rather than assumed, because a stale Streamlit
    process holding 8501 would make the app appear to hang.
"""
from __future__ import annotations

import multiprocessing
import os
import socket
import sys
import threading
import time
import webbrowser

DEFAULT_PORT = 8501
PORT_ATTEMPTS = 25


def _find_free_port(start: int = DEFAULT_PORT, attempts: int = PORT_ATTEMPTS) -> int:
    for offset in range(attempts):
        port = start + offset
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    return 0  # let the OS choose


def _open_browser_later(url: str, delay: float = 2.5) -> None:
    def go() -> None:
        time.sleep(delay)
        try:
            webbrowser.open_new_tab(url)
        except Exception:
            pass
    threading.Thread(target=go, daemon=True).start()


def main() -> int:
    multiprocessing.freeze_support()

    # Import after freeze_support so a re-executed child exits cleanly first.
    from kairos import paths as P

    script = P.app_script()
    if not os.path.exists(script):
        sys.stderr.write(
            f"Could not find the application script at:\n  {script}\n\n"
            "If running from source, launch this file from the project root.\n"
            "If running the executable, it was built without --add-data for "
            "the gui folder; rebuild with BUILD_EXE.bat.\n"
        )
        return 2

    port = _find_free_port()
    url = f"http://localhost:{port}"

    # Streamlit reads a good deal of configuration from the environment,
    # and env vars are honoured before any config file, which makes them
    # the reliable channel in a frozen app that has no home directory.
    os.environ.setdefault("STREAMLIT_GLOBAL_DEVELOPMENT_MODE", "false")
    os.environ.setdefault("STREAMLIT_SERVER_HEADLESS", "true")
    os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")
    os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    os.environ.setdefault("STREAMLIT_SERVER_PORT", str(port))
    os.environ.setdefault("STREAMLIT_SERVER_ADDRESS", "127.0.0.1")
    os.environ.setdefault("STREAMLIT_THEME_BASE", "dark")
    os.environ.setdefault("STREAMLIT_CLIENT_TOOLBAR_MODE", "viewer")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

    flag_options = {
        "server.port": port,
        "server.address": "127.0.0.1",
        "server.headless": True,
        "server.fileWatcherType": "none",
        "browser.gatherUsageStats": False,
        "global.developmentMode": False,
    }

    print("=" * 62)
    print("  Kairos StarVector")
    print("=" * 62)
    print(f"  Script : {script}")
    print(f"  URL    : {url}")
    print("  Close this window to stop the server.")
    print("=" * 62)

    _open_browser_later(url)

    from streamlit.web import bootstrap

    try:
        bootstrap.load_config_options(flag_options=flag_options)
    except Exception:
        # Older and newer releases have moved this around; the environment
        # variables set above already carry the same settings.
        pass

    try:
        bootstrap.run(script, False, [], flag_options)
        return 0
    except TypeError:
        # Signature drift between Streamlit versions. Fall back to the CLI
        # entry point, which accepts the same options as arguments.
        from streamlit.web import cli as stcli
        sys.argv = [
            "streamlit", "run", script,
            f"--server.port={port}",
            "--server.address=127.0.0.1",
            "--server.headless=true",
            "--server.fileWatcherType=none",
            "--browser.gatherUsageStats=false",
            "--global.developmentMode=false",
        ]
        return int(stcli.main() or 0)
    except KeyboardInterrupt:
        print("\nStopped.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
