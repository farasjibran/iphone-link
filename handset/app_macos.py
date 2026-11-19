"""macOS launcher: runs the server and opens iPhone Link in its own app window.

Mirrors :mod:`handset.app` (the Windows launcher) but for macOS: the server
runs in this process and the UI opens in a Chromium-based browser in
``--app`` mode, so it looks like a native window without a terminal.

No tray icon and no auto-updater -- both are Windows-only in this project.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from handset.server import HOST, PORT, Handler, SingleServer, create_http_server

PORT = int(os.environ.get("PANE_PORT", PORT))
# The server binds HOST (0.0.0.0 on some setups), but the browser and the
# "is it already running?" probe both need a concrete address to reach.
LOCAL_HOST = "127.0.0.1"
URL = f"http://{LOCAL_HOST}:{PORT}/"
DATA_DIR = Path(os.environ.get("XDG_DATA_HOME", Path.home() / "Library/Application Support")) / "iPhone Link"

# Chromium-based browsers that support --app=<url>. Checked in order; the
# first one installed wins. The value is the executable inside the .app bundle.
CHROMIUM_APPS = (
    "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
)


def _setup_logging() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(DATA_DIR / "iphone-link.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.basicConfig(level=logging.INFO, handlers=[handler])
    if sys.stdout is None or sys.stderr is None:
        log = open(DATA_DIR / "iphone-link.log", "a", encoding="utf-8", buffering=1)
        sys.stdout = sys.stdout or log
        sys.stderr = sys.stderr or log


def _already_running() -> bool:
    from handset.netutil import port_is_in_use

    return port_is_in_use(LOCAL_HOST, PORT)


def _chromium_binary() -> str | None:
    """Return the path of an installed Chromium-based browser, or None."""
    for app in CHROMIUM_APPS:
        if Path(app).is_file():
            return app
    # A browser installed elsewhere but on PATH still works.
    for name in ("brave-browser", "google-chrome", "chromium", "chromium-browser"):
        found = shutil.which(name)
        if found:
            return found
    return None


def _wait_for_server(timeout: float = 10.0) -> None:
    """Block until the server accepts connections, so the window loads a live page."""
    from handset.netutil import port_is_in_use

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if port_is_in_use(LOCAL_HOST, PORT):
            return
        time.sleep(0.1)


def _open_window() -> subprocess.Popen | None:
    """Open the UI in an app-style browser window.

    Returns the browser process, or None when the UI was handed to the
    default browser (which we cannot track, so we must not wait on it).
    """
    browser = _chromium_binary()
    if browser:
        args = [
            browser,
            f"--app={URL}",
            f"--user-data-dir={DATA_DIR / 'window'}",
            "--window-size=1280,860",
            "--no-first-run",
            "--no-default-browser-check",
        ]
        try:
            return subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError as exc:
            logging.warning("Could not open %s: %s", browser, exc)

    # Fall back to the default browser.
    try:
        subprocess.Popen(["open", URL])
    except OSError as exc:
        logging.warning("Could not open the default browser: %s", exc)
    return None


def main() -> None:
    _setup_logging()
    if _already_running():
        _open_window()
        return

    server = create_http_server(host=HOST, port=PORT)
    threading.Thread(target=server.serve_forever, name="pane-server", daemon=True).start()
    logging.info("iPhone Link listening on %s", URL)

    _wait_for_server()
    window = _open_window()
    if window is None:
        # The UI is in the default browser; keep serving.
        while True:
            time.sleep(3600)

    # Chromium hands a launch request to an already-running instance and exits
    # straight away with code 0. That means the window lives in another
    # process, so exiting here would take the server down with it -- keep
    # serving instead. Only a browser we launched ourselves that later exits
    # counts as "the user closed the window".
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if window.poll() is not None:
            break
        time.sleep(0.2)

    if window.poll() is not None and window.returncode == 0:
        logging.info("Browser handed the window to a running instance; keeping the server up")
        while True:
            time.sleep(3600)

    # The app window was closed.
    server.shutdown()
    os._exit(0)


if __name__ == "__main__":
    main()
