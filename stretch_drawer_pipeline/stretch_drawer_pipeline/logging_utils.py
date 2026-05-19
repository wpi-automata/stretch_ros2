"""Per-node file logging for the drawer pipeline.

Redirects file descriptors 1 (stdout) and 2 (stderr) through a pipe so that
a background thread can tee every byte to both the original terminal and a
per-node log file.  This captures C-level rcutils output, Python print()
calls, and third-party library output (e.g. Detic) without touching the
ROS2 logger at all — no CallerId or filter conflicts.
"""

import os
import sys
import threading
from datetime import datetime
from pathlib import Path

_LOGS_DIR = Path(__file__).resolve().parent / "logs"
_saved_streams = []
_setup_done = False


def setup_file_logging(node):
    """Tee all stdout/stderr to ``logs/<node>_<timestamp>.log``.

    Call once, right after ``super().__init__(...)``.
    """
    global _setup_done
    if _setup_done:
        return
    _setup_done = True

    _LOGS_DIR.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = _LOGS_DIR / f"{node.get_name()}_{stamp}.log"

    sys.stdout.flush()
    sys.stderr.flush()

    # Prevent GC of old Python stream objects (would close the fds)
    _saved_streams.extend([sys.stdout, sys.stderr])

    log_fd = os.open(str(log_path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)

    # Save the original terminal fd
    orig_fd = os.dup(1)

    # Create pipe; redirect both stdout and stderr into its write end
    pipe_r, pipe_w = os.pipe()
    os.dup2(pipe_w, 1)
    os.dup2(pipe_w, 2)
    os.close(pipe_w)

    def _tee():
        try:
            while True:
                data = os.read(pipe_r, 8192)
                if not data:
                    break
                os.write(orig_fd, data)
                os.write(log_fd, data)
        except OSError:
            pass

    threading.Thread(target=_tee, daemon=True).start()

    sys.stdout = os.fdopen(1, "w", closefd=False, buffering=1)
    sys.stderr = os.fdopen(2, "w", closefd=False, buffering=1)

    node.get_logger().info(f"Logging to {log_path}")
