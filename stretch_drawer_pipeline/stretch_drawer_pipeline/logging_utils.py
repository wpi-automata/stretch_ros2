"""Per-node file logging for the drawer pipeline."""

import sys
from datetime import datetime
from pathlib import Path

from rclpy.logging import LoggingSeverity

_LOGS_DIR = Path(__file__).resolve().parent / "logs"

_SEVERITY_NAMES = {
    LoggingSeverity.DEBUG: "DEBUG",
    LoggingSeverity.INFO:  "INFO",
    LoggingSeverity.WARN:  "WARN",
    LoggingSeverity.ERROR: "ERROR",
    LoggingSeverity.FATAL: "FATAL",
}


class _TeeStream:
    """Writes to both the original stream and a log file."""

    def __init__(self, original, log_file):
        self._original = original
        self._log_file = log_file

    def write(self, data):
        self._original.write(data)
        self._log_file.write(data)
        self._log_file.flush()

    def flush(self):
        self._original.flush()
        self._log_file.flush()

    def fileno(self):
        return self._original.fileno()

    def isatty(self):
        return self._original.isatty()


def setup_file_logging(node):
    """Wrap *node*'s ROS2 logger to also write to a file, and tee stdout/stderr.

    Call once, right after ``super().__init__(...)``.
    """
    _LOGS_DIR.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = _LOGS_DIR / f"{node.get_name()}_{stamp}.log"
    log_file = open(log_path, "a")

    logger = node.get_logger()
    original_log = logger.log

    def wrapped_log(message, severity, **kwargs):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        level = _SEVERITY_NAMES.get(severity, str(severity))
        log_file.write(f"[{ts}] [{level}] {message}\n")
        log_file.flush()
        return original_log(message, severity, **kwargs)

    logger.log = wrapped_log

    sys.stdout = _TeeStream(sys.stdout, log_file)
    sys.stderr = _TeeStream(sys.stderr, log_file)

    node.get_logger().info(f"Logging to {log_path}")
