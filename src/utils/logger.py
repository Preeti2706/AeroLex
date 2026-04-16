"""
AeroLex — Centralized Structured Logger
Every module imports this logger.

Usage:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Something happened")
    logger.error("Something broke", exc_info=True)
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ── Constants ────────────────────────────────────────────────────────────────

# Log level from .env — default to DEBUG in development
LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG").upper()

# Log directory — create if it doesn't exist
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Log file name with date — new file every day
LOG_FILE = os.path.join(LOG_DIR, f"aerolex_{datetime.now().strftime('%Y%m%d')}.log")

# ── Formatter ────────────────────────────────────────────────────────────────
# This is what each log line looks like
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str) -> logging.Logger:
    """
    Returns a configured logger for the given module name.

    Args:
        name: Usually __name__ of the calling module

    Returns:
        logging.Logger: Configured logger instance

    Why RotatingFileHandler?
        Log files can grow huge in production. RotatingFileHandler automatically
        creates a new file when current file hits max size (5MB here).
        Keeps last 5 backup files — so total max 25MB of logs.
        Alternative: TimedRotatingFileHandler (rotates daily) — better for production.
        We use size-based here for simplicity during development.
    """

    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if logger already configured
    # This happens when same module is imported multiple times
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, LOG_LEVEL, logging.DEBUG))

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    # ── Handler 1: Console (stdout) ───────────────────────────────────────
    # Shows logs in terminal during development
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, LOG_LEVEL, logging.DEBUG))
    console_handler.setFormatter(formatter)

    # ── Handler 2: Rotating File Handler ─────────────────────────────────
    # Writes logs to file — persists after terminal closes
    # maxBytes=5MB, backupCount=5 means max 25MB total log storage
    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=5,
        encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)  # Always log everything to file
    file_handler.setFormatter(formatter)

    # ── Handler 3: Error File Handler ────────────────────────────────────
    # Separate file ONLY for ERROR and CRITICAL — easy to spot problems
    error_log_file = os.path.join(LOG_DIR, f"aerolex_errors_{datetime.now().strftime('%Y%m%d')}.log")
    error_handler = RotatingFileHandler(
        error_log_file,
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8"
    )
    error_handler.setLevel(logging.ERROR)  # Only ERROR and CRITICAL
    error_handler.setFormatter(formatter)

    # Add all handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.addHandler(error_handler)

    return logger


# ── Module-level test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Quick test — run: python src/utils/logger.py
    logger = get_logger(__name__)

    logger.debug("DEBUG — Detailed info for diagnosing problems")
    logger.info("INFO — Confirming things are working as expected")
    logger.warning("WARNING — Something unexpected, but not breaking")
    logger.error("ERROR — Something broke, needs attention")
    logger.critical("CRITICAL — Serious error, system may stop working")

    print(f"\n✅ Logger working! Check log file at: {LOG_FILE}")