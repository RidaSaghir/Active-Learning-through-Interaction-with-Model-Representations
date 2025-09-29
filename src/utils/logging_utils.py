# logging_utils.py
import logging, os
from logging.handlers import RotatingFileHandler
from contextlib import contextmanager
import time

DEFAULT_LOGGER_NAME = "imlvr"

def setup_logging(
    name: str = DEFAULT_LOGGER_NAME,
    level: str | int = None,
    log_dir: str = "logs",
    filename: str = "app.log",
    max_bytes: int = 5_000_000,
    backup_count: int = 3,
) -> logging.Logger:
    """
    One-time setup. Call early in main/server startup.
    """
    lvl = level or os.getenv("LOG_LEVEL", "INFO")
    lvl = getattr(logging, lvl.upper(), logging.INFO)

    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(lvl)
    logger.propagate = False  # don't double-log via root

    # Clear duplicate handlers on reload
    if logger.handlers:
        for h in list(logger.handlers):
            logger.removeHandler(h)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    sh = logging.StreamHandler()
    sh.setLevel(lvl)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = RotatingFileHandler(os.path.join(log_dir, filename),
                             maxBytes=max_bytes, backupCount=backup_count)
    fh.setLevel(lvl)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger

def get_logger(child: str) -> logging.Logger:
    return logging.getLogger(f"{DEFAULT_LOGGER_NAME}.{child}")

@contextmanager
def log_duration(logger: logging.Logger, msg: str, **kv):
    """Usage: with log_duration(log, 'compute embeddings'): ..."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = (time.perf_counter() - t0) * 1000.0
        if kv:
            logger.info(f"{msg} | duration_ms={dt:.2f} | {kv}")
        else:
            logger.info(f"{msg} | duration_ms={dt:.2f}")
