import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(name: str = "JEPA", log_file: Optional[str] = None, level=logging.INFO):
    """
    Sets up a logger with standard formatting.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Check if handlers exist to avoid duplicates
    if logger.hasHandlers():
        return logger

    formatter = logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # Console Handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File Handler
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
