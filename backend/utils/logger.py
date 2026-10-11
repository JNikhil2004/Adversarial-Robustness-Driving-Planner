"""
Shared application logger.

A single configured logger keeps log formatting consistent across every
robustness / attack / evaluation module instead of each file rolling its own.
"""
import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger with a consistent, readable format."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger
