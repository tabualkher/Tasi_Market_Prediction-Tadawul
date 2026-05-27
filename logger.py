"""
utils/logger.py — Structured, colored logging for the TASI pipeline
"""

import logging
import sys
from datetime import datetime


COLORS = {
    "DEBUG":    "\033[36m",   # Cyan
    "INFO":     "\033[32m",   # Green
    "WARNING":  "\033[33m",   # Yellow
    "ERROR":    "\033[31m",   # Red
    "CRITICAL": "\033[35m",   # Magenta
    "RESET":    "\033[0m",
    "BOLD":     "\033[1m",
    "DIM":      "\033[2m",
}


class ColoredFormatter(logging.Formatter):
    def format(self, record):
        color = COLORS.get(record.levelname, COLORS["RESET"])
        reset = COLORS["RESET"]
        bold  = COLORS["BOLD"]
        dim   = COLORS["DIM"]

        ts = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        level = f"{color}{bold}{record.levelname:8s}{reset}"
        name  = f"{dim}{record.name}{reset}"
        msg   = record.getMessage()

        return f"{dim}{ts}{reset}  {level}  {name}  {msg}"


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(ColoredFormatter())
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False
    return logger


def section(title: str):
    """Print a visual section divider."""
    width = 60
    line = "─" * width
    print(f"\n{COLORS['BOLD']}{COLORS['DIM']}{line}{COLORS['RESET']}")
    print(f"{COLORS['BOLD']}  {title}{COLORS['RESET']}")
    print(f"{COLORS['BOLD']}{COLORS['DIM']}{line}{COLORS['RESET']}\n")
