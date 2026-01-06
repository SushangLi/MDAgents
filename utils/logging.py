"""
Logging configuration using Rich for beautiful console output.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install as install_rich_traceback


# Install rich traceback handler for better error messages
install_rich_traceback(show_locals=True)


def setup_logger(
    name: str = "multiagents",
    level: str = "INFO",
    log_file: Optional[Path] = None,
) -> logging.Logger:
    """
    Setup logger with Rich handler for console output.

    Args:
        name: Logger name
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path to also log to a file

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level.upper())

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Create Rich console that outputs to stderr (important for MCP servers)
    console = Console(file=sys.stderr, stderr=True)

    # Create Rich console handler
    console_handler = RichHandler(
        console=console,
        rich_tracebacks=True,
        show_time=True,
        show_path=True,
        markup=True,
    )
    console_handler.setLevel(level.upper())

    # Format for console
    console_format = logging.Formatter(
        "%(message)s",
        datefmt="[%X]",
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # Add file handler if log_file is specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level.upper())

        # Format for file (more detailed)
        file_format = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "multiagents") -> logging.Logger:
    """
    Get existing logger or create new one.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        # If logger doesn't exist, create it with default settings
        return setup_logger(name)
    return logger


# Create default console for direct Rich output (uses stderr to avoid MCP conflicts)
console = Console(file=sys.stderr, stderr=True)
