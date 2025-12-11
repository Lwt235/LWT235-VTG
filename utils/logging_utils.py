"""
Logging utilities for Video Temporal Localization Framework.

Provides consistent logging across all modules.
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

# Global logger registry
_loggers: dict = {}

# Default format
DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logger(
    name: str = "vtg",
    level: Union[int, str] = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
    console: bool = True,
    format_str: str = DEFAULT_FORMAT,
    date_format: str = DEFAULT_DATE_FORMAT,
) -> logging.Logger:
    """
    Set up a logger with console and optional file handlers.
    
    Args:
        name: Logger name.
        level: Logging level (e.g., logging.INFO or "INFO").
        log_file: Optional path to log file.
        console: Whether to output to console.
        format_str: Log message format string.
        date_format: Date format for log messages.
    
    Returns:
        Configured logger instance.
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers = []
    
    formatter = logging.Formatter(format_str, datefmt=date_format)
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    # Register logger
    _loggers[name] = logger
    
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger by name, creating if necessary.
    
    Args:
        name: Logger name. If None, returns the root VTG logger.
    
    Returns:
        Logger instance.
    """
    if name is None:
        name = "vtg"
    
    if name in _loggers:
        return _loggers[name]
    
    # Create a new logger with default settings
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        # Set up with default configuration
        formatter = logging.Formatter(DEFAULT_FORMAT, datefmt=DEFAULT_DATE_FORMAT)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    
    _loggers[name] = logger
    return logger


def create_experiment_logger(
    experiment_name: str,
    output_dir: Union[str, Path],
    level: Union[int, str] = logging.INFO,
) -> logging.Logger:
    """
    Create a logger for a specific experiment with file logging.
    
    Args:
        experiment_name: Name of the experiment.
        output_dir: Directory to save log files.
        level: Logging level.
    
    Returns:
        Configured logger instance.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_dir / f"{experiment_name}_{timestamp}.log"
    
    return setup_logger(
        name=experiment_name,
        level=level,
        log_file=log_file,
        console=True,
    )


class LoggerContextManager:
    """Context manager for temporary logger configuration."""
    
    def __init__(
        self,
        logger: logging.Logger,
        level: Optional[int] = None,
        add_handler: Optional[logging.Handler] = None,
    ):
        self.logger = logger
        self.original_level = logger.level
        self.new_level = level
        self.added_handler = add_handler
    
    def __enter__(self):
        if self.new_level is not None:
            self.logger.setLevel(self.new_level)
        if self.added_handler is not None:
            self.logger.addHandler(self.added_handler)
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.setLevel(self.original_level)
        if self.added_handler is not None:
            self.logger.removeHandler(self.added_handler)
        return False


def silence_loggers(*logger_names: str):
    """
    Silence specific loggers by setting their level to WARNING.
    
    Args:
        logger_names: Names of loggers to silence.
    """
    for name in logger_names:
        logging.getLogger(name).setLevel(logging.WARNING)


def set_verbosity(level: Union[int, str]):
    """
    Set the verbosity level for all VTG loggers.
    
    Args:
        level: Logging level.
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    
    for logger in _loggers.values():
        logger.setLevel(level)
