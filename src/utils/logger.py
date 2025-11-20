"""
Logging Configuration
Uses loguru for structured logging across all agents and components
"""

import sys
from pathlib import Path
from loguru import logger
from typing import Optional

def setup_logger(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    rotation: str = "10 MB",
    retention: str = "1 week"
) -> None:
    """
    Configure loguru logger with file and console output
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file. If None, logs to logs/agentic_gaming.log
        rotation: When to rotate log file (size or time)
        retention: How long to keep old logs
    """
    # Remove default handler
    logger.remove()
    
    # Add console handler with color
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=log_level,
        colorize=True
    )
    
    # Add file handler if specified
    if log_file is None:
        log_file = "logs/agentic_gaming.log"
    
    # Create logs directory if it doesn't exist
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=log_level,
        rotation=rotation,
        retention=retention,
        compression="zip"
    )
    
    logger.info(f"Logger initialized with level {log_level}")

def get_logger(name: str):
    """
    Get a logger instance for a specific module
    
    Args:
        name: Name of the module (typically __name__)
    
    Returns:
        logger: Configured logger instance
    """
    return logger.bind(name=name)

# Initialize default logger
setup_logger()