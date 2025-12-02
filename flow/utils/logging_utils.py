#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   logging_utils.py
@Time    :   2025/08/01 11:15:49
@Author  :   Dawn Li
@Version :   1.0
@Contact :   dawnli619215645@gmail.com
@Desc    :   Unified logging for flow pipeline.
"""

import logging
import os
import warnings


def is_main_process() -> bool:
    """Check if the current process is the main process"""
    return os.environ.get("LOCAL_RANK", "0") == "0"


class ParallelLogger(logging.Logger):
    """
    A custom logger class that extends logging.Logger.
    It automatically checks if the current process is the main process
    before deciding to emit a log record, based on the logger's effective level
    and the record's level.
    """

    def __init__(self, name, level=logging.NOTSET):
        super().__init__(name, level)

    def callHandlers(self, record):
        if not is_main_process():
            return
        super().callHandlers(record)


def configure_third_party_loggers():
    """Configure third-party loggers to reduce noise"""
    # Reduce logging from transformers, datasets, etc.
    for logger_name in [
        "unsloth",
        "DeepSpeed",
        "accelerate",
        "transformers",
        "torch",
        "datasets",
    ]:
        third_party_logger = logging.getLogger(logger_name)
        third_party_logger.setLevel(logging.WARNING)

    # Filter warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)


def setup_logging(
    level: str = "INFO",
    enable_detailed_logs: bool = False,
    use_parallel_logger: bool = True,
) -> None:
    """
    Setup global logging configuration for the flow pipeline.

    This function configures the root logger and sets up the logging system
    to work with distributed training and provide colored output.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        enable_detailed_logs: Whether to show detailed logs from third-party libraries
        use_colors: Whether to use colored output in terminal
        use_parallel_logger: Whether to use ParallelLogger for distributed training

    Example:
        >>> setup_logging(level="INFO", use_colors=True)
        >>> logger = logging.getLogger("my_module")
        >>> logger.info("This will be colored and distributed-training aware")
    """

    # Set up ParallelLogger as default logger class if requested
    if use_parallel_logger:
        logging.setLoggerClass(ParallelLogger)

    # Only configure once and only in main process
    if is_main_process():
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        logging.info("🛠️  Logging is set up.")

    # Configure third-party loggers if not in detailed mode
    if not enable_detailed_logs:
        configure_third_party_loggers()
