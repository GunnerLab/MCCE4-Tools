#!/usr/bin/env python3

import logging
import sys


def setup_logging(level: str = "INFO"):
    """Configure logger to screen & file."""
    log_format = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                                   datefmt='%m-%d %H:%M')
    screen_handler = logging.StreamHandler(sys.stdout)
    screen_handler.setFormatter(log_format)
    file_handler = logging.FileHandler("ms_hbnets.log", encoding="utf-8")
    file_handler.setFormatter(log_format)
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        #format=log_format,
        handlers=[screen_handler, file_handler],
    )


setup_logging()