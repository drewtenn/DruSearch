"""Standard logging setup for pipeline jobs."""

from __future__ import annotations

import logging
import os


def configure(name: str = "drusearch") -> logging.Logger:
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    return logging.getLogger(name)
