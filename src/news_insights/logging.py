from __future__ import annotations

import logging
from typing import Optional

from rich.logging import RichHandler


def setup_logging(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("news_insights")
    if logger.handlers:
        return logger
    logger.setLevel(level.upper())

    handler = RichHandler(rich_tracebacks=True, markup=True)
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Silence noisy libraries a bit
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)

    return logger

