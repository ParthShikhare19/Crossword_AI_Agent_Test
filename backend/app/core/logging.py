"""
Structured application logging.

The assignment requires enough observability to inspect retrieval results,
tool calls, conversation context, final responses, errors, and handoffs.

This module provides a single logger configuration so individual components
do not need to configure logging independently.
"""

import logging
import sys

from app.core.config import settings


def configure_logging() -> None:
    """
    Configure application-wide logging.

    Logs are written to stdout so they can be consumed easily by local
    terminals, Docker, or a production log collector.
    """

    log_level = logging.DEBUG if settings.debug else logging.INFO

    logging.basicConfig(
        level=log_level,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(name)s | "
            "%(message)s"
        ),
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


logger = logging.getLogger("aster_row")