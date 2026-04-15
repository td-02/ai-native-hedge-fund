from __future__ import annotations

import logging
import sys

import structlog


_configured = False


def configure_logging() -> None:
    global _configured
    if _configured:
        return
    timestamper = structlog.processors.TimeStamper(fmt="iso", utc=True)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            timestamper,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    _configured = True


def get_logger(name: str):
    configure_logging()
    return structlog.get_logger(name)

