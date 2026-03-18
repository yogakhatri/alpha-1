"""Central logger factory.

All modules call this helper so the project uses a consistent log format.
"""

import logging

_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    """Return configured logger for the given module name."""
    global _CONFIGURED
    if not _CONFIGURED:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
        _CONFIGURED = True
    return logging.getLogger(name)
