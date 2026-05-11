import logging


def configure_logging(level: str = "INFO"):
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(level=numeric, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
import logging


def configure_logging(level: str = "INFO"):
    level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


__all__ = ["configure_logging"]
