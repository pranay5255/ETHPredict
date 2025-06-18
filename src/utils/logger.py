from contextlib import contextmanager
from pathlib import Path
import sys
from loguru import logger


@contextmanager
def init_logging(cfg):
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    logger.add(log_dir / "pipeline.jsonl", level="INFO", rotation="100 MB", serialize=True)
    try:
        yield logger
    finally:
        logger.info("run complete")

