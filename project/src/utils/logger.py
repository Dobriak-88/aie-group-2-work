import logging
import sys
from .helpers import LOGS_PATH

def setup_logger(name: str = "car_price", level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # Консоль
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    # Файл – внутри src/logs/
    LOGS_PATH.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(LOGS_PATH / "car_price.log", encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        setup_logger()
    return logger

setup_logger()