import logging
import sys
import os

def configure_logging(level: str = "INFO", log_file: str = "logs/app.log"):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # console
    console = logging.StreamHandler(sys.stdout)

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    console.setFormatter(fmt)
    file_handler.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    root.handlers.clear()
    root.addHandler(console)
    root.addHandler(file_handler)

def get_logger(name:str):
    return logging.getLogger(name)