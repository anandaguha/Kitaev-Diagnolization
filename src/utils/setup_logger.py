import logging
import os
from typing import Optional, Any, List
from pathlib import Path
from datetime import datetime


PROJECT_ROOT = Path(__file__).resolve().parent.parent

def __normalize_log_level__(level: Optional[str | int]) -> int:
    if isinstance(level, str):
        level = level.upper()
        if level in logging._nameToLevel:
            return logging._nameToLevel[level]
        raise ValueError(f"Invalid log level string: {level}")

    if isinstance(level, int):
        if level in logging._levelToName:
            return level
        raise ValueError(f"Invalid log level int: {level}")

    raise TypeError("Log level must be a string or integer")

def setup_logger(Log_path: Optional[str]= None, Log_level:Optional[str | int] = None):
    try:
        if Log_level is None:
            Log_level = __normalize_log_level__("INFO")
        else:
            Log_level = __normalize_log_level__(Log_level)
    except ValueError:
        print(f"Wrong value was given please pick a valid value next time!\nPossible values{logging._nameToLevel}\nUsing default value for now")
        Log_level = __normalize_log_level__("INFO")
    except TypeError:
        print(f"Please pass a string or a int!\nUsing a default value for logging")
        Log_level = __normalize_log_level__("INFO")
    
    logger = logging.getLogger("Kitaev Lattice")
    logger.setLevel(Log_level)
    

    if not logger.handlers:
        if Log_path is None:
            Log_path = PROJECT_ROOT / "run_logs" / f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        os.makedirs(Log_path.parent, exist_ok=True)
        file_handler = logging.FileHandler(Log_path, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(filename)s:%(lineno)d | %(message)s"
        ))
        logger.addHandler(file_handler)

    return logger

