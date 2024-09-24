import logging
import os

from zeroband.utils.world_info import get_world_info

logger = None


"""
ZERO_BAND_LOG_LEVEL=DEBUG allow to control the log level for all ranks
ZERO_BAND_LOG_ALL_RANK=true allow to control if all ranks should log or only the local rank 0
"""


class CustomFormatter(logging.Formatter):
    def __init__(self, local_rank: int):
        super().__init__()
        self.local_rank = local_rank

    def format(self, record):
        log_format = "{asctime} [{levelname}] [Rank {local_rank}] {message}"
        formatter = logging.Formatter(log_format, style="{", datefmt="%H:%M:%S")
        record.local_rank = self.local_rank  # Add this line to set the local rank in the record
        return formatter.format(record)


def get_logger():
    global logger  # Add this line to modify the global logger variable
    if logger is not None:
        return logger

    world_info = get_world_info()
    logger = logging.getLogger(__name__)

    log_level = os.getenv("ZERO_BAND_LOG_LEVEL", "INFO")

    if world_info.local_rank == 0:
        logger.setLevel(level=getattr(logging, log_level, logging.INFO))
    else:
        if os.getenv("ZERO_BAND_LOG_ALL_RANK", "false").lower() == "true":
            logger.setLevel(level=getattr(logging, log_level, logging.INFO))
        else:
            logger.setLevel(level=logging.CRITICAL)  # Disable logging for non-zero ranks

    handler = logging.StreamHandler()
    handler.setFormatter(CustomFormatter(world_info.local_rank))
    logger.addHandler(handler)
    logger.propagate = False  # Prevent the log messages from being propagated to the root logger

    return logger
