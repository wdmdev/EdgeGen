# utils/logging.py

import logging
import os

# Optionally, specify the directory where log files will be stored
LOG_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'output', 'logging')
os.makedirs(LOG_DIR, exist_ok=True)

# Create handlers
info_handler = logging.FileHandler(os.path.join(LOG_DIR, 'info.log'))
info_handler.setLevel(logging.INFO)

error_handler = logging.FileHandler(os.path.join(LOG_DIR, 'error.log'))
error_handler.setLevel(logging.ERROR)

formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
info_handler.setFormatter(formatter)
error_handler.setFormatter(formatter)

# Define a custom filter to exclude ERROR and above from the info handler
class MaxLevelFilter(logging.Filter):
    def __init__(self, exclusive_max_level):
        super(MaxLevelFilter, self).__init__()
        self.exclusive_max_level = exclusive_max_level

    def filter(self, record):
        # Allow records that are less severe than the exclusive_max_level
        return record.levelno < self.exclusive_max_level

# Add the filter to the info handler
info_handler.addFilter(MaxLevelFilter(logging.ERROR))

def get_logger(name=None):
    """
    Returns a logger with the specified name, configured to log INFO and ERROR levels
    to separate files.

    :param name: Name of the logger (usually __name__)
    :return: Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Set the logger to the lowest level

    # Prevent adding handlers multiple times if get_logger is called multiple times
    if not logger.handlers:
        logger.addHandler(info_handler)
        logger.addHandler(error_handler)
        logger.propagate = False  # Prevent messages from propagating to the root logger

    return logger