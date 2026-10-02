from loguru import logger
import logging
import os
import sys
import time

# Set timezone (tzset is only available on Unix; on Windows the local time zone is used)
os.environ['TZ'] = 'Europe/Vienna'
if hasattr(time, 'tzset'):
    time.tzset()

# Ensure the log directory exists (relative to the working directory unless LOG_DIR is set)
LOG_DIR = os.environ.get("LOG_DIR", "log")
os.makedirs(LOG_DIR, exist_ok=True)

# Remove any preconfigured handlers from loguru
logger.remove(0)

# Define a more readable log format
log_format = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "Process: <cyan>{process:>5}</cyan> | "
    "<level>{level: <8}</level> | "
    "<level>{message}</level>"
)

# Add logging to stdout with colors
logger.add(sys.stdout, format=log_format, colorize=True)

# Add logging to file, named after the script, without color codes
script_name = os.path.basename(sys.argv[0]).replace(".py", ".log")
logger.add(
    os.path.join(LOG_DIR, script_name),
    format=log_format,
    colorize=False,  # Ensure no color codes are written to the file
    backtrace=True,
    diagnose=True
)

# Logging function
def debug_msg(msg, level=logging.INFO):
    if level == logging.INFO:
        logger.info(msg)
    elif level == logging.DEBUG:
        logger.debug(msg)
    elif level == logging.WARNING:
        logger.warning(msg)
    elif level == logging.CRITICAL:
        logger.critical(msg)
    elif level == logging.ERROR:
        logger.error(msg)

# Example usage
if __name__ == "__main__":
    debug_msg("This is an info message.")
    debug_msg("This is a debug message.", level=logging.DEBUG)
    debug_msg("Warning message!", level=logging.WARNING)
    debug_msg("Critical issue!", level=logging.CRITICAL)
