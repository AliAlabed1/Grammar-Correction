import logging
import os

# Define the log directory relative to this file's location
curr = os.getcwd()
curr = curr.split('src')[0]
LOG_DIR = f"{curr}logs"
print(f'log is{LOG_DIR}')
print("log Direcotre abspath" ,LOG_DIR)

os.makedirs(LOG_DIR, exist_ok=True)

def setup_logger(name, log_file, level=logging.INFO):
    """Sets up a logger with specified name, log file, and loggind level."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Create a file handler for loggind
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)

    # Create a console handler for logging 
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    # Define log massage formate
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

# Instantiate loggers
app_logger = setup_logger(name="app_logger", log_file=os.path.join(LOG_DIR, "app.log"))