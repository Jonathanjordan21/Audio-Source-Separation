import logging
import os
from datetime import datetime

LOG_FILENAME = f"{datetime.now().strftime('%m_%d_%Y_%H_%M')}.log" 
LOG_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

LOG_PATH = os.path.join(LOG_DIR, LOG_FILENAME)

logging.basicConfig(
    filename=LOG_PATH,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

logging.basicConfig(
    filename=LOG_PATH,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.DEBUG
)
