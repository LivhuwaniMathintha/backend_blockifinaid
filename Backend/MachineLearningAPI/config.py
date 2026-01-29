import os
import sys
from dotenv import load_dotenv
import logging

# configure environment variables
load_dotenv(dotenv_path=".env.development", override=False, verbose=True)

RANDOM_FOREST_CLASSIFIER_FRAUD_MODEL = os.getenv("RANDOM_FOREST_CLASSIFIER_FRAUD_MODEL")
RABBITMQ_URL = os.getenv("RABBITMQ_URL")


# configure logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logger_formatter = logging.Formatter(log_format)

# create console handler and set level to debug
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logger_formatter)

# create file handler and set level to debug
file_handler = logging.FileHandler('app.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logger_formatter)