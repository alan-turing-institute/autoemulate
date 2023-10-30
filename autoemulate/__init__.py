import logging

logging.basicConfig(
    level=logging.INFO,
    # format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    format="%(message)s",
    # filename='autoemulate.log'
)
logging.captureWarnings(True)
