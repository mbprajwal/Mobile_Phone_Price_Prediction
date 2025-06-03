import logging

def setup_logging(log_file):
    logging.basicConfig(
        filename=log_file,
        filemode="a",
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )

def log_info(msg):
    print(msg)
    logging.info(msg)

def log_error(msg):
    print(msg)