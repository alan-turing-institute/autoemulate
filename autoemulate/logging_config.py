import logging


def configure_logging(verbose=False):
    # Set a basic configuration for the root logger
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(message)s")

    # Capture warnings from the `warnings` module into the logging system
    logging.captureWarnings(True)

    # Adjust the logger for this package specifically
    logger = logging.getLogger("autoemulate")
    logger.setLevel(
        logging.INFO if verbose else logging.ERROR
    )  # Always INFO for the package

    # Adjust other loggers based on the verbose setting
    # if verbose == 0:
    #     for logger_name in logging.root.manager.loggerDict:
    #         if logger_name != 'autoemulate':
    #             logging.getLogger(logger_name).setLevel(logging.WARNING)

    # If you want to redirect logs to a file
    # file_handler = logging.FileHandler('autoemulate.log', mode='w')
    # file_handler.setLevel(logging.INFO if verbose else logging.WARNING)
    # logger.addHandler(file_handler)
