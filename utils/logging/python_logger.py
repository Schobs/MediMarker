import logging


def initialize_logging():
    # This sets the root logger to write to stdout (your console).
    # Your script/app needs to call this somewhere at least once.
    logging.basicConfig()

    # By default the root logger is set to WARNING and all loggers you define
    # inherit that value. Here we set the root logger to NOTSET. This logging
    # level is automatically inherited by all existing and new sub-loggers
    # that do not set a less verbose level.
    logging.root.setLevel(logging.NOTSET)

    # The following line sets the root logger level as well.
    # It's equivalent to both previous statements combined:
    logging.basicConfig(level=logging.NOTSET)


def get_logger(save_log_path, name=None):
    # logging.basicConfig(filename='myapp.log', level=logging.INFO)

    if name is None:
        logger = logging.getLogger()
    else:
        logger = logging.getLogger(name)
        
    logger.setLevel(logging.DEBUG)  # <<< Added Line
    # create console handler and set level to info
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)

    # create file handler and set level to info
    file_handler = logging.FileHandler(filename=save_log_path)
    file_handler.setLevel(logging.DEBUG)

    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s -  %(module)s - %(message)s')

    # add formatters to our handlers
    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # add Handlers to our logger
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger
