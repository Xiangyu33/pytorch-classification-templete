import logging


def get_logger(logger_tag):
    """
    logger_tag:name of logger

    USEAGE:
            logger = get_logger(name)
            logger.info(msg =" ")
            loger.debug(msg =" ")
    """
    logger = logging.getLogger(logger_tag)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s  |  %(filename)s:%(lineno)s  |  %(message)s",
    )

    # output to log file
    fh = logging.FileHandler("log/log.txt", mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    # output to terminal
    sh = logging.StreamHandler()
    sh.setLevel(logging.DEBUG)
    sh.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


if __name__ == "__main__":
    logger = get_logger("test_logger")
    for i in range(100):
        logger.info("this is a line ")
