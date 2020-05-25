import logging


def get_logger(name='hactap'):
    logging.basicConfig(
        level=logging.DEBUG,
        format='[%(levelname)s - %(asctime)s] %(message)s'
    )

    return logging.getLogger(name)
