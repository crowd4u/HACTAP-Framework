import logging
from colorlog import ColoredFormatter

_default_handler = None


def get_logger(name='hactap'):
    global _default_handler

    if not _default_handler:
        formatter = ColoredFormatter(
            "%(log_color)s[%(levelname)1.1s %(asctime)s]%(reset)s %(message)s",
            datefmt=None,
            reset=True,
            log_colors={
                'DEBUG':    'cyan',
                'INFO':     'green',
                'WARNING':  'yellow',
                'ERROR':    'red',
                'CRITICAL': 'red,bg_white',
            },
            secondary_log_colors={},
            style='%'
        )

        _default_handler = logging.StreamHandler()
        _default_handler.setFormatter(formatter)

        library_root_logger = logging.getLogger(name)
        library_root_logger.addHandler(_default_handler)
        library_root_logger.setLevel(logging.WARNING)
        library_root_logger.propagate = False

    return logging.getLogger(name)
