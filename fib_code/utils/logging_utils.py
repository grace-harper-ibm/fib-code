import datetime
import logging
import os


def new_logger_for_classic_fib_code_decoder(
    log_location_path,
    name,
    log_level,
):
    tt = datetime.datetime.now()
    unique_log_id = name + "_" + str(tt)
    if log_level == logging.NOTSET:  # Then turn off logging
        logger = logging.getLogger(
            unique_log_id
        )  # TODO -- find better way to  not log output
        logger.addFilter(lambda record: 0)
        logger.setLevel(log_level)
        return logger
    # Create a custom logger
    logger = logging.getLogger(unique_log_id)  # TODO -- find better way to log output
    f_handler = logging.FileHandler(
        os.path.join(log_location_path, unique_log_id + "ClassicFibCode_probs.log")
    )  # TODO remove hardcoded logs, make init resonsible for creating log dir tho
    f_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    f_handler.setFormatter(f_format)
    f_handler.setLevel(log_level)
    logger.addHandler(f_handler)
    logger.setLevel(log_level)
    return logger
