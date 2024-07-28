import os
import time
import logging


def create_logger(logger_name: str, log_file_path: os.PathLike = None):
    """
    Create a logger with the specified name and log file path.
    """
    logger = logging.getLogger(logger_name)
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    assert log_file_path is not None, "log_file_path is required"
    fh = logging.FileHandler(log_file_path)
    fh_formatter = logging.Formatter("%(asctime)s : %(levelname)s, %(funcName)s Message: %(message)s")
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)
    logger.info(f"logging start: {logger_name}")
    return logger


class Timer:
    """
    A simple timer class for measuring elapsed time.
    """

    def __init__(self, filename: os.PathLike = "timer_log.log", reset: bool = False):
        """
        Initialize the Timer object.
        """
        self.start_time = None
        self.last_checkpoint = None
        self.filename = filename
        self.logger = create_logger("Timer", filename)
        if reset:
            self._reset_log_file()

    def _reset_log_file(self):
        """
        Reset the log file by clearing its contents.
        """
        with open(self.filename, "w") as file:
            file.write("")

    def start(self):
        """
        Start the timer.
        """
        self.start_time = time.time()
        self.last_checkpoint = self.start_time
        self.logger.info("Timer started.")

    def check(self, message):
        """
        Log a checkpoint with the current time and time since the last checkpoint.

        Args:
            message (str): The message to include in the log.
        """
        if self.start_time is None:
            self.logger.warning("Timer has not been started.")
        else:
            log_message = (
                f"Current time count: {time.time() - self.start_time:.4f} seconds, "
                f"Time since last checkpoint: {time.time() - self.last_checkpoint:.4f} seconds, "
                f"for {message}"
            )
            self.last_checkpoint = time.time()
            self.logger.info(log_message)

    def stop(self):
        """
        Stop the timer and log the elapsed time.
        """
        if self.start_time is None:
            self.logger.warning("Timer has not been started.")
        else:
            self.end_time = time.time()
            self.logger.info(f"Total elapsed time: {self.end_time - self.start_time} seconds\n")


if __name__ == "__main__":
    # Test the Timer class
    timer = Timer(filename="timer_log.log", reset=True)
    timer.start()
    timer.check("First checkpoint")
    time.sleep(1)
    timer.check("Second checkpoint")
    timer.stop()
