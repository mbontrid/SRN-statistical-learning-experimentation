import logging
import sys
from pathlib import Path


def get_logger(name: str = "logger"):
    return logging.getLogger(name)


class Logger(logging.Logger):
    def __init__(
        self,
        name: str = "logger",
        level: int = logging.INFO,
        log_file: str | None = None,
    ):
        super().__init__(name, level)
        """
        Setup a simple logger with console and file output.

        Debug: Detailed information for diagnosing problems.
        Info: Confirms things are working as expected.
        Warning: Indicates unexpected issues or potential future problems.
        Error: A serious problem that prevents a function from running.
        Critical: A severe error that may stop the program from running.

        Args:
            level: Logging level as string (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Path to log file

        Returns:
            Configured logger instance
        """
        self.handlers.clear()

        formatter = logging.Formatter(
            # "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
            "%(levelname)s - %(message)s"
        )

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.addHandler(console_handler)

        if log_file is not None:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(filename=log_file, mode="a")
            file_handler.setFormatter(formatter)
            self.addHandler(file_handler)

    # Source - https://stackoverflow.com/a/35804945
    # Posted by Mad Physicist, modified by community. See post 'Timeline' for change history
    # Retrieved 2026-04-27, License - CC BY-SA 4.0
    def addLoggingLevel(self, levelName, levelNum, methodName=None):
        """
        Comprehensively adds a new logging level to the `logging` module and the
        currently configured logging class.

        `levelName` becomes an attribute of the `logging` module with the value
        `levelNum`. `methodName` becomes a convenience method for both `logging`
        itself and the class returned by `logging.getLoggerClass()` (usually just
        `logging.Logger`). If `methodName` is not specified, `levelName.lower()` is
        used.

        To avoid accidental clobberings of existing attributes, this method will
        raise an `AttributeError` if the level name is already an attribute of the
        `logging` module or if the method name is already present

        Example
        -------
        >>> addLoggingLevel('TRACE', logging.DEBUG - 5)
        >>> logging.getLogger(__name__).setLevel("TRACE")
        >>> logging.getLogger(__name__).trace('that worked')
        >>> logging.trace('so did this')
        >>> logging.TRACE
        5
        """
        if not methodName:
            methodName = levelName.lower()

        if hasattr(logging, levelName):
            raise AttributeError(
                "{} already defined in logging module".format(levelName)
            )
        if hasattr(logging, methodName):
            raise AttributeError(
                "{} already defined in logging module".format(methodName)
            )
        if hasattr(logging.getLoggerClass(), methodName):
            raise AttributeError(
                "{} already defined in logger class".format(methodName)
            )

        # This method was inspired by the answers to Stack Overflow post
        # http://stackoverflow.com/q/2183233/2988730, especially
        # http://stackoverflow.com/a/13638084/2988730
        def logForLevel(self, message, *args, **kwargs):
            if self.isEnabledFor(levelNum):
                self._log(levelNum, message, args, **kwargs)

        def logToRoot(message, *args, **kwargs):
            logging.log(levelNum, message, *args, **kwargs)

        logging.addLevelName(levelNum, levelName)
        setattr(logging, levelName, levelNum)
        setattr(logging.getLoggerClass(), methodName, logForLevel)
        setattr(logging, methodName, logToRoot)
