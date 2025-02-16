import logging
import datetime
import os
import threading


class CustomLogger(logging.Logger):
    def __init__(self, name, log_file=None, log_level=logging.INFO):
        super().__init__(name, log_level)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(console_formatter)
        self.addHandler(console_handler)

        self.file_handler = None
        self.multithreading = False
        self.log_queues = {}
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            self.file_handler = logging.FileHandler(log_file)
            self.file_handler.setLevel(log_level)
            file_formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            self.file_handler.setFormatter(file_formatter)

        self.lock = threading.Lock()

    def set_multithreading(self):
        self.multithreading = True

    def disable_multithreading(self):
        self.multithreading = False

    def _log_with_file_option(
        self,
        level,
        msg,
        args,
        exc_info=None,
        extra=None,
        stack_info=False,
        to_file=True,
    ):
        """Logs a message to console and conditionally file if not
        multithreading, otherwise adds messages to a list to be
        flushed later."""
        if not self.multithreading:
            super()._log(level, msg, args, exc_info, extra, stack_info)

            if to_file and self.file_handler:
                self.file_handler.handle(
                    self.makeRecord(
                        self.name, level, None, None, msg, args, exc_info, extra
                    )
                )
        else:
            if threading.get_ident() not in self.log_queues:
                self.log_queues[threading.get_ident()] = []
            self.log_queues[threading.get_ident()].append((level, msg, args))

    def flush_thread_log_queue(self, thread_id):
        """Flush the log queue for the given thread."""
        with self.lock:
            if thread_id in self.log_queues:
                for level, msg, args in self.log_queues[thread_id]:
                    super()._log(level, msg, args)
                    if self.file_handler:
                        self.file_handler.handle(
                            self.makeRecord(
                                self.name, level, None, None, msg, args, None, None
                            )
                        )
                # Clear the log queue for this thread after flushing
                del self.log_queues[thread_id]

    def flush_all_threads(self):
        thread_ids = list(self.log_queues.keys())
        for thread_id in thread_ids:
            self.flush_thread_log_queue(thread_id)

    def info(self, msg, *args, **kwargs):
        to_file = kwargs.pop("to_file", True)
        self._log_with_file_option(logging.INFO, msg, args, to_file=to_file, **kwargs)

    def warn(self, msg, *args, **kwargs):
        to_file = kwargs.pop("to_file", True)
        self._log_with_file_option(logging.WARN, msg, args, to_file=to_file, **kwargs)

    def debug(self, msg, *args, **kwargs):
        to_file = kwargs.pop("to_file", True)
        self._log_with_file_option(logging.DEBUG, msg, args, to_file=to_file, **kwargs)

    def error(self, msg, *args, **kwargs):
        to_file = kwargs.pop("to_file", True)
        self._log_with_file_option(logging.ERROR, msg, args, to_file=to_file, **kwargs)


def reinitialize_logger():
    global logger
    log_file_path = (
        f"./logs/{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.log"
    )
    logger = CustomLogger(name="div_gen", log_file=log_file_path)


log_file_path = f"./logs/{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.log"
logger = CustomLogger(name="div_gen", log_file=log_file_path)
console_logger = CustomLogger(name="div_gen_console_logger", log_file=None)
