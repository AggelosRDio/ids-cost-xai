from datetime import datetime

class Logger:

    def __init__(self, show_time=True, log_level=0):
        self.show_time = show_time
        self.log_level = log_level
        self.colours = {
            "info": "\033[96m",
            "success": "\033[92m",
            "warning": "\033[93m",
            "error": "\033[91m",
        }
        self.reset = "\033[0m"


    def _log(self, level, message, colour):
        timestamp = f"[{datetime.now().strftime('%H:%M:%S')}] " if self.show_time else ""
        print(colour + f"{timestamp}[{level}]: {message}" + self.reset)

    def blank(self, message):
        if self.log_level <= 0:
            print(message)


    def info(self, message):
        if self.log_level <= 0:
            self._log("info", message, self.colours["info"])

    def success(self, message):
        if self.log_level <= 0:
            self._log("success", message, self.colours["success"])

    def warn(self, message):
        if self.log_level <= 1:
            self._log("warning", message, self.colours["warning"])

    def error(self, message):
        if self.log_level <= 2:
            self._log("error", message, self.colours["error"])