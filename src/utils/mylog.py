import os
import time
from .paths import Path


class MyLog(object):

    def timestamp():
        localtime = time.localtime()
        format_str = r"%y%m%d_%H%M%S"
        timestamp = time.strftime(format_str, localtime)
        return timestamp

    timestamp = timestamp()

    @staticmethod
    def open_log(suffix=None):
        if suffix is None:
            file_name = MyLog.timestamp+"_log"
        else:
            file_name = MyLog.timestamp+"_"+suffix
        file_path = os.path.join(Path.log, file_name)
        f = open(file_path, "a")
        return f


if __name__ == "__main__":
    log = MyLog.open_log()
    print(MyLog.timestamp, file=log)
    log.close()
