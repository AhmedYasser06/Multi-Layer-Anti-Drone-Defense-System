import csv
import os
import time

from . import config


class EventLogger:
    def __init__(self, log_dir=None):
        self.log_dir = log_dir or config.LOG_DIR
        os.makedirs(self.log_dir, exist_ok=True)
        self.path = os.path.join(
            self.log_dir, f"session_{int(time.time())}.csv"
        )
        self._f = open(self.path, "w", newline="")
        self._writer = csv.writer(self._f)
        self._writer.writerow(
            ["timestamp", "event", "track_id", "cls_name", "score", "detail"]
        )
        self._f.flush()

    def log(self, event, track_id=None, cls_name=None, score=None, detail=""):
        self._writer.writerow(
            [time.strftime("%Y-%m-%d %H:%M:%S"), event, track_id, cls_name, score, detail]
        )
        self._f.flush()

    def close(self):
        self._f.close()
