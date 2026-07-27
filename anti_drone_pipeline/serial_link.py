"""
Thin wrapper around pyserial. Runs a background thread that continuously
reads lines from the Arduino so radar samples never block the main
detection loop, and exposes simple send_* methods for commands.
"""

import threading
import queue
import time
import serial

from . import config


class SerialLink:
    def __init__(self, port=None, baud=None, timeout=None):
        self.port = port or config.SERIAL_PORT
        self.baud = baud or config.SERIAL_BAUD
        self.timeout = timeout or config.SERIAL_TIMEOUT_S

        self._ser = serial.Serial(self.port, self.baud, timeout=self.timeout)
        time.sleep(2.0)  # let the Arduino reset after the port opens

        self.radar_queue = queue.Queue(maxsize=200)
        self._ack_queue = queue.Queue(maxsize=50)

        self._stop_flag = threading.Event()
        self._reader_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._reader_thread.start()

    # ---------------- background reader ----------------
    def _read_loop(self):
        buf = b""
        while not self._stop_flag.is_set():
            try:
                chunk = self._ser.read(256)
            except serial.SerialException:
                break
            if not chunk:
                continue
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                self._handle_line(line.decode(errors="ignore").strip())

    def _handle_line(self, line):
        if not line:
            return
        parts = line.split(",")
        tag = parts[0].upper()

        if tag == "RADAR" and len(parts) == 3:
            try:
                angle = float(parts[1])
                dist_cm = float(parts[2])
                self.radar_queue.put_nowait((angle, dist_cm))
            except (ValueError, queue.Full):
                pass
        elif tag in ("ACK", "PONG"):
            try:
                self._ack_queue.put_nowait(line)
            except queue.Full:
                pass
        # Unknown lines are ignored (e.g. boot banner text)

    # ---------------- commands to Arduino ----------------
    def _send(self, line: str):
        self._ser.write((line + "\n").encode())

    def send_pan_tilt(self, pan_deg: float, tilt_deg: float):
        self._send(f"PT,{int(round(pan_deg))},{int(round(tilt_deg))}")

    def set_jammer(self, on: bool):
        self._send(f"JAM,{1 if on else 0}")

    def set_laser(self, on: bool):
        self._send(f"LASER,{1 if on else 0}")

    def set_radar_sweeping(self, on: bool):
        self._send(f"RADAR,{1 if on else 0}")

    def ping(self, timeout=1.0) -> bool:
        # Drain stale acks first
        while not self._ack_queue.empty():
            self._ack_queue.get_nowait()
        self._send("PING")
        try:
            reply = self._ack_queue.get(timeout=timeout)
            return reply.upper() == "PONG"
        except queue.Empty:
            return False

    # ---------------- radar consumption ----------------
    def get_radar_sample(self, block=False, timeout=None):
        """Returns (angle_deg, dist_cm) or None if nothing is queued."""
        try:
            return self.radar_queue.get(block=block, timeout=timeout)
        except queue.Empty:
            return None

    def drain_radar(self):
        """Return all currently queued radar samples as a list."""
        samples = []
        while not self.radar_queue.empty():
            samples.append(self.radar_queue.get_nowait())
        return samples

    def close(self):
        self._stop_flag.set()
        self._reader_thread.join(timeout=1.0)
        self._ser.close()
