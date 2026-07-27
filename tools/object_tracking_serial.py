"""
Object Tracking -> Angle -> Serial
-----------------------------------
1. Detects an object in the webcam feed (color-based detection by default).
2. Gets its (x, y) pixel center.
3. Converts (x, y) into pan/tilt servo angles.
4. Sends the angles over a serial port (e.g., to an Arduino driving 2 servos).

Requirements:
    pip install opencv-python numpy pyserial
"""

import cv2
import numpy as np
import serial
import time

# ---------------- CONFIGURATION ----------------
SERIAL_PORT = 'COM3'       # Windows: 'COM3', 'COM4'...  Linux/Mac: '/dev/ttyUSB0' or '/dev/ttyACM0'
BAUD_RATE = 9600

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Servo angle limits
PAN_MIN, PAN_MAX = 0, 180
TILT_MIN, TILT_MAX = 0, 180

# HSV color range for detection -> replace with your object's color
# (this example targets a bright green object)
LOWER_COLOR = np.array([29, 86, 6])
UPPER_COLOR = np.array([64, 255, 255])

MIN_CONTOUR_AREA = 300  # ignore small noise blobs
# -------------------------------------------------


def map_range(value, in_min, in_max, out_min, out_max):
    """Linearly map a value from one range to another."""
    return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


def get_object_center(frame):
    """
    Detect the object by color and return its pixel center (x, y).
    Returns None if nothing is found.
    Swap this function out for a YOLO/DNN detector if you need
    more robust detection -- just make it return (x, y) or None.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_COLOR, UPPER_COLOR)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < MIN_CONTOUR_AREA:
        return None

    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy)


def xy_to_angles(x, y, frame_w, frame_h):
    """Convert pixel (x, y) into pan/tilt servo angles."""
    # Flip pan mapping (PAN_MAX -> PAN_MIN) if your servo turns the "wrong" way
    pan = map_range(x, 0, frame_w, PAN_MAX, PAN_MIN)
    tilt = map_range(y, 0, frame_h, TILT_MIN, TILT_MAX)

    pan = int(np.clip(pan, PAN_MIN, PAN_MAX))
    tilt = int(np.clip(tilt, TILT_MIN, TILT_MAX))
    return pan, tilt


def main():
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)  # give the Arduino time to reset after the serial connection opens

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    last_sent = (None, None)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            center = get_object_center(frame)

            if center is not None:
                cx, cy = center
                pan, tilt = xy_to_angles(cx, cy, FRAME_WIDTH, FRAME_HEIGHT)

                # Only write to serial when the angle actually changes
                if (pan, tilt) != last_sent:
                    msg = f"{pan},{tilt}\n"
                    ser.write(msg.encode('utf-8'))
                    last_sent = (pan, tilt)

                # Visual feedback
                cv2.circle(frame, (cx, cy), 6, (0, 255, 0), -1)
                cv2.putText(frame, f"Pan:{pan}  Tilt:{tilt}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Object Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        ser.close()


if __name__ == "__main__":
    main()
