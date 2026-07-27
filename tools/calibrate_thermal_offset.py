"""
Thermal -> RGB spatial calibration tool.

WHY a constant (dx, dy) offset isn't enough:
A fixed offset is only correct if both cameras have identical focal length,
identical sensor size, and are perfectly parallel. Cheap thermal + RGB
camera pairs almost never match on all three -- so a target near the frame
edge will be misaligned even after you tune the offset for the center.
This tool instead fits a similarity transform (rotation + uniform scale +
translation) from several point correspondences, which handles focal
length mismatch and slight camera rotation, not just a shift.

HOW TO USE:
  1. Set up your final camera mounting first (this calibration is only
     valid for a fixed camera rig -- redo it if you physically move either
     camera).
  2. Get a single small heat source you can also see in RGB: a lit
     lighter/incense stick, a soldering iron tip, or just your fingertip
     held close to the thermal lens works. It must show up as a clear hot
     blob in the thermal feed AND be visually identifiable in RGB.
  3. Run this script:
         python -m tools.calibrate_thermal_offset
     Two windows open: "RGB - click point" and "THERMAL - click point".
  4. Move the heat source to a new position (cover multiple regions of the
     frame: corners + center, at least 5 positions), hold it still, then:
       a. Click its location in the RGB window.
       b. Click its location in the THERMAL window.
     Each pair is captured automatically once both clicks are registered.
  5. After 5+ pairs, press 's' to solve and save the transform, or 'q' to
     quit without saving. Aim for points spread across the whole frame,
     not clustered in the center -- that's what actually catches scale/
     rotation mismatch.

Output:
  models/thermal_to_rgb_transform.json  containing a 2x3 affine matrix.
  fusion.py automatically uses this file if present, falling back to the
  constant THERMAL_OFFSET_X/Y in config.py only if it's missing.
"""

import json
import os

import cv2
import numpy as np

from anti_drone_pipeline import config

RGB_WIN = "RGB - click point"
THERMAL_WIN = "THERMAL - click point"
OUT_PATH = os.path.join("models", "thermal_to_rgb_transform.json")

rgb_click = None
thermal_click = None
pairs_rgb = []
pairs_thermal = []


def on_rgb_click(event, x, y, flags, param):
    global rgb_click
    if event == cv2.EVENT_LBUTTONDOWN:
        rgb_click = (x, y)
        print(f"RGB click: {rgb_click}")


def on_thermal_click(event, x, y, flags, param):
    global thermal_click
    if event == cv2.EVENT_LBUTTONDOWN:
        thermal_click = (x, y)
        print(f"Thermal click: {thermal_click}")


def try_capture_pair():
    global rgb_click, thermal_click
    if rgb_click is not None and thermal_click is not None:
        pairs_rgb.append(rgb_click)
        pairs_thermal.append(thermal_click)
        print(f"--> Pair #{len(pairs_rgb)} captured: "
              f"RGB={rgb_click}  THERMAL={thermal_click}")
        rgb_click = None
        thermal_click = None


def solve_and_save():
    if len(pairs_rgb) < 3:
        print(f"Need at least 3 point pairs, only have {len(pairs_rgb)}. Not saving.")
        return False

    src = np.array(pairs_thermal, dtype=np.float32)  # thermal points
    dst = np.array(pairs_rgb, dtype=np.float32)       # corresponding rgb points

    # Similarity transform: rotation + uniform scale + translation.
    # Robust to a few noisy clicks via RANSAC.
    matrix, inliers = cv2.estimateAffinePartial2D(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=5.0
    )
    if matrix is None:
        print("Transform solve failed -- points may be degenerate (collinear). "
              "Recollect with more spread-out positions.")
        return False

    n_inliers = int(inliers.sum()) if inliers is not None else len(pairs_rgb)
    print(f"Solved transform using {n_inliers}/{len(pairs_rgb)} inlier pairs.")
    print(matrix)

    os.makedirs("models", exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump({"matrix_2x3": matrix.tolist(),
                   "n_points": len(pairs_rgb),
                   "n_inliers": n_inliers}, f, indent=2)
    print(f"Saved to {OUT_PATH}")
    return True


def main():
    rgb_cap = cv2.VideoCapture(config.RGB_CAM_INDEX)
    thermal_cap = cv2.VideoCapture(config.THERMAL_CAM_INDEX)
    rgb_cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    rgb_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    thermal_cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    thermal_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    cv2.namedWindow(RGB_WIN)
    cv2.namedWindow(THERMAL_WIN)
    cv2.setMouseCallback(RGB_WIN, on_rgb_click)
    cv2.setMouseCallback(THERMAL_WIN, on_thermal_click)

    print("Move the heat source, click its position in BOTH windows, repeat.")
    print("Press 's' to solve+save once you have >=5 pairs (min 3 accepted).")
    print("Press 'q' to quit without saving.")

    while True:
        ok1, rgb_frame = rgb_cap.read()
        ok2, thermal_frame = thermal_cap.read()
        if not (ok1 and ok2):
            continue

        disp_rgb = rgb_frame.copy()
        disp_thermal = thermal_frame.copy()

        for i, (p) in enumerate(pairs_rgb):
            cv2.circle(disp_rgb, p, 5, (0, 255, 0), -1)
            cv2.putText(disp_rgb, str(i + 1), p, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        for i, (p) in enumerate(pairs_thermal):
            cv2.circle(disp_thermal, p, 5, (0, 255, 0), -1)
            cv2.putText(disp_thermal, str(i + 1), p, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if rgb_click:
            cv2.circle(disp_rgb, rgb_click, 6, (0, 0, 255), 2)
        if thermal_click:
            cv2.circle(disp_thermal, thermal_click, 6, (0, 0, 255), 2)

        cv2.putText(disp_rgb, f"Pairs captured: {len(pairs_rgb)}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow(RGB_WIN, disp_rgb)
        cv2.imshow(THERMAL_WIN, disp_thermal)

        try_capture_pair()

        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            if solve_and_save():
                break
        elif key == ord("q"):
            print("Quit without saving.")
            break

    rgb_cap.release()
    thermal_cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
