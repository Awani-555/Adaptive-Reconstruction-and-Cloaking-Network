import cv2
import numpy as np
import time

def capture_background(cap, duration=3):
    """
    Capture a static background image before the person enters the frame.
    The background will later replace the cloak area.
    """
    print("[INFO] Capturing background... Please stay out of frame.")
    time.sleep(2)  # Give user time to move out of view

    for _ in range(duration * 10):  # capture multiple frames for stability
        ret, background = cap.read()
        if not ret:
            raise Exception("Error: Could not read background frame.")
        background = cv2.flip(background, 1)

    print("[INFO] Background captured successfully.")
    return background


def detect_cloak(frame, lower_hsv, upper_hsv):
    """
    Detects the cloak color using HSV color thresholding.
    Returns a clean, refined binary mask of the cloak.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

    # Refine mask using blur and morphological transformations
    mask = cv2.GaussianBlur(mask, (7, 7), 0)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((5, 5), np.uint8))
    return mask


def apply_invisibility(frame, background, mask):
    """
    Replaces the detected cloak area with the static background
    to create the 'invisible' effect.
    """
    # Invert mask → cloak area becomes 0, rest becomes 255
    mask_inv = cv2.bitwise_not(mask)

    # Keep only non-cloak parts of the current frame
    part1 = cv2.bitwise_and(frame, frame, mask=mask_inv)

    # Bring in background for cloak region
    part2 = cv2.bitwise_and(background, background, mask=mask)

    # Combine both parts to get final output
    final_output = cv2.addWeighted(part1, 1, part2, 1, 0)
    return final_output
