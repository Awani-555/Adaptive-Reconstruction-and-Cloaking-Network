"""
Combine detection (HSV color segmentation) and compositing (background substitution).

This file contains:
- capture_background(): produce a stable background reference
- detect_cloak(): produce a refined binary mask for a given HSV range
- apply_invisibility(): compose frame+background using the mask
"""




import cv2
import numpy as np
import time

def capture_background(cap, num_frames: int = 60, delay_sec: float = 2.0):
    """
    Capture a static background image before the person enters the frame.
    The background will later replace the cloak area.
    """
    print("[INFO] Capturing background... Please stay out of frame.")
    time.sleep(delay_sec)  # Give user time to move out of view

    frames = []
    collected = 0

    while collected < num_frames:
        ret, frame = cap.read()
        if not ret:
        # If read fails, just continue trying - transient read failures can happen.
            continue
        frames.append(frame.astype(np.float32))
        collected += 1

    background = np.median(np.stack(frames, axis=0), axis=0).astype(np.uint8)
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


def apply_invisibility(frame: np.ndarray, background: np.ndarray, mask: np.ndarray):
    """
    Replaces the detected cloak area with the static background
    to create the 'invisible' effect.

    Includes basic safety checks for real-world robustness.
    """
    #  Step 1: Ensure background and frame match in size
    if background.shape[:2] != frame.shape[:2]:
        background = cv2.resize(background, (frame.shape[1], frame.shape[0]))

    #  Step 2: Ensure mask is single-channel (grayscale)
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    #  Step 3: Invert mask to get visible and invisible areas
    mask_inv = cv2.bitwise_not(mask)

    #  Step 4: Extract non-cloak region from current frame
    visible_part = cv2.bitwise_and(frame, frame, mask=mask_inv)

    #  Step 5: Extract cloak region from static background
    invisible_part = cv2.bitwise_and(background, background, mask=mask)

    #  Step 6: Merge both to create the invisibility illusion
    final_output = cv2.add(visible_part, invisible_part)
    return final_output