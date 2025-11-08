#This module handles camera initialization and frame capture using OpenCV.

import cv2

#Initialize the camera. 
def init_camera(index: int = 0, width: int = 640, height: int = 480) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)  # CAP_DSHOW helps on Windows; optional on others
    if not cap.isOpened():
        raise RuntimeError("Could not open camera (index {}).".format(index))

    # Setting resolution (may be ignored by some cameras)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(3, height)  # 3 is the code for CAP_PROP_FRAME_HEIGHT
    return cap

# Read a single frame from the camera.
# Returns the frame (BGR numpy array). Raises Exception on failure.
def read_frame(cap: cv2.VideoCapture):
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Failed to read frame from camera.")
    return frame

# Release camera resources 
def release_camera(cap: cv2.VideoCapture):
    try:
        cap.release()
    except (cv2.error, RuntimeError):
        pass
