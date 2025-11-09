#This module handles camera initialization and frame capture using OpenCV.

import cv2

def init_camera(index: int = 0, width: int = 640, height: int = 480) -> cv2.VideoCapture:
    """ 
    Initialize and return a cv2.VideoCapture object.
    - index: camera index (0 default)
    - width, height: desired capture resolution 
    """ 
   
    # Use CAP_DSHOW on Windows to avoid long delays when opening cameras (optional).
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)  # CAP_DSHOW helps on Windows; optional on others
    
    # Immediately check whether opening the camera worked. If camera didn't open, raise an exception for the caller to handle.
    if not cap.isOpened():
        raise RuntimeError("Could not open camera (index {}).".format(index))

    # Setting resolution (may be ignored by some cameras)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)  # 3 is the code for CAP_PROP_FRAME_HEIGHT
    return cap


def read_frame(cap: cv2.VideoCapture):
    """
    Read one frame from the provided VideoCapture `cap`.
    Returns a BGR image (numpy array) on success.
    Raises RuntimeError if read fails.
    """
    ret, frame = cap.read()
    if not ret:
        # A failed read is critical; raise so higher-level code can decide to reconnect or stop.
        raise RuntimeError("Failed to read frame from camera.")
    return frame

def release_camera(cap: cv2.VideoCapture):
    """
    Release camera and close any OpenCV windows (safe cleanup).
    This function ignores exceptions to make cleanup idempotent.
    """
    try:
        cap.release()
    except (cv2.error, RuntimeError):
        pass
