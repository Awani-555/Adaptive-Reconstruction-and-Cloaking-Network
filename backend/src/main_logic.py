"""
main_logic.py
High-level controller for ARCNet – Adaptive Reconstruction and Cloaking Network.

Responsibilities:
- Initialize camera and capture background.
- Run invisibility processing loop in a background thread.
- Provide start/stop control and video streaming interface.
"""

import threading
import time
import cv2
import numpy as np
from src.capture import init_camera, read_frame, release_camera
from src.detect_and_process import capture_background, detect_cloak, apply_invisibility

# ---------------------------
# Global Controller Variables
# ---------------------------

_worker_thread = None        # Thread object for the processing loop
_worker_stop = False         # Signal flag to stop the loop
_latest_frame_jpeg = None    # Stores the latest composed JPEG frame (bytes)
_state_lock = threading.Lock()  # Ensures thread-safe access to shared variables
_camera = None               # Active camera object
_background = None           # Captured static background


# ---------------------------
# Internal Processing Function
# ---------------------------

def _processing_loop(hsv_ranges):
    """
    Core loop that continuously captures frames, applies cloak detection, and
    encodes the final composite (invisible) frame into JPEG bytes.
    """

    global _worker_stop, _latest_frame_jpeg, _camera, _background

    try:
        # Initialize camera safely
        if _camera is None:
            _camera = init_camera()

        # Capture background before starting
        if _background is None:
            _background = capture_background(_camera)

        print("[ARCNet] Processing loop started.")

        while not _worker_stop:
            try:
                frame = read_frame(_camera)
                if frame is None:
                    print("[ARCNet] Warning: Empty frame received. Reinitializing camera...")
                    release_camera(_camera)
                    _camera = init_camera()
                    continue

                # --- Cloak detection across multiple HSV ranges ---
                masks = []
                for lower, upper in hsv_ranges:
                    masks.append(detect_cloak(frame, lower, upper))

                # Combine all masks safely
                mask = masks[0]
                for m in masks[1:]:
                    mask = cv2.add(mask, m)

                # --- Apply invisibility effect ---
                composed = apply_invisibility(frame, _background, mask)

                # --- Encode the final frame to JPEG format ---
                success, jpeg = cv2.imencode('.jpg', composed, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                if not success:
                    print("[ARCNet] JPEG encoding failed, skipping frame.")
                    time.sleep(0.01)
                    continue

                # Thread-safe frame update
                with _state_lock:
                    _latest_frame_jpeg = jpeg.tobytes()

                # Sleep for stability (controls FPS ~30–40)
                time.sleep(0.02)

            except cv2.error as e:
                print("[ARCNet] OpenCV error:", e)
                time.sleep(0.1)
                continue
            except RuntimeError as e:
                print("[ARCNet] Runtime error:", e)
                time.sleep(0.1)
                continue

    except Exception as exc:
        print("[ARCNet] Fatal processing loop error:", exc)
        _worker_stop = True

    finally:
        # Ensure proper camera release
        if _camera is not None:
            release_camera(_camera)
            _camera = None
        print("[ARCNet] Processing loop stopped and camera released.")


# ---------------------------
# Public Control Functions
# ---------------------------

def start_arcnet(hsv_ranges):
    """
    Start the ARCNet invisibility process in a background thread.
    Args:
        hsv_ranges (list): List of (lower_bound, upper_bound) NumPy arrays.
    Returns:
        bool: True if started successfully, False if already running.
    """
    global _worker_thread, _worker_stop, _latest_frame_jpeg, _background

    if _worker_thread and _worker_thread.is_alive():
        print("[ARCNet] Attempted to start, but worker is already running.")
        return False

    print("[ARCNet] Starting ARCNet processing thread...")
    _worker_stop = False
    _latest_frame_jpeg = None
    _background = None  # Force new background capture each start

    _worker_thread = threading.Thread(target=_processing_loop, args=(hsv_ranges,), daemon=True)
    _worker_thread.start()
    return True


def stop_arcnet():
    """
    Stop the ARCNet invisibility process and clean up resources.
    """
    global _worker_stop, _worker_thread

    print("[ARCNet] Stopping ARCNet processing...")
    _worker_stop = True

    if _worker_thread:
        _worker_thread.join(timeout=5.0)
        _worker_thread = None

    print("[ARCNet] ARCNet successfully stopped.")
    return True


def frame_generator():
    """
    Generator that yields MJPEG frames for streaming.
    Used by Flask to serve `/video_feed`.
    """
    global _latest_frame_jpeg

    print("[ARCNet] Starting video stream generator...")

    while not _worker_stop:
        with _state_lock:
            data = _latest_frame_jpeg

        if data is None:
            # Wait briefly until first frame is ready
            time.sleep(0.05)
            continue

        # Yield formatted MJPEG frame
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + data + b'\r\n')

        time.sleep(0.02)  # Adjust frame rate

    print("[ARCNet] Video stream generator stopped.")


def get_latest_frame():
    """
    Returns the latest frame as a BGR NumPy image (for internal server usage).
    Returns None if no frame is available yet.
    """
    global _latest_frame_jpeg

    with _state_lock:
        data = _latest_frame_jpeg

    if data is None:
        return None

    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img
    