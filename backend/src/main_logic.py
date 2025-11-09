"""
main_logic.py
High-level controller that runs ARCNet processing loop in a worker thread.

It exposes:
- start_arcnet(): begin processing loop in background thread
- stop_arcnet(): signal the loop to stop
- frame_generator(): yields JPEG frames for MJPEG streaming
- get_latest_frame(): retrieve the latest composed frame (BGR numpy) or None
"""

import threading
import time
import cv2
import numpy as np
from src.capture import init_camera, read_frame, release_camera
from src.detect_and_process import capture_background, detect_cloak, apply_invisibility

# Global controller state (module-level for simplicity)
_worker_thread = None        # worker thread object
_worker_stop = False         # signal flag to stop loop
_latest_frame_jpeg = None    # latest composed frame encoded to JPEG (bytes)
_state_lock = threading.Lock()
_camera = None               # camera object
_background = None           # captured background (BGR numpy)


def _processing_loop(hsv_ranges):
    """
    Internal loop that reads frames, computes mask, composes frame, and stores JPEG.
    hsv_ranges: list of (lower, upper) tuples to combine (e.g., two red ranges).
    """
    global _worker_stop, _latest_frame_jpeg, _camera, _background

    try:
        # Ensure camera and background exist
        if _camera is None:
            _camera = init_camera()
        if _background is None:
            _background = capture_background(_camera)

        while not _worker_stop:
            # Read a fresh frame
            frame = read_frame(_camera)

            # Build mask from all provided ranges and sum them
            masks = []
            for lower, upper in hsv_ranges:
                masks.append(detect_cloak(frame, lower, upper))
            # Combine masks (cv2.add handles overflow safely)
            mask = masks[0]
            for m in masks[1:]:
                mask = cv2.add(mask, m)

            # Compose invisibility effect
            composed = apply_invisibility(frame, _background, mask)

            # Encode composed frame to JPEG for lightweight transmission
            ret, jpeg = cv2.imencode('.jpg', composed, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ret:
                # Skip storing if encoding failed
                time.sleep(0.01)
                continue

            # Store latest JPEG bytes under lock to make it thread-safe for API access
            with _state_lock:
                _latest_frame_jpeg = jpeg.tobytes()

            # Small sleep to yield CPU (tune this for desired FPS)
            time.sleep(0.02)

    except Exception as exc:
        # On unexpected error, print and set stop flag so system can be restarted safely
        print("[ARCNet] Processing loop error:", exc)
        _worker_stop = True
    finally:
        # Cleanup camera resource on exit
        try:
            if _camera is not None:
                release_camera(_camera)
        finally:
            _camera = None


def start_arcnet(hsv_ranges):
    """
    Start the processing worker in a background thread.
    hsv_ranges: list of (lower_numpy_array, upper_numpy_array)
    Returns True if worker started, False if already running.
    """
    global _worker_thread, _worker_stop, _latest_frame_jpeg, _background, _camera

    if _worker_thread and _worker_thread.is_alive():
        return False  # already running

    # Reset control flags/state
    _worker_stop = False
    _latest_frame_jpeg = None

    # Start worker thread
    _worker_thread = threading.Thread(target=_processing_loop, args=(hsv_ranges,), daemon=True)
    _worker_thread.start()
    return True


def stop_arcnet():
    """
    Signal the processing loop to stop and wait for thread to join.
    """
    global _worker_stop, _worker_thread
    _worker_stop = True
    if _worker_thread:
        _worker_thread.join(timeout=5.0)
        _worker_thread = None
    return True


def frame_generator():
    """
    Generator that yields bytes suitable for an MJPEG HTTP response:
    Each yield is a multipart frame:
      b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg_bytes + b'\r\n'
    If no frame is available yet, yields a small placeholder or waits.
    """
    global _latest_frame_jpeg
    while not _worker_stop:
        with _state_lock:
            data = _latest_frame_jpeg
        if data is None:
            # No frame yet; wait briefly to avoid busy loop
            time.sleep(0.05)
            continue
        # Yield a single multipart MJPEG frame
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + data + b'\r\n')
        # Small delay influences client frame rate; adjust as needed
        time.sleep(0.01)


def get_latest_frame():
    """
    Returns the latest BGR numpy frame decoded from JPEG bytes, or None.
    Useful if you want to access the frame server-side rather than streaming.
    """
    global _latest_frame_jpeg
    with _state_lock:
        data = _latest_frame_jpeg
    if data is None:
        return None
    # Decode back to numpy image
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img