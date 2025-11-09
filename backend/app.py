"""
app.py
Flask backend server for ARCNet – Adaptive Reconstruction and Cloaking Network.

Responsibilities:
- Start/stop ARCNet’s invisibility process via API routes.
- Stream the processed frames as MJPEG video.
"""

from flask import Flask, Response, jsonify
import numpy as np
import threading  # Required for managing concurrency
from src.main_logic import start_arcnet, stop_arcnet, frame_generator

app = Flask(__name__)

# Define HSV color ranges for cloak detection 
HSV_RANGES = [
    ((0, 120, 70), (10, 255, 255)),   # Lower red range
    ((170, 120, 70), (180, 255, 255)) # Upper red range
]

@app.route('/start',  methods=['GET', 'POST'])
def start():
    """
    Start the ARCNet invisibility process.
    Returns JSON indicating success or if already running.
    """
    started = start_arcnet([(np.array(l), np.array(u)) for l, u in HSV_RANGES])
    if started:
        return jsonify({"status": "started"})
    else:
        return jsonify({"status": "already_running"})


@app.route('/stop',  methods=['GET', 'POST'])
def stop():
    """
    Stop the ARCNet invisibility process and release the camera.
    """
    stop_arcnet()
    return jsonify({"status": "stopped"})


@app.route('/video_feed')
def video_feed():
    """
    Stream MJPEG video feed from ARCNet processing.
    Can be displayed directly in an HTML <img> tag.
    """
    return Response(frame_generator(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def home():
    """
    Optional home route — simple message.
    """
    return "<h3>ARCNet Backend is Running </h3><p>Use /start, /stop, or /video_feed.</p>"


if __name__ == '__main__':
    # host=0.0.0.0 allows network access (for deployment/testing on other devices)
    app.run(host='0.0.0.0', port=5000, debug=True)
