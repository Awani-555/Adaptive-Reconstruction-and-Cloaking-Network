"""
app.py
Flask backend server for ARCNet – Adaptive Reconstruction and Cloaking Network.

Responsibilities:
- Serve frontend UI (HTML + static files).
- Start/stop ARCNet invisibility process via API.
- Stream processed frames as MJPEG video feed.
"""

from flask import Flask, Response, jsonify, render_template
import numpy as np
import threading
from src.main_logic import start_arcnet, stop_arcnet, frame_generator

# Create Flask app instance
app = Flask(__name__, template_folder='templates', static_folder='static')

# HSV color ranges for red cloak detection
HSV_RANGES = [
    ((0, 120, 70), (10, 255, 255)),   # Lower red
    ((170, 120, 70), (180, 255, 255)) # Upper red
]

@app.route('/')
def home():
    """Render frontend UI."""
    return render_template('index.html')

@app.route('/start', methods=['POST'])
def start():
    """Start ARCNet invisibility thread."""
    started = start_arcnet([(np.array(l), np.array(u)) for l, u in HSV_RANGES])
    return jsonify({"status": "started" if started else "already_running"})

@app.route('/stop', methods=['POST'])
def stop():
    """Stop ARCNet invisibility process."""
    stop_arcnet()
    return jsonify({"status": "stopped"})

@app.route('/video_feed')
def video_feed():
    """Stream live MJPEG frames."""
    return Response(frame_generator(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)