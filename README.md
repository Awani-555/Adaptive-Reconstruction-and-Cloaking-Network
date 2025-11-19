# ARCNet – Adaptive Reconstruction and Cloaking Network

**ARCNet** is a computer vision-based system that uses background subtraction and color segmentation to create an *invisibility cloak effect* — inspired by advanced visual occlusion and object reconstruction techniques used in AR/VR and surveillance.

---

##  Features
- Real-time background subtraction using OpenCV
- HSV-based color detection (default: red cloak)
- Morphological mask refinement for smooth detection
- Frame-by-frame blending to create a seamless invisible effect
- Modular code structure for easy extension into full-stack apps

---

##  Architecture Overview

### 1. Background Capture
Captures a clean static frame (without the person).  
This is used as the reference background for substitution.

### 2. Cloak Detection
Uses **HSV color space** to detect a specific color range (red by default).  
The system supports multiple hue bands for complex colors (e.g., two red ranges).

### 3. Morphological Processing
Applies Gaussian blur and morphological open/dilate operations  
to remove noise and smooth out cloak edges.

### 4. Background Replacement
Replaces the detected cloak area with the captured background  
— giving the illusion of invisibility.

---
 Future Scope
ARCNet can evolve into:

Full-stack web app for live AR video streaming and background replacement.

AI-powered segmentation using U-Net or DeepLab for real-time cloaking without color dependency.

Virtual Meeting Integration — allow background invisibility or replacement during video calls.

Security & Surveillance — object disappearance/reconstruction in real-time monitoring.

 Tech Stack & Dependencies

Python 3.8+

OpenCV (cv2)

NumPy

Flask (for backend streaming API)

