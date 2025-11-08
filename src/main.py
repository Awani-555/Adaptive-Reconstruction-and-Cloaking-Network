from capture import init_camera, read_frame, release_camera
from detect_and_process import capture_background, detect_cloak, apply_invisibility
import cv2
import numpy as np

def main():
    """
    Main controller for ARCNet project.
    This script:
    1. Captures background.
    2. Detects cloak color in real-time.
    3. Applies invisibility effect dynamically.
    """
    # Step 1: Initialize webcam
    cap = init_camera()

    # Step 2: Capture background
    background = capture_background(cap)

    # Step 3: Define color range (here, red cloak)
    # These HSV ranges can be changed for other colors (e.g., blue, green)
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    print("[INFO] Starting cloak detection... Press 'q' to exit.")

    # Step 4: Continuous real-time processing
    while True:
        frame = read_frame(cap)

        # Detect red cloak regions using both hue ranges
        mask1 = detect_cloak(frame, lower_red, upper_red)
        mask2 = detect_cloak(frame, lower_red2, upper_red2)
        mask = mask1 + mask2

        # Apply invisibility effect
        final_frame = apply_invisibility(frame, background, mask)

        # Display results
        cv2.imshow("ARCNet - Cloak Mask", mask)
        cv2.imshow("ARCNet - Original Feed", frame)
        cv2.imshow("ARCNet - Invisibility Effect", final_frame)

        # Quit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Exiting ARCNet...")
            break

    # Step 5: Clean up resources
    release_camera(cap)


if __name__ == "__main__":
    main()
