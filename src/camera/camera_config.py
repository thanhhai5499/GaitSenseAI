"""
Camera Configuration Module

This module defines the configuration for the cameras used in the system.
"""

# Camera configuration constants
CAMERA_CONFIG = {
    "front": {
        "type": "realsense",
        "id": 0,  # This is typically not used for RealSense but kept for consistency
        "width": 1280,
        "height": 720,
        "fps": 30,
        "name": "Intel RealSense D435i"
    },
    "back": {
        "type": "placeholder",  # No camera assigned yet
        "id": None,
        "width": 1280,
        "height": 720,
        "fps": 30,
        "name": "Placeholder Camera"
    },
    "left": {
        "type": "webcam",
        "id": 1,  # Logitech Brio 300 - Left camera
        "width": 1920,  # Full HD resolution (max for Brio 300)
        "height": 1080,
        "fps": 30,  # 30 FPS (max for Brio 300 at Full HD)
        "name": "Logitech Brio 300 (Left)",
        "buffer_size": 3  # Increased buffer size to prevent frame drops
    },
    "right": {
        "type": "webcam",
        "id": 2,  # Logitech Brio 300 - Right camera
        "width": 1920,  # Full HD resolution (max for Brio 300)
        "height": 1080,
        "fps": 30,  # 30 FPS (max for Brio 300 at Full HD)
        "name": "Logitech Brio 300 (Right)",
        "buffer_size": 3  # Increased buffer size to prevent frame drops
    }
}
