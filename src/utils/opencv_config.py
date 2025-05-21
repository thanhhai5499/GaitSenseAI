"""
OpenCV Configuration Module

This module configures OpenCV to suppress warning messages.
"""

import cv2
import os
import sys

def disable_opencv_warnings():
    """
    Disable OpenCV warning messages by redirecting stderr temporarily.
    This function should be called before any OpenCV operations.
    """
    # Try to use environment variables to suppress OpenCV warnings
    os.environ["OPENCV_LOG_LEVEL"] = "ERROR"  # Set OpenCV log level to ERROR only

    # Some OpenCV warnings come from the C++ code and can't be suppressed through Python
    # In those cases, we need to suppress stderr output
    class NullWriter:
        def write(self, text):
            # Filter out all OpenCV warnings
            if ("VIDEOIO" in text or "CvCapture" in text or "obsensor" in text or
                "cap_msmf.cpp" in text or "cap.cpp" in text or "WARN:" in text or
                "global" in text or "Failed to select stream" in text or
                "backend is generally available" in text or "cv::VideoCapture" in text):
                return
            # Only write critical errors
            if "ERROR:" in text or "CRITICAL:" in text:
                sys.__stderr__.write(text)
        def flush(self):
            sys.__stderr__.flush()

    # Store the original stderr
    original_stderr = sys.stderr

    # Replace stderr with our null writer
    sys.stderr = NullWriter()

    # Return the original stderr so it can be restored if needed
    return original_stderr

# Call this function to disable OpenCV warnings
original_stderr = disable_opencv_warnings()
