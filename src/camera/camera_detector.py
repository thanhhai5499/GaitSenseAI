"""
Camera Detector Module

This module provides a class for detecting cameras connected to the system.
"""

import time
import logging
import threading
from typing import Dict, List, Callable, Any

from .webcam_camera import WebcamCamera
from .realsense_camera import RealSenseCamera, REALSENSE_AVAILABLE

# Set up logging
logger = logging.getLogger("CameraDetector")

class CameraDetector:
    """
    Class for detecting cameras connected to the system.
    """

    def __init__(self):
        """
        Initialize the camera detector.
        """
        self.available_cameras = {}
        self.callbacks = []

        logger.info("Initialized camera detector")

    def detect_cameras_once(self) -> None:
        """
        Detect cameras connected to the system once.
        """
        logger.info("Performing one-time camera detection")
        try:
            # Detect cameras
            self._detect_cameras()
            logger.info("One-time camera detection completed")
        except Exception as e:
            logger.error(f"Error in one-time camera detection: {e}")

    def _detect_cameras(self) -> None:
        """
        Detect cameras connected to the system.
        """
        try:
            # Get available cameras
            available_cameras = {}

            # Detect webcams
            logger.debug("Detecting webcams...")
            webcams = WebcamCamera.list_available_cameras()
            logger.debug(f"Found {len(webcams)} webcams")

            for webcam in webcams:
                camera_id = f"webcam_{webcam['id']}"
                available_cameras[camera_id] = {
                    'id': camera_id,
                    'name': webcam['name'],
                    'width': webcam['width'],  # Use detected values
                    'height': webcam['height'],
                    'fps': webcam['fps'],
                    'type': 'webcam',
                    'raw_id': webcam['id']
                }
                logger.debug(f"Added webcam: {webcam['name']} (ID: {camera_id}, raw_id: {webcam['id']})")

            # Detect RealSense cameras
            if REALSENSE_AVAILABLE:
                logger.debug("Detecting RealSense cameras...")
                realsense_cameras = RealSenseCamera.list_available_cameras()
                logger.debug(f"Found {len(realsense_cameras)} RealSense cameras")

                for realsense in realsense_cameras:
                    camera_id = f"realsense_{realsense['id']}"
                    available_cameras[camera_id] = {
                        'id': camera_id,
                        'name': realsense['name'],
                        'width': realsense['width'],  # Use detected values
                        'height': realsense['height'],
                        'fps': realsense['fps'],
                        'type': 'realsense',
                        'raw_id': realsense['id']
                    }
                    logger.debug(f"Added RealSense camera: {realsense['name']} (ID: {camera_id}, raw_id: {realsense['id']})")

            # Log current state
            logger.debug(f"Current available cameras: {list(self.available_cameras.keys())}")
            logger.debug(f"New available cameras: {list(available_cameras.keys())}")

            # Check for changes
            added_cameras = {k: v for k, v in available_cameras.items() if k not in self.available_cameras}
            removed_cameras = {k: v for k, v in self.available_cameras.items() if k not in available_cameras}

            # Update available cameras
            self.available_cameras = available_cameras

            # Log changes
            for camera_id, camera in added_cameras.items():
                logger.info(f"Detected new camera: {camera['name']} (ID: {camera_id})")

            for camera_id, camera in removed_cameras.items():
                logger.info(f"Camera disconnected: {camera['name']} (ID: {camera_id})")

            # Notify callbacks if there are changes
            if added_cameras or removed_cameras:
                logger.info(f"Camera changes detected: {len(added_cameras)} added, {len(removed_cameras)} removed")
                self._notify_callbacks(added_cameras, removed_cameras)

            # Always notify callbacks even if no changes (to update UI with latest camera info)
            elif len(available_cameras) > 0:
                logger.debug("No camera changes detected, but notifying callbacks to update UI")
                self._notify_callbacks({}, {})

        except Exception as e:
            logger.error(f"Error detecting cameras: {e}")

    def get_available_cameras(self) -> List[Dict]:
        """
        Get a list of available cameras.

        Returns:
            List[Dict]: List of available cameras
        """
        return list(self.available_cameras.values())

    def register_callback(self, callback: Callable[[Dict[str, Dict], Dict[str, Dict]], None]) -> None:
        """
        Register a callback function to be called when cameras are added or removed.

        Args:
            callback: Callback function that takes two arguments:
                      - added_cameras: Dict of added cameras
                      - removed_cameras: Dict of removed cameras
        """
        if callback not in self.callbacks:
            self.callbacks.append(callback)

    def unregister_callback(self, callback: Callable[[Dict[str, Dict], Dict[str, Dict]], None]) -> None:
        """
        Unregister a callback function.

        Args:
            callback: Callback function to unregister
        """
        if callback in self.callbacks:
            self.callbacks.remove(callback)

    def _notify_callbacks(self, added_cameras: Dict[str, Dict], removed_cameras: Dict[str, Dict]) -> None:
        """
        Notify all registered callbacks about camera changes.

        Args:
            added_cameras: Dict of added cameras
            removed_cameras: Dict of removed cameras
        """
        for callback in self.callbacks:
            try:
                callback(added_cameras, removed_cameras)
            except Exception as e:
                logger.error(f"Error in camera detection callback: {e}")

    def create_camera(self, camera_id: str) -> Any:
        """
        Create a camera object for the given camera ID.

        Args:
            camera_id: Camera ID

        Returns:
            BaseCamera: Camera object or None if the camera is not available
        """
        if camera_id not in self.available_cameras:
            logger.error(f"Camera {camera_id} is not available")
            return None

        camera_info = self.available_cameras[camera_id]
        camera_type = camera_info['type']
        raw_id = camera_info['raw_id']

        try:
            if camera_type == 'webcam':
                # For webcam, raw_id should be an integer
                if isinstance(raw_id, str):
                    try:
                        raw_id = int(raw_id)
                    except ValueError:
                        logger.error(f"Invalid webcam ID: {raw_id}")
                        return None

                logger.info(f"Creating webcam with ID: {raw_id}, name: {camera_info['name']}")
                return WebcamCamera(
                    camera_id=raw_id,
                    name=camera_info['name'],
                    width=1920,  # Force 1920x1080 30fps
                    height=1080,
                    fps=30
                )
            elif camera_type == 'realsense':
                logger.info(f"Creating RealSense camera with ID: {raw_id}, name: {camera_info['name']}")
                return RealSenseCamera(
                    camera_id=raw_id,
                    name=camera_info['name'],
                    width=1920,  # Force 1920x1080 30fps
                    height=1080,
                    fps=30
                )
            else:
                logger.error(f"Unknown camera type: {camera_type}")
                return None
        except Exception as e:
            logger.error(f"Error creating camera {camera_id}: {e}")
            return None
