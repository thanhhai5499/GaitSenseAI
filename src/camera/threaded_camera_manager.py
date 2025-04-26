"""
Threaded Camera Manager Module

This module provides a class for managing cameras in separate threads.
"""

import time
import logging
import threading
from typing import Dict, List, Optional, Any, Callable

from .base_camera import BaseCamera
from .camera_detector import CameraDetector

# Set up logging
logger = logging.getLogger("ThreadedCameraManager")

class ThreadedCameraManager:
    """
    Singleton class for managing cameras in separate threads.
    """

    _instance = None

    @classmethod
    def get_instance(cls):
        """
        Get the singleton instance of the camera manager.

        Returns:
            ThreadedCameraManager: Singleton instance
        """
        if cls._instance is None:
            cls._instance = ThreadedCameraManager()
        return cls._instance

    def __init__(self):
        """
        Initialize the camera manager.
        """
        if ThreadedCameraManager._instance is not None:
            raise RuntimeError("ThreadedCameraManager is a singleton class. Use get_instance() instead.")

        self.camera_detector = CameraDetector()
        self.active_cameras = {}  # position -> camera
        self.camera_threads = {}  # position -> thread
        self.camera_stop_events = {}  # position -> stop event
        self.camera_frames = {}  # position -> frame
        self.camera_frame_locks = {}  # position -> lock
        self.camera_change_callbacks = []  # Callbacks for camera changes

        # Perform one-time camera detection
        self.camera_detector.detect_cameras_once()

        logger.info("Initialized threaded camera manager")

    def _on_camera_change(self, added_cameras, removed_cameras):
        """
        Callback for camera changes.

        Args:
            added_cameras: Dict of added cameras
            removed_cameras: Dict of removed cameras
        """
        # Check if any active camera was removed
        for position, camera in list(self.active_cameras.items()):
            camera_id = camera.camera_id
            camera_type = 'webcam' if hasattr(camera, 'cap') else 'realsense'
            full_id = f"{camera_type}_{camera_id}"

            if full_id in removed_cameras:
                logger.info(f"Active camera {camera.name} was disconnected")
                self.disconnect_camera(position)

        # Notify callbacks about camera changes
        if added_cameras or removed_cameras:
            for callback in self.camera_change_callbacks:
                try:
                    callback()
                except Exception as e:
                    logger.error(f"Error in camera change callback: {e}")

    def register_camera_change_callback(self, callback: Callable[[], None]) -> None:
        """
        Register a callback function to be called when cameras are added or removed.

        Args:
            callback: Callback function with no arguments
        """
        if callback not in self.camera_change_callbacks:
            self.camera_change_callbacks.append(callback)

    def unregister_camera_change_callback(self, callback: Callable[[], None]) -> None:
        """
        Unregister a callback function.

        Args:
            callback: Callback function to unregister
        """
        if callback in self.camera_change_callbacks:
            self.camera_change_callbacks.remove(callback)

    def get_available_cameras(self) -> List[Dict]:
        """
        Get a list of available cameras.

        Returns:
            List[Dict]: List of available cameras
        """
        return self.camera_detector.get_available_cameras()

    def connect_camera(self, camera_id: str, position: str) -> bool:
        """
        Connect to a camera and start a thread for it.

        Args:
            camera_id: Camera ID
            position: Camera position (front, back, left, right)

        Returns:
            bool: True if connection was successful, False otherwise
        """
        # Check if position is already in use
        if position in self.active_cameras:
            logger.error(f"Position {position} is already in use")
            return False

        logger.info(f"Connecting camera {camera_id} to position {position}")

        # Create camera object
        camera = self.camera_detector.create_camera(camera_id)

        if camera is None:
            logger.error(f"Failed to create camera for ID {camera_id}")
            return False

        # Set position
        camera.position = position

        # Connect to camera
        logger.info(f"Attempting to connect to camera {camera.name} (ID: {camera_id})")
        if not camera.connect():
            logger.error(f"Failed to connect to camera {camera.name} (ID: {camera_id})")
            return False

        # Verify camera is working by getting a test frame
        test_frame = camera.get_frame()
        if test_frame is None:
            logger.error(f"Camera {camera.name} (ID: {camera_id}) connected but failed to provide a frame")
            camera.disconnect()
            return False

        logger.info(f"Successfully connected to camera {camera.name} (ID: {camera_id})")

        # Create stop event and frame lock
        stop_event = threading.Event()
        frame_lock = threading.Lock()

        # Start camera thread
        thread = threading.Thread(
            target=self._camera_thread,
            args=(camera, position, stop_event, frame_lock),
            daemon=True
        )
        thread.start()

        # Store camera, thread, stop event, and frame lock
        self.active_cameras[position] = camera
        self.camera_threads[position] = thread
        self.camera_stop_events[position] = stop_event
        self.camera_frame_locks[position] = frame_lock
        self.camera_frames[position] = test_frame  # Store initial frame

        logger.info(f"Connected to camera {camera.name} (ID: {camera_id}) at position {position}")

        return True

    def disconnect_camera(self, position: str) -> None:

        if position not in self.active_cameras:
            logger.info(f"No camera at position {position}")
            return

        # Get camera, thread, and stop event
        camera = self.active_cameras[position]
        thread = self.camera_threads[position]
        stop_event = self.camera_stop_events[position]

        # Set stop event
        stop_event.set()

        # Wait for thread to stop
        thread.join(timeout=5.0)

        # Disconnect from camera
        camera.disconnect()

        # Remove camera, thread, stop event, and frame
        del self.active_cameras[position]
        del self.camera_threads[position]
        del self.camera_stop_events[position]

        with self.camera_frame_locks[position]:
            if position in self.camera_frames:
                del self.camera_frames[position]

        del self.camera_frame_locks[position]

        logger.info(f"Disconnected from camera {camera.name} at position {position}")

    def disconnect_all_cameras(self) -> None:
        """
        Disconnect from all cameras and stop their threads.
        """
        for position in list(self.active_cameras.keys()):
            self.disconnect_camera(position)

        logger.info("Disconnected from all cameras")

    def is_camera_active(self, position: str) -> bool:
        """
        Check if a camera is active at the given position.

        Args:
            position: Camera position (front, back, left, right)

        Returns:
            bool: True if a camera is active at the given position, False otherwise
        """
        return position in self.active_cameras

    def get_frame(self, position: str) -> Optional[Any]:
        """
        Get a frame from the camera at the given position.

        Args:
            position: Camera position (front, back, left, right)

        Returns:
            np.ndarray: Frame from the camera or None if no frame is available
        """
        if position not in self.active_cameras:
            return None

        if position not in self.camera_frames:
            return None

        with self.camera_frame_locks[position]:
            if self.camera_frames[position] is None:
                return None

            return self.camera_frames[position].copy()

    def get_active_cameras(self) -> Dict[str, str]:
        """
        Get a dictionary of active cameras.

        Returns:
            Dict[str, str]: Dictionary mapping positions to camera IDs
        """
        return {position: camera.camera_id for position, camera in self.active_cameras.items()}

    def get_active_camera_info(self, position: str) -> Optional[Dict]:
        """
        Get information about the active camera at the given position.

        Args:
            position: Camera position (front, back, left, right)

        Returns:
            Dict: Dictionary containing camera information or None if no camera is active
        """
        if position not in self.active_cameras:
            return None

        return self.active_cameras[position].get_camera_info()

    def _camera_thread(self, camera: BaseCamera, position: str, stop_event: threading.Event, frame_lock: threading.Lock) -> None:
        """
        Thread function for a camera.

        Args:
            camera: Camera object
            position: Camera position (front, back, left, right)
            stop_event: Event to signal thread to stop
            frame_lock: Lock for frame access
        """
        logger.info(f"Started camera thread for {camera.name} at position {position}")

        # Variables for monitoring camera health
        consecutive_failures = 0
        last_successful_frame_time = time.time()
        reconnect_attempts = 0
        max_reconnect_attempts = 3

        while not stop_event.is_set():
            try:
                # Get frame from camera
                frame = camera.get_frame()

                if frame is not None:
                    # Reset failure counters on success
                    consecutive_failures = 0
                    last_successful_frame_time = time.time()
                    reconnect_attempts = 0

                    # Store frame
                    with frame_lock:
                        self.camera_frames[position] = frame
                else:
                    # Increment failure counter
                    consecutive_failures += 1

                    # If we've had too many consecutive failures or it's been too long since the last frame
                    if consecutive_failures > 30 or (time.time() - last_successful_frame_time > 5.0):
                        logger.warning(f"Camera {camera.name} at position {position} appears to be disconnected (failures: {consecutive_failures})")

                        # Try to reconnect if we haven't exceeded max attempts
                        if reconnect_attempts < max_reconnect_attempts:
                            logger.info(f"Attempting to reconnect to camera {camera.name} (attempt {reconnect_attempts + 1}/{max_reconnect_attempts})")

                            # Disconnect and reconnect
                            camera.disconnect()
                            time.sleep(1.0)  # Wait a bit before reconnecting

                            if camera.connect():
                                logger.info(f"Successfully reconnected to camera {camera.name}")
                                consecutive_failures = 0
                                last_successful_frame_time = time.time()
                                reconnect_attempts += 1
                            else:
                                logger.error(f"Failed to reconnect to camera {camera.name}")
                                reconnect_attempts += 1
                                time.sleep(2.0)  # Wait longer before next attempt
                        else:
                            logger.error(f"Maximum reconnection attempts reached for camera {camera.name}. Giving up.")
                            # We could choose to exit the thread here, but instead we'll keep trying periodically
                            time.sleep(5.0)  # Wait longer between attempts after max is reached
                            reconnect_attempts = 0  # Reset counter to try again later

                # Sleep to limit CPU usage (adaptive based on frame rate)
                target_frame_time = 1.0 / max(camera.actual_fps if hasattr(camera, 'actual_fps') else 30, 1)
                time.sleep(min(target_frame_time / 2, 0.01))  # Sleep at most 10ms or half the frame time

            except Exception as e:
                logger.error(f"Error in camera thread for {camera.name} at position {position}: {e}")
                consecutive_failures += 1
                time.sleep(0.5)  # Avoid busy loop in case of errors

        # Clean up when thread is stopping
        try:
            camera.disconnect()
        except Exception as e:
            logger.error(f"Error disconnecting camera {camera.name} in thread cleanup: {e}")

        logger.info(f"Stopped camera thread for {camera.name} at position {position}")
