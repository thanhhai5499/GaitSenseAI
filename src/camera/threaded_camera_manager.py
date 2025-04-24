"""
Threaded Camera Manager Module

This module provides a camera manager that automatically starts all cameras
upon initialization and manages them using multithreading to prevent stream congestion.
"""

import cv2
import threading
import time
import numpy as np
import queue
import logging
import sys
from typing import Dict, List, Tuple, Optional, Any

from .base_camera import BaseCamera
from .realsense_camera import RealSenseCamera, REALSENSE_AVAILABLE
from .logitech_camera import LogitechCamera
from .placeholder_camera import PlaceholderCamera
from .camera_config import CAMERA_CONFIG

# Set up logging
logger = logging.getLogger("ThreadedCameraManager")

# Maximum queue size for frame buffers
MAX_QUEUE_SIZE = 10


class CameraThread(threading.Thread):
    """
    Thread class for handling camera streaming.
    Each camera runs in its own thread to prevent blocking the main thread.
    """

    def __init__(self, camera: BaseCamera, position: str, frame_queue: queue.Queue):
        """
        Initialize the camera thread.

        Args:
            camera: The camera object to stream from
            position: Position identifier for the camera (front, back, left, right)
            frame_queue: Queue to store captured frames
        """
        threading.Thread.__init__(self)
        self.daemon = True  # Thread will close when main program exits

        self.camera = camera
        self.position = position
        self.frame_queue = frame_queue
        self.running = False

        # Frame processing attributes
        self.last_frame = None
        self.frame_count = 0
        self.fps = 0
        self.last_fps_update = time.time()

        logger.info(f"Initialized {position} camera thread with camera: {camera.name}")

    def run(self):
        """Main thread function that captures frames from the camera."""
        self.running = True

        # Connect to the camera
        if not self.camera.connect():
            logger.error(f"Failed to connect to {self.position} camera")
            self.running = False
            return

        logger.info(f"Starting capture loop for {self.position} camera")

        # Main capture loop
        while self.running:
            try:
                # Capture frame
                frame = self.camera.get_frame()

                if frame is None:
                    logger.warning(f"Failed to capture frame from {self.position} camera")
                    # Add a small delay to prevent CPU overload in case of continuous failures
                    time.sleep(0.1)
                    continue

                # Update FPS calculation
                self._update_fps()

                # Add text overlay with camera info
                self._add_overlay(frame)

                # Store the frame in the queue, removing old frames if necessary
                if self.frame_queue.qsize() >= MAX_QUEUE_SIZE:
                    try:
                        self.frame_queue.get_nowait()  # Remove oldest frame
                    except queue.Empty:
                        pass  # Queue was emptied by another thread

                self.frame_queue.put((self.position, frame))
                self.last_frame = frame

            except Exception as e:
                logger.error(f"Error in {self.position} camera thread: {str(e)}")
                time.sleep(0.5)  # Prevent rapid error logging

        # Clean up resources when thread stops
        self.camera.disconnect()
        logger.info(f"{self.position} camera thread stopped")

    def _update_fps(self):
        """Update the FPS calculation."""
        self.frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - self.last_fps_update

        # Update FPS every second
        if elapsed_time > 1.0:
            self.fps = self.frame_count / elapsed_time
            self.frame_count = 0
            self.last_fps_update = current_time

    def _add_overlay(self, frame: np.ndarray):
        """
        Add text overlay with camera information to the frame.

        Args:
            frame: The frame to add overlay to
        """
        # Add camera name
        cv2.putText(
            frame,
            f"{self.position.upper()} CAMERA",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # Add FPS
        cv2.putText(
            frame,
            f"FPS: {self.fps:.1f}",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        # Add resolution
        cv2.putText(
            frame,
            f"Resolution: {frame.shape[1]}x{frame.shape[0]}",
            (10, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    def stop(self):
        """Stop the camera thread."""
        self.running = False
        logger.info(f"Stopping {self.position} camera thread")


class ThreadedCameraManager:
    """
    Manager class for handling multiple camera threads.
    Automatically starts all cameras upon initialization.
    Implements the same interface as the original CameraManager for compatibility.
    """

    _instance = None

    @classmethod
    def get_instance(cls):
        """Get the singleton instance of the camera manager."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        """Initialize the threaded camera manager and start all cameras."""
        self.cameras: Dict[Any, BaseCamera] = {}
        self.camera_threads: Dict[str, CameraThread] = {}
        self.frame_queues: Dict[str, queue.Queue] = {}
        self.latest_frames: Dict[str, Optional[np.ndarray]] = {}

        self.active_cameras: Dict[str, Any] = {
            "front": None,
            "back": None,
            "left": None,
            "right": None
        }

        # Initialize frame queues for each camera position
        for position in self.active_cameras:
            self.frame_queues[position] = queue.Queue(maxsize=MAX_QUEUE_SIZE)
            self.latest_frames[position] = None

        logger.info("Threaded Camera Manager initialized")

        # Automatically start all cameras
        self._initialize_cameras()
        self._start_cameras()

    def _initialize_cameras(self) -> None:
        """Initialize cameras based on configuration."""
        # Create cameras based on configuration
        for position, config in CAMERA_CONFIG.items():
            camera_type = config["type"]
            camera_id = config["id"]

            if camera_type == "realsense" and REALSENSE_AVAILABLE:
                camera = RealSenseCamera(
                    camera_id=camera_id,
                    name=config["name"],
                    width=config["width"],
                    height=config["height"],
                    fps=config["fps"]
                )
                self.cameras[f"rs_{camera_id}"] = camera
                self.active_cameras[position] = f"rs_{camera_id}"
            elif camera_type == "webcam" and camera_id is not None:
                # Create Logitech camera with maximum settings
                camera = LogitechCamera(
                    camera_id=camera_id,
                    name=config["name"],
                    width=config["width"],
                    height=config["height"],
                    fps=config["fps"],
                    position=position
                )
                self.cameras[camera_id] = camera
                self.active_cameras[position] = camera_id
            elif camera_type == "placeholder" or camera_id is None:
                camera = PlaceholderCamera(
                    camera_id=-1,
                    name=config["name"],
                    width=config["width"],
                    height=config["height"],
                    fps=config["fps"],
                    position=position
                )
                self.cameras[f"placeholder_{position}"] = camera
                self.active_cameras[position] = f"placeholder_{position}"

        logger.info(f"Initialized {len(self.cameras)} cameras")

    def _start_cameras(self) -> None:
        """Start all cameras in separate threads."""
        # Start cameras with a delay between each to prevent USB bandwidth issues
        for position, camera_id in self.active_cameras.items():
            if camera_id is not None and camera_id in self.cameras:
                camera = self.cameras[camera_id]

                # Create and start camera thread
                camera_thread = CameraThread(
                    camera=camera,
                    position=position,
                    frame_queue=self.frame_queues[position]
                )

                camera_thread.start()
                self.camera_threads[position] = camera_thread

                # Small delay between camera starts to prevent USB bandwidth issues
                time.sleep(0.5)

                logger.info(f"Started camera at position {position}")

    def initialize_cameras(self) -> None:
        """
        Initialize and detect available cameras.
        This method is included for compatibility with the original CameraManager.
        """
        # This is already done in __init__, but included for compatibility
        pass

    def get_available_cameras(self) -> List[Dict]:
        """
        Get a list of available cameras.

        Returns:
            List of dictionaries containing camera information
        """
        camera_list = []

        # Check for RealSense cameras
        if REALSENSE_AVAILABLE:
            realsense_cameras = RealSenseCamera.list_available_cameras()
            for camera_info in realsense_cameras:
                camera_id = f"rs_{camera_info['id']}"
                camera_list.append({
                    'id': camera_id,
                    'name': camera_info['name']
                })

        # Check for regular webcams
        for i in range(10):
            try:
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                        actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                        actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

                        if actual_width == 1920 and actual_height == 1080:
                            camera_info = {
                                'id': i,
                                'name': f"Logitech Brio 300FHD ({i})"
                            }
                        else:
                            camera_info = {
                                'id': i,
                                'name': f"Camera {i}"
                            }

                        camera_list.append(camera_info)
                    cap.release()
            except Exception as e:
                logger.error(f"Error checking camera {i}: {str(e)}")

        return camera_list

    def get_active_cameras(self) -> Dict[str, Any]:
        """
        Get the currently active cameras.

        Returns:
            Dictionary mapping camera positions to camera IDs
        """
        return self.active_cameras.copy()

    def connect_camera(self, camera_id: Any, position: str) -> bool:
        """
        Connect a camera to a specific position.

        Args:
            camera_id: ID of the camera to connect
            position: Position to connect the camera to (front, back, left, right)

        Returns:
            bool: True if connection was successful, False otherwise
        """
        if position not in self.active_cameras:
            logger.error(f"Invalid position: {position}")
            return False

        # Check if camera is already in use at another position
        for pos, cam_id in self.active_cameras.items():
            if cam_id is not None and cam_id == camera_id and pos != position:
                logger.error(f"Camera {camera_id} is already in use at position {pos}")
                return False

        # Disconnect any existing camera at this position
        if position in self.camera_threads and self.camera_threads[position] is not None:
            self.disconnect_camera(position)

        # Handle placeholder camera
        if camera_id is None:
            camera = PlaceholderCamera(
                camera_id=-1,
                width=CAMERA_CONFIG[position]["width"],
                height=CAMERA_CONFIG[position]["height"],
                fps=CAMERA_CONFIG[position]["fps"],
                position=position
            )
            camera_id = f"placeholder_{position}"
        # Handle RealSense camera
        elif isinstance(camera_id, str) and camera_id.startswith("rs_"):
            rs_id = int(camera_id[3:])
            camera = RealSenseCamera(
                camera_id=rs_id,
                width=1280,
                height=720,
                fps=30,
                name=f"RealSense D435i ({position})"
            )
        # Handle regular webcam (Logitech Brio 300)
        else:
            # Get configuration for this position
            config = CAMERA_CONFIG.get(position, {})
            camera = LogitechCamera(
                camera_id=int(camera_id),
                width=config.get("width", 1920),
                height=config.get("height", 1080),
                fps=config.get("fps", 30),
                position=position
            )

        # Create and start camera thread
        self.cameras[camera_id] = camera
        self.active_cameras[position] = camera_id

        camera_thread = CameraThread(
            camera=camera,
            position=position,
            frame_queue=self.frame_queues[position]
        )

        camera_thread.start()
        self.camera_threads[position] = camera_thread

        logger.info(f"Connected camera {camera_id} to position {position}")
        return True

    def disconnect_camera(self, position: str) -> None:
        """
        Disconnect the camera at the specified position.

        Args:
            position: Position of the camera to disconnect
        """
        if position in self.camera_threads and self.camera_threads[position] is not None:
            # Stop the camera thread
            self.camera_threads[position].stop()
            # Wait for thread to finish
            self.camera_threads[position].join(timeout=1.0)
            # Remove thread reference
            self.camera_threads[position] = None

        # Clear the active camera for this position
        self.active_cameras[position] = None
        # Clear the latest frame
        self.latest_frames[position] = None

        logger.info(f"Disconnected camera at position {position}")

    def disconnect_all_cameras(self) -> None:
        """Disconnect all active cameras."""
        for position in list(self.camera_threads.keys()):
            if self.camera_threads[position] is not None:
                self.disconnect_camera(position)

        logger.info("All cameras disconnected")

    def get_frame(self, position: str) -> Optional[np.ndarray]:
        """
        Get the latest frame from the camera at the specified position.

        Args:
            position: Position of the camera to get frame from

        Returns:
            np.ndarray or None: The latest frame or None if no frame is available
        """
        # Check if there are new frames in the queue
        try:
            while not self.frame_queues[position].empty():
                pos, frame = self.frame_queues[position].get_nowait()
                if pos == position:
                    self.latest_frames[position] = frame
        except queue.Empty:
            pass  # No new frames
        except Exception as e:
            logger.error(f"Error getting frame from {position} queue: {str(e)}")

        return self.latest_frames[position]

    def is_camera_active(self, position: str) -> bool:
        """
        Check if a camera is active at the specified position.

        Args:
            position: Position to check

        Returns:
            bool: True if a camera is active at the position, False otherwise
        """
        return (position in self.camera_threads and
                self.camera_threads[position] is not None and
                self.camera_threads[position].running)

    def get_active_camera_info(self, position: str) -> Optional[Dict]:
        """
        Get information about the active camera at the specified position.

        Args:
            position: Position to get camera info for

        Returns:
            Dict or None: Camera information or None if no camera is active
        """
        if not self.is_camera_active(position):
            return None

        camera_id = self.active_cameras[position]
        if camera_id is None or camera_id not in self.cameras:
            return None

        camera = self.cameras[camera_id]
        return camera.get_camera_info()
