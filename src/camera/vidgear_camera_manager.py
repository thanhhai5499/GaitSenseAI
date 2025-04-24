"""
VidGear Camera Manager Module

This module provides a camera manager that automatically starts all cameras
upon initialization and manages them using VidGear's multithreading to prevent stream congestion.
"""

import cv2
import threading
import time
import numpy as np
import queue
import logging
import sys
from typing import Dict, List, Tuple, Optional, Any

try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False

from vidgear.gears import CamGear
from .base_camera import BaseCamera
from .camera_config import CAMERA_CONFIG

# Set up logging with null handler to suppress logs
logging.getLogger("VidGearCameraManager").addHandler(logging.NullHandler())

# Maximum queue size for frame buffers
MAX_QUEUE_SIZE = 10

class VidGearCameraThread(threading.Thread):
    """
    Thread class for handling individual camera streams using VidGear.
    """

    def __init__(self, camera_id: int, position: str, frame_queue: queue.Queue,
                 width: int, height: int, fps: int, name: str = None):
        """
        Initialize the camera thread.

        Args:
            camera_id: The camera device ID
            position: Position identifier for the camera (front, back, left, right)
            frame_queue: Queue to store captured frames
            width: Frame width in pixels
            height: Frame height in pixels
            fps: Frames per second
            name: Camera name
        """
        threading.Thread.__init__(self)
        self.daemon = True  # Thread will close when main program exits

        self.camera_id = camera_id
        self.position = position
        self.frame_queue = frame_queue
        self.running = False
        self.width = width
        self.height = height
        self.fps = fps
        self.name = name if name else f"Camera {camera_id}"

        # Frame processing attributes
        self.last_frame = None
        self.frame_count = 0
        self.fps_counter = 0
        self.last_fps_update = time.time()

        # VidGear camera object
        self.camera = None

    def run(self):
        """Main thread function that captures frames from the camera."""
        self.running = True

        # Configure camera options
        options = {
            "CAP_PROP_FRAME_WIDTH": self.width,
            "CAP_PROP_FRAME_HEIGHT": self.height,
            "CAP_PROP_FPS": self.fps,
            "CAP_PROP_BUFFERSIZE": 3  # Increased buffer size to prevent frame drops
        }

        try:
            # Initialize VidGear CamGear with the specified options
            self.camera = CamGear(source=self.camera_id, logging=False, **options)
            self.camera.start()

            # Main capture loop
            while self.running:
                try:
                    # Capture frame
                    frame = self.camera.read()

                    if frame is None:
                        # Add a small delay to prevent CPU overload in case of continuous failures
                        time.sleep(0.1)
                        continue
                except Exception:
                    # Handle any exceptions during frame reading
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

            # Clean up resources when thread stops
            if self.camera:
                self.camera.stop()

        except Exception as e:
            self.running = False
            if self.camera:
                self.camera.stop()

    def _update_fps(self):
        """Update the FPS calculation."""
        self.frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - self.last_fps_update

        # Update FPS every second
        if elapsed_time > 1.0:
            self.fps_counter = self.frame_count / elapsed_time
            self.frame_count = 0
            self.last_fps_update = current_time

    def _add_overlay(self, frame):
        """Add text overlay with camera information."""
        if frame is not None:
            text = f"{self.position.upper()} - {self.name} ({self.width}x{self.height}@{self.fps_counter:.1f}fps)"
            cv2.putText(
                frame,
                text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    def stop(self):
        """Stop the camera thread."""
        self.running = False
        try:
            if self.camera:
                self.camera.stop()
        except Exception:
            pass


class RealSenseCameraThread(threading.Thread):
    """
    Thread class for handling RealSense camera streams.
    """

    def __init__(self, camera_id: int, position: str, frame_queue: queue.Queue,
                 width: int, height: int, fps: int, name: str = None):
        """
        Initialize the RealSense camera thread.

        Args:
            camera_id: The camera device ID
            position: Position identifier for the camera (front, back, left, right)
            frame_queue: Queue to store captured frames
            width: Frame width in pixels
            height: Frame height in pixels
            fps: Frames per second
            name: Camera name
        """
        threading.Thread.__init__(self)
        self.daemon = True  # Thread will close when main program exits

        self.camera_id = camera_id
        self.position = position
        self.frame_queue = frame_queue
        self.running = False
        self.width = width
        self.height = height
        self.fps = fps
        self.name = name if name else "RealSense D435i"

        # Frame processing attributes
        self.last_frame = None
        self.frame_count = 0
        self.fps_counter = 0
        self.last_fps_update = time.time()

        # RealSense specific attributes
        self.pipeline = None
        self.config = None
        self.align = None

    def run(self):
        """Main thread function that captures frames from the RealSense camera."""
        if not REALSENSE_AVAILABLE:
            return

        self.running = True
        self.pipeline_started = False

        try:
            # Initialize RealSense pipeline
            self.pipeline = rs.pipeline()
            self.config = rs.config()

            # Configure streams
            self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
            self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)

            # Start streaming
            profile = self.pipeline.start(self.config)
            self.pipeline_started = True

            # Create alignment object
            self.align = rs.align(rs.stream.color)

            # Main capture loop
            while self.running:
                try:
                    # Wait for a coherent pair of frames
                    frames = self.pipeline.wait_for_frames(timeout_ms=1000)

                    # Align the depth frame to color frame
                    aligned_frames = self.align.process(frames)

                    # Get aligned frames
                    color_frame = aligned_frames.get_color_frame()

                    # Convert images to numpy arrays
                    frame = np.asanyarray(color_frame.get_data())

                    if frame is None or frame.size == 0:
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
                except Exception:
                    # Handle any exceptions during frame reading
                    time.sleep(0.1)
                    continue

            # Clean up resources when thread stops
            if self.pipeline and self.pipeline_started:
                try:
                    self.pipeline.stop()
                except Exception:
                    pass

        except Exception:
            self.running = False
            # Create a blank frame to indicate error
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            cv2.putText(
                frame,
                f"REALSENSE CAMERA ERROR",
                (int(self.width/2) - 200, int(self.height/2)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

            # Store the error frame in the queue
            if self.frame_queue.qsize() >= MAX_QUEUE_SIZE:
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass

            self.frame_queue.put((self.position, frame))
            self.last_frame = frame

    def _update_fps(self):
        """Update the FPS calculation."""
        self.frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - self.last_fps_update

        # Update FPS every second
        if elapsed_time > 1.0:
            self.fps_counter = self.frame_count / elapsed_time
            self.frame_count = 0
            self.last_fps_update = current_time

    def _add_overlay(self, frame):
        """Add text overlay with camera information."""
        if frame is not None:
            text = f"{self.position.upper()} - {self.name} ({self.width}x{self.height}@{self.fps_counter:.1f}fps)"
            cv2.putText(
                frame,
                text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    def stop(self):
        """Stop the camera thread."""
        self.running = False
        if self.pipeline and hasattr(self, 'pipeline_started') and self.pipeline_started:
            try:
                self.pipeline.stop()
            except Exception:
                pass


class PlaceholderCameraThread(threading.Thread):
    """
    Thread class for handling placeholder camera streams.
    """

    def __init__(self, position: str, frame_queue: queue.Queue,
                 width: int, height: int, fps: int, name: str = None):
        """
        Initialize the placeholder camera thread.

        Args:
            position: Position identifier for the camera (front, back, left, right)
            frame_queue: Queue to store captured frames
            width: Frame width in pixels
            height: Frame height in pixels
            fps: Frames per second
            name: Camera name
        """
        threading.Thread.__init__(self)
        self.daemon = True  # Thread will close when main program exits

        self.position = position
        self.frame_queue = frame_queue
        self.running = False
        self.width = width
        self.height = height
        self.fps = fps
        self.name = name if name else "Placeholder Camera"

        # Frame processing attributes
        self.last_frame = None
        self.frame_count = 0
        self.fps_counter = 0
        self.last_fps_update = time.time()

    def run(self):
        """Main thread function that generates placeholder frames."""
        self.running = True

        # Calculate delay based on FPS
        delay = 1.0 / self.fps

        # Main loop
        while self.running:
            # Create a blank frame
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

            # Add text to indicate this is a placeholder
            cv2.putText(
                frame,
                f"{self.position.upper()} CAMERA NOT CONNECTED",
                (int(self.width/2) - 200, int(self.height/2)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )

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

            # Sleep to maintain the desired frame rate
            time.sleep(delay)

    def _update_fps(self):
        """Update the FPS calculation."""
        self.frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - self.last_fps_update

        # Update FPS every second
        if elapsed_time > 1.0:
            self.fps_counter = self.frame_count / elapsed_time
            self.frame_count = 0
            self.last_fps_update = current_time

    def _add_overlay(self, frame):
        """Add text overlay with camera information."""
        if frame is not None:
            text = f"{self.position.upper()} - {self.name} ({self.width}x{self.height}@{self.fps_counter:.1f}fps)"
            cv2.putText(
                frame,
                text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    def stop(self):
        """Stop the camera thread."""
        self.running = False


class VidGearCameraManager:
    """
    Singleton class for managing multiple cameras using VidGear.
    """

    _instance = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls):
        """Get the singleton instance of the camera manager."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def __init__(self):
        """Initialize the VidGear camera manager and start all cameras."""
        self.camera_threads: Dict[str, threading.Thread] = {}
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

        # Automatically start all cameras
        self._initialize_cameras()

    def _initialize_cameras(self) -> None:
        """Initialize cameras based on configuration."""
        # Start cameras with a delay between each to prevent USB bandwidth issues
        for position, config in CAMERA_CONFIG.items():
            camera_type = config["type"]
            camera_id = config["id"]
            width = config["width"]
            height = config["height"]
            fps = config["fps"]
            name = config["name"]

            try:
                if camera_type == "realsense" and REALSENSE_AVAILABLE and camera_id is not None:
                    # Create and start RealSense camera thread
                    camera_thread = RealSenseCameraThread(
                        camera_id=camera_id,
                        position=position,
                        frame_queue=self.frame_queues[position],
                        width=width,
                        height=height,
                        fps=fps,
                        name=name
                    )
                    camera_thread.start()
                    self.camera_threads[position] = camera_thread
                    self.active_cameras[position] = camera_id

                elif camera_type == "webcam" and camera_id is not None:
                    # Create and start VidGear camera thread
                    camera_thread = VidGearCameraThread(
                        camera_id=camera_id,
                        position=position,
                        frame_queue=self.frame_queues[position],
                        width=width,
                        height=height,
                        fps=fps,
                        name=name
                    )
                    camera_thread.start()
                    self.camera_threads[position] = camera_thread
                    self.active_cameras[position] = camera_id

                else:
                    # Create and start placeholder camera thread
                    self._create_placeholder_camera(position, width, height, fps, name)
            except Exception:
                # If camera initialization fails, create a placeholder camera
                self._create_placeholder_camera(position, width, height, fps, f"Placeholder ({position})")

            # Small delay between camera starts to prevent USB bandwidth issues
            time.sleep(0.5)

    def _create_placeholder_camera(self, position, width, height, fps, name):
        """Create a placeholder camera for the specified position."""
        camera_thread = PlaceholderCameraThread(
            position=position,
            frame_queue=self.frame_queues[position],
            width=width,
            height=height,
            fps=fps,
            name=name
        )
        camera_thread.start()
        self.camera_threads[position] = camera_thread
        self.active_cameras[position] = None

    def get_frame(self, position: str) -> Optional[np.ndarray]:
        """
        Get the latest frame from the specified camera position.

        Args:
            position: Camera position (front, back, left, right)

        Returns:
            np.ndarray or None: The latest frame or None if no frame is available
        """
        # Check if the position is valid
        if position not in self.frame_queues:
            return None

        # Get all available frames from the queue
        frames = []
        while not self.frame_queues[position].empty():
            try:
                frame_data = self.frame_queues[position].get_nowait()
                frames.append(frame_data)
            except queue.Empty:
                break

        # Return the latest frame if available
        if frames:
            # The frame data is a tuple of (position, frame)
            self.latest_frames[position] = frames[-1][1]
            return self.latest_frames[position]
        else:
            # Return the last known frame if no new frames are available
            return self.latest_frames[position]

    def is_camera_active(self, position: str) -> bool:
        """
        Check if a camera at the specified position is active.

        Args:
            position: Camera position (front, back, left, right)

        Returns:
            bool: True if the camera is active, False otherwise
        """
        return position in self.camera_threads and self.camera_threads[position] is not None and self.camera_threads[position].running

    def connect_camera(self, camera_id: Any, position: str) -> bool:
        """
        Connect a camera to the specified position.

        Args:
            camera_id: Camera identifier
            position: Camera position (front, back, left, right)

        Returns:
            bool: True if the connection was successful, False otherwise
        """
        # Disconnect any existing camera at this position
        if position in self.camera_threads and self.camera_threads[position] is not None:
            self.disconnect_camera(position)

        # Get configuration for this position
        config = CAMERA_CONFIG.get(position, {})
        width = config.get("width", 1920)
        height = config.get("height", 1080)
        fps = config.get("fps", 30)
        name = config.get("name", f"Camera {camera_id}")

        try:
            if camera_id is None:
                # Create and start placeholder camera thread
                camera_thread = PlaceholderCameraThread(
                    position=position,
                    frame_queue=self.frame_queues[position],
                    width=width,
                    height=height,
                    fps=fps,
                    name="Placeholder Camera"
                )
            elif isinstance(camera_id, str) and camera_id.startswith("rs_"):
                # Create and start RealSense camera thread
                rs_id = int(camera_id[3:])
                camera_thread = RealSenseCameraThread(
                    camera_id=rs_id,
                    position=position,
                    frame_queue=self.frame_queues[position],
                    width=width,
                    height=height,
                    fps=fps,
                    name=f"RealSense D435i ({position})"
                )
            else:
                # Create and start VidGear camera thread
                camera_thread = VidGearCameraThread(
                    camera_id=int(camera_id),
                    position=position,
                    frame_queue=self.frame_queues[position],
                    width=width,
                    height=height,
                    fps=fps,
                    name=name
                )

            camera_thread.start()
            self.camera_threads[position] = camera_thread
            self.active_cameras[position] = camera_id

            return True
        except Exception as e:
            return False

    def disconnect_camera(self, position: str) -> None:
        """
        Disconnect the camera at the specified position.

        Args:
            position: Camera position (front, back, left, right)
        """
        if position in self.camera_threads and self.camera_threads[position] is not None:
            try:
                # Stop the camera thread
                self.camera_threads[position].stop()
            except Exception:
                pass

            self.camera_threads[position] = None
            self.active_cameras[position] = None
            self.latest_frames[position] = None

            # Clear the frame queue
            if position in self.frame_queues:
                while not self.frame_queues[position].empty():
                    try:
                        self.frame_queues[position].get_nowait()
                    except queue.Empty:
                        break

    def disconnect_all_cameras(self) -> None:
        """Disconnect all cameras."""
        for position in self.camera_threads:
            self.disconnect_camera(position)

    def get_active_cameras(self) -> Dict[str, Any]:
        """
        Get a dictionary of active cameras.

        Returns:
            Dict[str, Any]: Dictionary mapping positions to camera IDs
        """
        return self.active_cameras.copy()

    def get_active_camera_info(self, position: str) -> Optional[Dict]:
        """
        Get information about the active camera at the specified position.

        Args:
            position: Camera position (front, back, left, right)

        Returns:
            Dict or None: Dictionary containing camera information or None if no camera is active
        """
        if position not in self.camera_threads or self.camera_threads[position] is None:
            return None

        thread = self.camera_threads[position]

        # Get basic information
        info = {
            'id': getattr(thread, 'camera_id', None),
            'name': getattr(thread, 'name', 'Unknown Camera'),
            'width': getattr(thread, 'width', 0),
            'height': getattr(thread, 'height', 0),
            'fps': getattr(thread, 'fps', 0),
            'actual_fps': getattr(thread, 'fps_counter', 0),
            'is_running': getattr(thread, 'running', False),
            'position': position
        }

        return info

    def get_available_cameras(self) -> List[Dict]:
        """
        Get a list of available cameras.

        Returns:
            List[Dict]: List of dictionaries containing camera information
        """
        camera_list = []

        # Check for RealSense cameras
        if REALSENSE_AVAILABLE:
            try:
                ctx = rs.context()
                devices = ctx.query_devices()

                for i, device in enumerate(devices):
                    try:
                        name = device.get_info(rs.camera_info.name)
                        serial = device.get_info(rs.camera_info.serial_number)
                        camera_info = {
                            'id': f"rs_{i}",
                            'name': f"RealSense {name} ({serial})",
                            'serial': serial,
                            'type': 'realsense'
                        }
                        camera_list.append(camera_info)
                    except Exception:
                        pass
            except Exception:
                pass

        # Check for regular webcams - use the results from our camera detection
        opencv_cameras = self._detect_opencv_cameras()
        for camera in opencv_cameras:
            camera_id = camera['id']
            camera_list.append(camera)

        return camera_list

    def _detect_opencv_cameras(self) -> List[Dict]:
        """
        Detect available OpenCV cameras.

        Returns:
            List[Dict]: List of dictionaries containing camera information
        """
        available_cameras = []

        # Manually add cameras based on the check_cameras.py results
        # This ensures we only try to access cameras that actually exist
        camera_configs = [
            {"id": 0, "name": "Camera 0 (640x480)", "width": 640, "height": 480, "fps": 30},
            {"id": 2, "name": "Camera 2 (640x480)", "width": 640, "height": 480, "fps": 30},
            {"id": 3, "name": "Camera 3 (640x480)", "width": 640, "height": 480, "fps": 30}
        ]

        for config in camera_configs:
            camera_info = {
                'id': config["id"],
                'name': config["name"],
                'width': config["width"],
                'height': config["height"],
                'fps': config["fps"],
                'type': "webcam"
            }
            available_cameras.append(camera_info)

        return available_cameras
