"""
Webcam Camera Module

This module provides a class for standard webcams, including Logitech Brio.
"""

import cv2
import numpy as np
import time
import logging
from typing import Optional, Dict, List, Tuple, Any
import threading

from .base_camera import BaseCamera

# Set up logging
logger = logging.getLogger("WebcamCamera")

class WebcamCamera(BaseCamera):
    """
    Class for standard webcams.
    """

    def __init__(self, camera_id: int, name: str = None, width: int = 1920, height: int = 1080, fps: int = 30, position: str = None):
        """
        Initialize the webcam camera.

        Args:
            camera_id: Camera identifier (OpenCV index)
            name: Human-readable name for the camera (default: auto-generated)
            width: Frame width in pixels (default: 1920)
            height: Frame height in pixels (default: 1080)
            fps: Frames per second (default: 30)
            position: Camera position (front, back, left, right)
        """
        # Generate name if not provided
        if name is None:
            name = f"Webcam {camera_id}"

        super().__init__(camera_id, name, width, height, fps, position)

        self.cap = None
        self.frame_lock = threading.Lock()
        self.latest_frame = None
        self.frame_time = 0

        # Check if camera supports the requested resolution and FPS
        supported_modes = self._check_camera_capabilities(camera_id)
        if supported_modes:
            best_mode = self._find_best_mode(supported_modes, width, height, fps)
            if best_mode:
                logger.info(f"Camera supports {best_mode['width']}x{best_mode['height']} @ {best_mode['fps']}fps")
                if best_mode['width'] != width or best_mode['height'] != height or best_mode['fps'] != fps:
                    logger.warning(f"Requested {width}x{height} @ {fps}fps may not be fully supported")
                    logger.warning(f"Best supported mode: {best_mode['width']}x{best_mode['height']} @ {best_mode['fps']}fps")

        logger.info(f"Initialized webcam camera {name} (ID: {camera_id})")

    def _check_camera_capabilities(self, camera_id: int) -> List[Dict]:
        """
        Check what resolutions and FPS the camera supports.

        Args:
            camera_id: Camera identifier (OpenCV index)

        Returns:
            List[Dict]: List of supported modes (width, height, fps)
        """
        supported_modes = []

        try:
            # Try to open the camera temporarily
            temp_cap = cv2.VideoCapture(camera_id)
            if not temp_cap.isOpened():
                temp_cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
                if not temp_cap.isOpened():
                    logger.warning(f"Could not open camera {camera_id} to check capabilities")
                    return supported_modes

            # Common resolutions to check
            resolutions = [
                (640, 480),    # VGA
                (1280, 720),   # HD
                (1920, 1080),  # Full HD
            ]

            # Common FPS values to check
            fps_values = [15, 30, 60]

            # Check each combination
            for width, height in resolutions:
                for fps in fps_values:
                    # Try to set resolution and FPS
                    temp_cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                    temp_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                    temp_cap.set(cv2.CAP_PROP_FPS, fps)

                    # Read a frame to apply settings
                    ret, _ = temp_cap.read()
                    if not ret:
                        continue

                    # Get actual values
                    actual_width = int(temp_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    actual_height = int(temp_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    actual_fps = int(temp_cap.get(cv2.CAP_PROP_FPS))

                    # Check if the values are close to what we requested
                    if (abs(actual_width - width) < 10 and
                        abs(actual_height - height) < 10 and
                        actual_fps > 0):

                        mode = {
                            'width': actual_width,
                            'height': actual_height,
                            'fps': actual_fps
                        }

                        if mode not in supported_modes:
                            supported_modes.append(mode)
                            logger.debug(f"Camera {camera_id} supports: {actual_width}x{actual_height} @ {actual_fps}fps")

            # Release the camera
            temp_cap.release()

        except Exception as e:
            logger.debug(f"Error checking camera capabilities: {e}")

        return supported_modes

    def _find_best_mode(self, modes: List[Dict], target_width: int, target_height: int, target_fps: int) -> Optional[Dict]:
        """
        Find the best supported mode closest to the target resolution and FPS.

        Args:
            modes: List of supported modes
            target_width: Target width
            target_height: Target height
            target_fps: Target FPS

        Returns:
            Dict: Best mode or None if no suitable mode found
        """
        if not modes:
            return None

        # First, try to find exact match
        for mode in modes:
            if (mode['width'] == target_width and
                mode['height'] == target_height and
                mode['fps'] == target_fps):
                return mode

        # If no exact match, find the closest match
        best_mode = None
        best_score = float('inf')

        for mode in modes:
            # Calculate score based on difference from target
            # Prioritize resolution over FPS
            width_diff = abs(mode['width'] - target_width)
            height_diff = abs(mode['height'] - target_height)
            fps_diff = abs(mode['fps'] - target_fps)

            # Weight resolution differences more heavily
            score = (width_diff / target_width) * 10 + (height_diff / target_height) * 10 + (fps_diff / target_fps)

            if score < best_score:
                best_score = score
                best_mode = mode

        return best_mode

    def connect(self) -> bool:
        """
        Connect to the webcam.

        Returns:
            bool: True if connection was successful, False otherwise
        """
        if self.is_connected and self.cap is not None and self.cap.isOpened():
            logger.info(f"Webcam {self.name} is already connected")
            return True

        # Make sure we're starting fresh
        self.disconnect()

        try:
            logger.info(f"Attempting to connect to webcam {self.name} (ID: {self.camera_id})")

            # Try with DirectShow first (better for high-resolution and high-fps on Windows)
            self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)

            if not self.cap.isOpened():
                logger.warning(f"Failed to open webcam with DirectShow, trying default backend")
                # Try with default backend
                self.cap = cv2.VideoCapture(self.camera_id)

                if not self.cap.isOpened():
                    logger.error(f"Failed to open webcam {self.name} (ID: {self.camera_id}) with any backend")
                    return False

            # Optimize buffer size (reduce latency)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # Set resolution first
            logger.info(f"Setting resolution to {self.width}x{self.height}")
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

            # Set FPS (may not be supported by all cameras)
            logger.info(f"Setting FPS to {self.fps}")
            fps_success = self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            logger.info(f"FPS setting {'successful' if fps_success else 'not supported by camera'}")

            # Try to set additional properties to improve performance
            # Set codec to MJPG which often supports higher FPS
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

            # Read multiple test frames to make sure the camera is working and stable
            # This helps with some cameras that need a few frames to "warm up"
            success = False
            for i in range(5):  # Try up to 5 frames
                ret, test_frame = self.cap.read()
                if ret and test_frame is not None:
                    success = True
                    break
                time.sleep(0.1)  # Short delay between attempts

            if not success:
                logger.error(f"Failed to read test frames from webcam {self.name} (ID: {self.camera_id})")
                self.cap.release()
                self.cap = None
                return False

            # Get actual resolution and FPS
            self.actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))

            # If reported FPS is too low, try to estimate real FPS
            if self.actual_fps < 10:
                logger.info(f"Reported FPS is low ({self.actual_fps}), estimating real FPS...")
                # Estimate real FPS by measuring time to capture 10 frames
                start_time = time.time()
                frame_count = 0
                for _ in range(15):  # Try to get 15 frames, but may get fewer
                    ret, _ = self.cap.read()
                    if ret:
                        frame_count += 1
                    else:
                        logger.warning(f"Failed to read frame during FPS estimation")
                end_time = time.time()

                if frame_count > 0:
                    estimated_fps = frame_count / (end_time - start_time)
                    logger.info(f"Estimated FPS: {estimated_fps:.2f} (got {frame_count} frames in {end_time - start_time:.2f}s)")

                    # Use estimated FPS if it's higher than reported FPS
                    if estimated_fps > self.actual_fps:
                        self.actual_fps = int(estimated_fps)
                        logger.info(f"Using estimated FPS: {self.actual_fps}")
                else:
                    logger.warning(f"Could not estimate FPS - no frames captured during test")

            logger.info(f"Successfully connected to webcam {self.name} (ID: {self.camera_id})")
            logger.info(f"Actual resolution: {self.actual_width}x{self.actual_height} @ {self.actual_fps}fps")

            # If actual FPS is still low, warn the user
            if self.actual_fps < 20:
                logger.warning(f"Camera reports low FPS ({self.actual_fps}). This may be a limitation of the camera or USB bandwidth.")
                logger.warning(f"For better performance, try a lower resolution or a camera that supports higher FPS at 1080p.")

            # Store the first frame
            ret, first_frame = self.cap.read()
            if ret and first_frame is not None:
                with self.frame_lock:
                    self.latest_frame = first_frame
                    self.frame_time = time.time()

            self.is_connected = True
            return True
        except Exception as e:
            logger.error(f"Error connecting to webcam {self.name} (ID: {self.camera_id}): {e}")
            if self.cap is not None:
                try:
                    self.cap.release()
                except:
                    pass
                self.cap = None
            return False

    def disconnect(self) -> None:
        """
        Disconnect from the webcam.
        """
        if not self.is_connected:
            return

        try:
            if self.cap is not None:
                self.cap.release()
                self.cap = None

            self.is_connected = False
            logger.info(f"Disconnected from webcam {self.name} (ID: {self.camera_id})")
        except Exception as e:
            logger.error(f"Error disconnecting from webcam {self.name} (ID: {self.camera_id}): {e}")

    def get_frame(self) -> Optional[np.ndarray]:
        """
        Get a frame from the webcam.

        Returns:
            np.ndarray: Frame from the webcam or None if no frame is available
        """
        if not self.is_connected or self.cap is None:
            return None

        try:
            # Check if the camera is still open
            if not self.cap.isOpened():
                logger.warning(f"Camera {self.name} (ID: {self.camera_id}) is no longer open")
                return None

            # Check if we need to get a new frame (limit to FPS)
            current_time = time.time()
            target_frame_time = 1.0 / max(self.actual_fps, 1)  # Use actual FPS, not requested FPS

            # If we have a recent frame and it's not time for a new one yet, return the cached frame
            if (current_time - self.frame_time < target_frame_time) and self.latest_frame is not None:
                return self.latest_frame.copy()

            # Get a new frame
            ret, frame = self.cap.read()

            if not ret or frame is None:
                # Try one more time before giving up
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    logger.warning(f"Failed to get frame from webcam {self.name} (ID: {self.camera_id})")

                    # Check if the camera is still open
                    if not self.cap.isOpened():
                        logger.error(f"Camera {self.name} (ID: {self.camera_id}) is no longer open after failed read")
                        self.is_connected = False
                        return None

                    # Return the last good frame if available
                    return self.latest_frame.copy() if self.latest_frame is not None else None

            # Update frame time and latest frame
            with self.frame_lock:
                self.frame_time = current_time
                self.latest_frame = frame

            return frame.copy()
        except Exception as e:
            logger.error(f"Error getting frame from webcam {self.name} (ID: {self.camera_id}): {e}")

            # Check if this is a critical error that indicates the camera is disconnected
            if "VIDEOIO ERROR" in str(e) or "device disconnected" in str(e).lower():
                logger.error(f"Critical camera error detected, marking camera as disconnected: {e}")
                self.is_connected = False
                return None

            # Return the last good frame if available
            return self.latest_frame.copy() if self.latest_frame is not None else None

    @staticmethod
    def list_available_cameras() -> List[Dict]:
        """
        List available webcams.

        Returns:
            List[Dict]: List of available webcams
        """
        available_cameras = []

        # Check for webcams (try up to 20 indices to be thorough)
        for i in range(20):
            try:
                # First try with DirectShow on Windows (better for some cameras)
                try:
                    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                    dshow_opened = cap.isOpened()
                except:
                    dshow_opened = False
                    cap = None

                # If DirectShow failed, try with default backend
                if not dshow_opened:
                    if cap is not None:
                        cap.release()

                    cap = cv2.VideoCapture(i)
                    if not cap.isOpened():
                        # No camera at this index with either backend
                        if cap is not None:
                            cap.release()
                        continue

                # Get camera name (not always available)
                name = f"Webcam {i}"

                # Try to get camera name from backend (not always supported)
                try:
                    backend_name = cap.getBackendName()
                    if backend_name:
                        name = f"{backend_name} Camera {i}"
                except:
                    pass

                # Try to read a test frame to confirm camera is working
                ret, frame = cap.read()
                if not ret or frame is None:
                    logger.debug(f"Camera {i} opened but failed to read frame")
                    cap.release()
                    continue

                # Try to set resolution to 1920x1080 to see if it's supported
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                cap.set(cv2.CAP_PROP_FPS, 30)

                # Read another frame to apply settings
                ret, frame = cap.read()

                # Get camera resolution
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))

                # If FPS is reported as 0, use a default value
                if fps <= 0:
                    fps = 30

                # Add camera to list
                logger.info(f"Found camera: {name} (ID: {i}) - Resolution: {width}x{height} @ {fps}fps")
                available_cameras.append({
                    'id': i,
                    'name': name,
                    'width': width,
                    'height': height,
                    'fps': fps,
                    'type': 'webcam'
                })

                # Release the camera
                cap.release()

                # Add a small delay to avoid USB issues when checking multiple cameras
                import time
                time.sleep(0.1)

            except Exception as e:
                logger.debug(f"Error checking webcam {i}: {e}")
                if 'cap' in locals() and cap is not None:
                    try:
                        cap.release()
                    except:
                        pass

        # Log summary
        if available_cameras:
            logger.info(f"Found {len(available_cameras)} cameras: {[c['name'] for c in available_cameras]}")
        else:
            logger.warning("No cameras found!")

        return available_cameras
