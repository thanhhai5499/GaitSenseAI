import cv2
import numpy as np
import logging
import time
from typing import Optional, Dict, List, Any, Tuple

from .base_camera import BaseCamera

# Set up logging
logger = logging.getLogger("LogitechCamera")

class LogitechCamera(BaseCamera):
    def __init__(self, camera_id: int, name: str = "Logitech Brio 300FHD",
                 width: int = 1920, height: int = 1080, fps: int = 30,
                 position: str = ""):
        """
        Initialize a Logitech camera.

        Args:
            camera_id: Camera device ID
            name: Camera name
            width: Frame width in pixels (default: 1920 for Full HD)
            height: Frame height in pixels (default: 1080 for Full HD)
            fps: Frames per second (default: 30)
            position: Camera position (front, back, left, right)
        """
        if position:
            name = f"{name} ({position})"

        super().__init__(camera_id, name, width, height, fps)
        self.position = position
        self.connection_attempts = 0
        self.max_connection_attempts = 3

    def connect(self) -> bool:
        """
        Connect to the camera and configure it with the specified settings.

        Returns:
            bool: True if connection was successful, False otherwise
        """
        self.connection_attempts += 1

        try:
            logger.info(f"Connecting to Logitech camera {self.camera_id} ({self.name})...")

            # Use DirectShow backend on Windows for better performance with Logitech cameras
            self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)

            if not self.cap.isOpened():
                logger.error(f"Failed to open camera {self.camera_id}")
                return self._handle_connection_failure("Failed to open camera")

            # Configure camera settings
            success = self._configure_camera_settings()
            if not success:
                return self._handle_connection_failure("Failed to configure camera settings")

            # Verify camera is working by reading a test frame
            ret, _ = self.cap.read()
            if not ret:
                logger.error(f"Failed to read frame from camera {self.camera_id}")
                return self._handle_connection_failure("Failed to read test frame")

            # Camera is successfully connected and configured
            self.is_running = True
            self.connection_attempts = 0  # Reset counter on successful connection

            # Log actual camera settings
            self._log_camera_settings()

            return True

        except Exception as e:
            logger.error(f"Exception connecting to camera {self.camera_id}: {str(e)}")
            return self._handle_connection_failure(f"Exception: {str(e)}")

    def _configure_camera_settings(self) -> bool:
        """
        Configure camera settings for optimal image quality.

        Returns:
            bool: True if configuration was successful, False otherwise
        """
        try:
            # Set resolution to maximum (Full HD for Brio 300)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

            # Set frame rate (FPS)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)

            # Increase buffer size to prevent frame drops
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)

            # Log the actual camera settings after configuration
            actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)

            logger.info(f"Camera {self.camera_id} configured with: "
                       f"Resolution: {actual_width}x{actual_height}, "
                       f"FPS: {actual_fps}")

            return True

        except Exception as e:
            logger.error(f"Error configuring camera {self.camera_id}: {str(e)}")
            return False

    def _log_camera_settings(self) -> None:
        """Log the actual camera settings after configuration."""
        actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)

        logger.info(f"Camera {self.camera_id} configured with: "
                   f"Resolution: {actual_width}x{actual_height}, "
                   f"FPS: {actual_fps}")

    def _handle_connection_failure(self, reason: str) -> bool:
        """
        Handle connection failure with retry logic.

        Args:
            reason: Reason for the connection failure

        Returns:
            bool: Always False (connection failed)
        """
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        # If we haven't exceeded max attempts, we'll retry on next connect() call
        if self.connection_attempts < self.max_connection_attempts:
            logger.warning(f"Connection attempt {self.connection_attempts}/{self.max_connection_attempts} "
                          f"failed for camera {self.camera_id}: {reason}. Will retry.")
        else:
            logger.error(f"Failed to connect to camera {self.camera_id} after "
                        f"{self.max_connection_attempts} attempts: {reason}")

        return False

    def disconnect(self) -> None:
        """Disconnect from the camera and release resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.is_running = False
        logger.info(f"Disconnected from camera {self.camera_id} ({self.name})")

    def get_frame(self) -> Optional[np.ndarray]:
        """
        Get a frame from the camera.

        Returns:
            np.ndarray or None: The captured frame or None if unsuccessful
        """
        if self.cap is None or not self.cap.isOpened():
            return None

        try:
            ret, frame = self.cap.read()
            if not ret:
                # If frame capture fails, try to reconnect
                logger.warning(f"Failed to read frame from camera {self.camera_id}, attempting to reconnect...")
                self.disconnect()
                if self.connect():
                    # Try reading again after reconnection
                    ret, frame = self.cap.read()
                    if not ret:
                        return None
                    return frame
                return None
            return frame
        except Exception as e:
            logger.error(f"Exception reading frame from camera {self.camera_id}: {str(e)}")
            return None

    @staticmethod
    def list_available_cameras() -> List[Dict]:
        """
        List all available Logitech cameras.

        Returns:
            List[Dict]: List of dictionaries containing camera information
        """
        available_cameras = []

        for i in range(10):  # Check first 10 camera indices
            try:
                # Use DirectShow backend on Windows for better performance
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        # Try to set Full HD resolution to identify Logitech Brio 300
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

                        # Get actual resolution (some cameras may not support Full HD)
                        actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                        actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

                        # Get camera name if available
                        camera_name = f"Camera {i}"
                        try:
                            # Some cameras provide their name through this property
                            backend_name = cap.getBackendName()
                            if "Logitech" in backend_name:
                                camera_name = backend_name
                        except:
                            pass

                        # Check if this is likely a Logitech Brio 300 (supports Full HD)
                        if actual_width == 1920 and actual_height == 1080:
                            camera_info = {
                                'id': i,
                                'name': f"Logitech Brio 300FHD ({i})"
                            }
                        else:
                            camera_info = {
                                'id': i,
                                'name': f"{camera_name} ({i}) - {int(actual_width)}x{int(actual_height)}"
                            }

                        available_cameras.append(camera_info)
                    cap.release()
            except Exception as e:
                logger.warning(f"Error checking camera {i}: {str(e)}")

        return available_cameras

    def get_camera_info(self) -> Dict:
        """
        Get information about this camera.

        Returns:
            Dict: Dictionary containing camera information
        """
        info = {
            'id': self.camera_id,
            'name': self.name,
            'resolution': f"{self.width}x{self.height}",
            'fps': self.fps,
            'position': self.position,
            'is_running': self.is_running
        }

        # Add actual settings if camera is connected
        if self.is_running and self.cap is not None:
            try:
                info['actual_width'] = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                info['actual_height'] = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                info['actual_fps'] = round(self.cap.get(cv2.CAP_PROP_FPS), 1)
            except:
                pass

        return info
