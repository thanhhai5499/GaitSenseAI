"""
RealSense Camera Module

This module provides a class for Intel RealSense cameras.
"""

import cv2
import numpy as np
import time
import logging
from typing import Optional, Dict, List, Tuple, Any
import threading

try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False
    logging.warning("PyRealSense2 not available. RealSense cameras will not be supported.")

from .base_camera import BaseCamera

# Set up logging
logger = logging.getLogger("RealSenseCamera")

class RealSenseCamera(BaseCamera):
    """
    Class for Intel RealSense cameras.
    """
    
    def __init__(self, camera_id: str, name: str = None, width: int = 1920, height: int = 1080, fps: int = 30, position: str = None):
        """
        Initialize the RealSense camera.
        
        Args:
            camera_id: Camera identifier (serial number)
            name: Human-readable name for the camera (default: auto-generated)
            width: Frame width in pixels (default: 1920)
            height: Frame height in pixels (default: 1080)
            fps: Frames per second (default: 30)
            position: Camera position (front, back, left, right)
        """
        # Generate name if not provided
        if name is None:
            name = f"RealSense {camera_id}"
        
        super().__init__(camera_id, name, width, height, fps, position)
        
        self.pipeline = None
        self.config = None
        self.profile = None
        self.frame_lock = threading.Lock()
        self.latest_frame = None
        self.frame_time = 0
        self.is_depth_camera = True
        
        logger.info(f"Initialized RealSense camera {name} (ID: {camera_id})")
    
    def connect(self) -> bool:
        """
        Connect to the RealSense camera.
        
        Returns:
            bool: True if connection was successful, False otherwise
        """
        if not REALSENSE_AVAILABLE:
            logger.error("PyRealSense2 not available. Cannot connect to RealSense camera.")
            return False
        
        if self.is_connected:
            logger.info(f"RealSense camera {self.name} is already connected")
            return True
        
        try:
            # Create pipeline and config
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            
            # Enable the device
            self.config.enable_device(self.camera_id)
            
            # Enable color stream
            self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
            
            # Start the pipeline
            self.profile = self.pipeline.start(self.config)
            
            # Get actual resolution and FPS
            color_stream = self.profile.get_stream(rs.stream.color)
            color_intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
            self.actual_width = color_intrinsics.width
            self.actual_height = color_intrinsics.height
            self.actual_fps = color_stream.fps()
            
            logger.info(f"Connected to RealSense camera {self.name} (ID: {self.camera_id})")
            logger.info(f"Actual resolution: {self.actual_width}x{self.actual_height} @ {self.actual_fps}fps")
            
            self.is_connected = True
            return True
        except Exception as e:
            logger.error(f"Error connecting to RealSense camera {self.name} (ID: {self.camera_id}): {e}")
            return False
    
    def disconnect(self) -> None:
        """
        Disconnect from the RealSense camera.
        """
        if not self.is_connected:
            return
        
        try:
            if self.pipeline is not None:
                self.pipeline.stop()
                self.pipeline = None
                self.config = None
                self.profile = None
            
            self.is_connected = False
            logger.info(f"Disconnected from RealSense camera {self.name} (ID: {self.camera_id})")
        except Exception as e:
            logger.error(f"Error disconnecting from RealSense camera {self.name} (ID: {self.camera_id}): {e}")
    
    def get_frame(self) -> Optional[np.ndarray]:
        """
        Get a frame from the RealSense camera.
        
        Returns:
            np.ndarray: Frame from the RealSense camera or None if no frame is available
        """
        if not REALSENSE_AVAILABLE:
            return None
        
        if not self.is_connected or self.pipeline is None:
            return None
        
        try:
            # Check if we need to get a new frame (limit to FPS)
            current_time = time.time()
            if current_time - self.frame_time < 1.0 / self.fps and self.latest_frame is not None:
                return self.latest_frame.copy()
            
            # Wait for a coherent pair of frames
            frames = self.pipeline.wait_for_frames()
            
            # Get color frame
            color_frame = frames.get_color_frame()
            
            if not color_frame:
                logger.warning(f"Failed to get color frame from RealSense camera {self.name} (ID: {self.camera_id})")
                return None
            
            # Convert to numpy array
            frame = np.asanyarray(color_frame.get_data())
            
            # Update frame time and latest frame
            with self.frame_lock:
                self.frame_time = current_time
                self.latest_frame = frame
            
            return frame.copy()
        except Exception as e:
            logger.error(f"Error getting frame from RealSense camera {self.name} (ID: {self.camera_id}): {e}")
            return None
    
    @staticmethod
    def list_available_cameras() -> List[Dict]:
        """
        List available RealSense cameras.
        
        Returns:
            List[Dict]: List of available RealSense cameras
        """
        available_cameras = []
        
        if not REALSENSE_AVAILABLE:
            return available_cameras
        
        try:
            # Create context
            ctx = rs.context()
            
            # Get devices
            devices = ctx.query_devices()
            
            for device in devices:
                try:
                    # Get device info
                    serial = device.get_info(rs.camera_info.serial_number)
                    name = device.get_info(rs.camera_info.name)
                    
                    # Add camera to list
                    available_cameras.append({
                        'id': serial,
                        'name': f"RealSense {name}",
                        'width': 1920,  # Default values
                        'height': 1080,
                        'fps': 30,
                        'type': 'realsense'
                    })
                except Exception as e:
                    logger.debug(f"Error getting RealSense device info: {e}")
        except Exception as e:
            logger.debug(f"Error listing RealSense cameras: {e}")
        
        return available_cameras
