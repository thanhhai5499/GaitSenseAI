"""
Base Camera Module

This module provides the base class for all camera types in the system.
"""

import cv2
import numpy as np
import time
import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, List, Tuple, Any

# Set up logging
logger = logging.getLogger("BaseCamera")

class BaseCamera(ABC):
    """
    Abstract base class for all camera types.
    """
    
    def __init__(self, camera_id: Any, name: str, width: int = 1920, height: int = 1080, fps: int = 30, position: str = None):
        """
        Initialize the base camera.
        
        Args:
            camera_id: Camera identifier
            name: Human-readable name for the camera
            width: Frame width in pixels (default: 1920)
            height: Frame height in pixels (default: 1080)
            fps: Frames per second (default: 30)
            position: Camera position (front, back, left, right)
        """
        self.camera_id = camera_id
        self.name = name
        self.width = width
        self.height = height
        self.fps = fps
        self.position = position
        
        self.is_connected = False
        self.actual_width = None
        self.actual_height = None
        self.actual_fps = None
        
        logger.info(f"Initialized camera {name} (ID: {camera_id})")
    
    @abstractmethod
    def connect(self) -> bool:
        """
        Connect to the camera.
        
        Returns:
            bool: True if connection was successful, False otherwise
        """
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """
        Disconnect from the camera.
        """
        pass
    
    @abstractmethod
    def get_frame(self) -> Optional[np.ndarray]:
        """
        Get a frame from the camera.
        
        Returns:
            np.ndarray: Frame from the camera or None if no frame is available
        """
        pass
    
    def get_camera_info(self) -> Dict:
        """
        Get information about this camera.
        
        Returns:
            Dict: Dictionary containing camera information
        """
        info = {
            'id': self.camera_id,
            'name': self.name,
            'width': self.width,
            'height': self.height,
            'fps': self.fps,
            'resolution': f"{self.width}x{self.height}",
            'is_connected': self.is_connected,
            'position': self.position
        }
        
        # Add actual values if available
        if self.actual_width is not None:
            info['actual_width'] = self.actual_width
        
        if self.actual_height is not None:
            info['actual_height'] = self.actual_height
        
        if self.actual_fps is not None:
            info['actual_fps'] = self.actual_fps
        
        return info
    
    @staticmethod
    @abstractmethod
    def list_available_cameras() -> List[Dict]:
        """
        List available cameras of this type.
        
        Returns:
            List[Dict]: List of available cameras
        """
        pass
