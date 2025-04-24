"""
Placeholder Camera Module

This module provides a placeholder camera class for positions where no physical camera is connected.
"""

import cv2
import numpy as np
from typing import Optional, Dict, List

from .base_camera import BaseCamera


class PlaceholderCamera(BaseCamera):
    """
    Placeholder camera class that generates blank frames with text.
    Used for positions where no physical camera is connected.
    """
    
    def __init__(self, camera_id: int = -1, name: str = "Placeholder Camera", 
                 width: int = 1280, height: int = 720, fps: int = 30, position: str = ""):
        if position:
            name = f"{name} ({position})"
        
        super().__init__(camera_id, name, width, height, fps)
        self.position = position
    
    def connect(self) -> bool:
        """
        Connect to the placeholder camera (always succeeds).
        
        Returns:
            bool: Always True
        """
        self.is_running = True
        return True
    
    def disconnect(self) -> None:
        """Disconnect from the placeholder camera."""
        self.is_running = False
    
    def get_frame(self) -> Optional[np.ndarray]:
        """
        Get a frame from the placeholder camera.
        
        Returns:
            np.ndarray: A blank frame with text
        """
        if not self.is_running:
            return None
        
        # Create a blank frame
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Add text to indicate this is a placeholder
        position_text = self.position.upper() if self.position else "UNKNOWN"
        cv2.putText(
            frame,
            f"{position_text} CAMERA NOT CONNECTED",
            (int(self.width/2) - 200, int(self.height/2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )
        
        return frame
    
    @staticmethod
    def list_available_cameras() -> List[Dict]:
        """
        List available placeholder cameras (always empty).
        
        Returns:
            List[Dict]: Empty list
        """
        return []
