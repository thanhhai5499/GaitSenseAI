import cv2
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Optional, List, Dict


class BaseCamera(ABC):
    def __init__(self, camera_id: int, name: str, width: int = 1920, height: int = 1080, fps: int = 30):
        self.camera_id = camera_id
        self.name = name
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
        self.is_running = False
    
    @abstractmethod
    def connect(self) -> bool:
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        pass
    
    @abstractmethod
    def get_frame(self) -> Optional[np.ndarray]:
        pass
    
    @staticmethod
    def list_available_cameras() -> List[Dict]:
        available_cameras = []
        
        for i in range(10):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    camera_info = {
                        'id': i,
                        'name': f"Camera {i}"
                    }
                    available_cameras.append(camera_info)
                cap.release()
        
        return available_cameras
    
    def is_connected(self) -> bool:
        return self.cap is not None and self.cap.isOpened()
    
    def get_camera_info(self) -> Dict:
        return {
            'id': self.camera_id,
            'name': self.name,
            'width': self.width,
            'height': self.height,
            'fps': self.fps,
            'is_connected': self.is_connected(),
            'is_running': self.is_running
        }
