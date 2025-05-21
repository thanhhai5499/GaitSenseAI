import numpy as np
from typing import Optional, Dict, Any

class BaseCamera:
    def __init__(self, camera_id: Any, name: str, width: int = 1920, height: int = 1080, fps: int = 30, position: str = None):
        self.camera_id = camera_id
        self.name = name
        self.width = width
        self.height = height
        self.fps = fps
        self.position = position
        self.is_connected = False
        self.actual_fps = fps
        self.actual_width = width
        self.actual_height = height

    def connect(self) -> bool:
        raise NotImplementedError("connect() method must be implemented by subclasses")

    def disconnect(self) -> bool:
        raise NotImplementedError("disconnect() method must be implemented by subclasses")

    def get_frame(self) -> Optional[np.ndarray]:
        raise NotImplementedError("get_frame() method must be implemented by subclasses")

    def get_camera_info(self) -> Dict[str, Any]:
        return {
            'id': self.camera_id,
            'name': self.name,
            'width': self.actual_width,
            'height': self.actual_height,
            'fps': self.fps,
            'actual_fps': self.actual_fps,
            'position': self.position,
            'is_connected': self.is_connected
        }

    def __str__(self) -> str:
        return f"{self.name} (ID: {self.camera_id}, {self.actual_width}x{self.actual_height} @ {self.actual_fps}fps)"
