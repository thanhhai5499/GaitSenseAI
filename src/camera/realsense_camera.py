import cv2
import numpy as np
from typing import Optional, Dict, List, Tuple

try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False

from .base_camera import BaseCamera


class RealSenseCamera(BaseCamera):
    def __init__(self, camera_id: int, name: str = "RealSense D435i", 
                 width: int = 1280, height: int = 720, fps: int = 30):
        super().__init__(camera_id, name, width, height, fps)
        
        if not REALSENSE_AVAILABLE:
            return
            
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.align = None
        self.depth_scale = None
    
    def connect(self) -> bool:
        if not REALSENSE_AVAILABLE:
            return False
            
        try:
            self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
            self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
            
            profile = self.pipeline.start(self.config)
            
            depth_sensor = profile.get_device().first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()
            
            self.align = rs.align(rs.stream.color)
            
            self.is_running = True
            return True
        except Exception as e:
            return False
    
    def disconnect(self) -> None:
        if self.is_running and REALSENSE_AVAILABLE:
            self.pipeline.stop()
            self.is_running = False
    
    def get_frame(self) -> Optional[np.ndarray]:
        if not self.is_running or not REALSENSE_AVAILABLE:
            return None
            
        try:
            frames = self.pipeline.wait_for_frames()
            
            aligned_frames = self.align.process(frames)
            
            color_frame = aligned_frames.get_color_frame()
            
            color_image = np.asanyarray(color_frame.get_data())
            
            return color_image
        except Exception as e:
            return None
    
    def get_depth_frame(self) -> Optional[np.ndarray]:
        if not self.is_running or not REALSENSE_AVAILABLE:
            return None
            
        try:
            frames = self.pipeline.wait_for_frames()
            
            aligned_frames = self.align.process(frames)
            
            depth_frame = aligned_frames.get_depth_frame()
            
            depth_image = np.asanyarray(depth_frame.get_data())
            
            return depth_image
        except Exception as e:
            return None
    
    def is_connected(self) -> bool:
        return REALSENSE_AVAILABLE and self.is_running
    
    @staticmethod
    def list_available_cameras() -> List[Dict]:
        if not REALSENSE_AVAILABLE:
            return []
            
        available_cameras = []
        
        try:
            ctx = rs.context()
            devices = ctx.query_devices()
            
            for i, device in enumerate(devices):
                camera_info = {
                    'id': i,
                    'name': f"RealSense {device.get_info(rs.camera_info.name)}",
                    'serial': device.get_info(rs.camera_info.serial_number)
                }
                available_cameras.append(camera_info)
        except Exception as e:
            pass
        
        return available_cameras
