from typing import Dict, List, Optional, Type, Any
import cv2
import numpy as np
import logging

from .base_camera import BaseCamera
from .realsense_camera import RealSenseCamera, REALSENSE_AVAILABLE
from .logitech_camera import LogitechCamera
from .placeholder_camera import PlaceholderCamera
from .camera_config import CAMERA_CONFIG

# Set up logging
logger = logging.getLogger("CameraManager")


class CameraManager:
    _instance = None
    _lock = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.cameras: Dict[int, BaseCamera] = {}
        self.active_cameras: Dict[str, int] = {
            "front": None,
            "back": None,
            "left": None,
            "right": None
        }

    def initialize_cameras(self) -> None:
        if REALSENSE_AVAILABLE:
            realsense_cameras = RealSenseCamera.list_available_cameras()
            for camera_info in realsense_cameras:
                camera_id = camera_info['id']
                camera = RealSenseCamera(
                    camera_id=camera_id,
                    name=camera_info['name'],
                    width=1280,
                    height=720,
                    fps=30
                )
                self.cameras[camera_id] = camera

        logitech_cameras = LogitechCamera.list_available_cameras()
        for camera_info in logitech_cameras:
            camera_id = camera_info['id']
            camera = LogitechCamera(
                camera_id=camera_id,
                name=camera_info['name'],
                width=1920,
                height=1080,
                fps=30
            )
            self.cameras[camera_id] = camera

    def get_available_cameras(self) -> List[Dict]:
        camera_list = []

        for i in range(10):
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

        if REALSENSE_AVAILABLE:
            realsense_cameras = RealSenseCamera.list_available_cameras()
            for camera_info in realsense_cameras:
                camera_info['id'] = f"rs_{camera_info['id']}"
                camera_list.append(camera_info)

        return camera_list

    def get_active_cameras(self) -> Dict[str, Optional[int]]:
        return self.active_cameras

    def connect_camera(self, camera_id: Any, position: str) -> bool:
        if position not in self.active_cameras:
            logger.error(f"Invalid position: {position}")
            return False

        for pos, cam_id in self.active_cameras.items():
            if cam_id is not None and cam_id == camera_id and pos != position:
                logger.error(f"Camera {camera_id} is already in use at position {pos}")
                return False

        if self.active_cameras[position] is not None:
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
        # Handle regular webcam
        else:
            camera = LogitechCamera(
                camera_id=int(camera_id),
                width=1920,
                height=1080,
                fps=30,
                position=position
            )

        if camera.connect():
            self.cameras[camera_id] = camera
            self.active_cameras[position] = camera_id
            return True
        else:
            return False

    def disconnect_camera(self, position: str) -> None:
        if position in self.active_cameras and self.active_cameras[position] is not None:
            camera_id = self.active_cameras[position]
            if camera_id in self.cameras:
                self.cameras[camera_id].disconnect()
                logger.info(f"Disconnected camera at position {position}")
            self.active_cameras[position] = None

    def disconnect_all_cameras(self) -> None:
        for camera in self.cameras.values():
            camera.disconnect()

        self.active_cameras = {
            "front": None,
            "back": None,
            "left": None,
            "right": None
        }

        logger.info("All cameras disconnected")

    def get_frame(self, position: str) -> Optional[np.ndarray]:
        if position in self.active_cameras and self.active_cameras[position] is not None:
            camera_id = self.active_cameras[position]
            if camera_id in self.cameras:
                try:
                    frame = self.cameras[camera_id].get_frame()
                    if frame is None:
                        logger.warning(f"Failed to get frame from {position} camera")
                    return frame
                except Exception as e:
                    logger.error(f"Error getting frame from {position} camera: {str(e)}")

        return None

    def is_camera_active(self, position: str) -> bool:
        if position in self.active_cameras and self.active_cameras[position] is not None:
            camera_id = self.active_cameras[position]
            if camera_id in self.cameras:
                return self.cameras[camera_id].is_connected()

        return False

    def get_active_camera_info(self, position: str) -> Optional[Dict]:
        if position in self.active_cameras and self.active_cameras[position] is not None:
            camera_id = self.active_cameras[position]
            if camera_id in self.cameras:
                return self.cameras[camera_id].get_camera_info()

        return None
