import cv2
import numpy as np
import time
import threading
import os
import platform
from typing import Optional, Dict, List, Any

from .base_camera import BaseCamera

os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

class WebcamCamera(BaseCamera):
    def __init__(self, camera_id: int, name: str = None, width: int = 1920, height: int = 1080, fps: int = 30, position: str = None, instance_id: str = None):
        if name is None:
            name = f"CAM{camera_id}"

        super().__init__(camera_id, name, width, height, fps, position)

        self.cap = None
        self.frame_lock = threading.Lock()
        self.latest_frame = None
        self.frame_time = 0
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 3
        self.instance_id = instance_id

    def _try_camera_settings(self) -> bool:
        if self.cap is None or not self.cap.isOpened():
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        try:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
        except:
            pass

        self.actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))

        if self.actual_fps < 10:
            self.actual_fps = 30

        return True

    def connect(self) -> bool:
        if self.is_connected and self.cap is not None and self.cap.isOpened():
            return True

        self.disconnect()
        self.reconnect_attempts = 0
        connected = False

        try:
            self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
            if self.cap.isOpened():
                connected = True
        except:
            pass

        if not connected:
            try:
                self.cap = cv2.VideoCapture(self.camera_id)
                if self.cap.isOpened():
                    connected = True
            except:
                pass

        if not connected:
            return False

        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except:
            pass

        self._try_camera_settings()

        success = False
        for attempt in range(3):
            ret, test_frame = self.cap.read()
            if ret and test_frame is not None:
                success = True
                with self.frame_lock:
                    self.latest_frame = test_frame
                    self.frame_time = time.time()
                break
            time.sleep(0.1)

        if not success:
            self.disconnect()
            return False

        self.is_connected = True
        return True

    def disconnect(self) -> None:
        if not self.is_connected and self.cap is None:
            return

        try:
            if self.cap is not None:
                try:
                    self.cap.release()
                except:
                    pass
                self.cap = None
            self.is_connected = False
        except:
            self.is_connected = False
            self.cap = None

    def get_frame(self) -> Optional[np.ndarray]:
        if not self.is_connected or self.cap is None:
            return None

        try:
            if not self.cap.isOpened():
                if self.reconnect_attempts < self.max_reconnect_attempts:
                    self.reconnect_attempts += 1
                    if self.connect():
                        self.reconnect_attempts = 0
                    else:
                        return self.latest_frame.copy() if self.latest_frame is not None else None
                else:
                    return self.latest_frame.copy() if self.latest_frame is not None else None

            ret, frame = self.cap.read()

            if not ret or frame is None:
                ret, frame = self.cap.read()

            if not ret or frame is None:
                if self.reconnect_attempts < self.max_reconnect_attempts:
                    self.reconnect_attempts += 1
                    if self.connect():
                        ret, frame = self.cap.read()
                        if ret and frame is not None:
                            with self.frame_lock:
                                self.frame_time = time.time()
                                self.latest_frame = frame
                            self.reconnect_attempts = 0
                            return frame.copy()

                return self.latest_frame.copy() if self.latest_frame is not None else None

            with self.frame_lock:
                self.frame_time = time.time()
                self.latest_frame = frame

            self.reconnect_attempts = 0

            return frame.copy()
        except Exception as e:
            if "VIDEOIO ERROR" in str(e) or "device disconnected" in str(e).lower():
                self.is_connected = False

                if self.reconnect_attempts < self.max_reconnect_attempts:
                    self.reconnect_attempts += 1
                    if self.connect():
                        try:
                            ret, frame = self.cap.read()
                            if ret and frame is not None:
                                with self.frame_lock:
                                    self.latest_frame = frame
                                    self.frame_time = time.time()
                                return frame.copy()
                        except:
                            pass

            return self.latest_frame.copy() if self.latest_frame is not None else None

    _available_cameras_cache = None

    @staticmethod
    def list_available_cameras() -> List[Dict]:
        if WebcamCamera._available_cameras_cache is not None:
            return WebcamCamera._available_cameras_cache

        available_cameras = []

        if platform.system() == 'Windows':
            try:
                from .windows_camera_detector import get_cameras_from_device_manager

                windows_cameras = get_cameras_from_device_manager()

                for camera in windows_cameras:
                    available_cameras.append({
                        'id': camera['id'],
                        'name': camera['name'],
                        'width': 1920,
                        'height': 1080,
                        'fps': 30,
                        'type': 'webcam',
                        'instance_id': camera.get('instance_id', '')
                    })

                if available_cameras:
                    WebcamCamera._available_cameras_cache = available_cameras
                    return available_cameras
            except Exception:
                pass

        for i in range(8):
            try:
                cap = cv2.VideoCapture(i)
                if not cap.isOpened():
                    continue

                ret, frame = cap.read()
                if not ret:
                    cap.release()
                    continue

                name = f"CAM{i}"

                available_cameras.append({
                    'id': i,
                    'name': name,
                    'width': 1920,
                    'height': 1080,
                    'fps': 30,
                    'type': 'webcam'
                })

                cap.release()
            except:
                if 'cap' in locals() and cap is not None:
                    try:
                        cap.release()
                    except:
                        pass

        WebcamCamera._available_cameras_cache = available_cameras

        return available_cameras
