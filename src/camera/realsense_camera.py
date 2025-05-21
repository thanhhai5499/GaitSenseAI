import numpy as np
import time
import threading
import os
from typing import Optional, Dict, List, Any

from .base_camera import BaseCamera

os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

REALSENSE_AVAILABLE = False
try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    pass

class RealSenseCamera(BaseCamera):
    def __init__(self, camera_id: str, name: str = None, width: int = 1920, height: int = 1080, fps: int = 30, position: str = None):
        if name is None:
            name = f"RealSense {camera_id}"

        super().__init__(camera_id, name, width, height, fps, position)

        if not REALSENSE_AVAILABLE:
            return

        self.pipeline = None
        self.config = None
        self.profile = None
        self.frame_lock = threading.Lock()
        self.latest_frame = None
        self.frame_time = 0
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 3

    def connect(self) -> bool:
        if not REALSENSE_AVAILABLE:
            return False

        if self.is_connected and self.pipeline is not None:
            return True

        self.disconnect()

        try:
            self.pipeline = rs.pipeline()
            self.config = rs.config()

            try:
                self.config.enable_device(self.camera_id)
            except:
                pass

            self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)

            try:
                self.profile = self.pipeline.start(self.config)
                color_stream = self.profile.get_stream(rs.stream.color)
                video_stream_profile = color_stream.as_video_stream_profile()
                intrinsics = video_stream_profile.get_intrinsics()
                self.actual_width = intrinsics.width
                self.actual_height = intrinsics.height
            except Exception:
                self.disconnect()
                return False

            for _ in range(3):
                try:
                    frames = self.pipeline.wait_for_frames(timeout_ms=500)
                    color_frame = frames.get_color_frame()
                    if color_frame:
                        frame = np.asanyarray(color_frame.get_data())
                        with self.frame_lock:
                            self.latest_frame = frame
                            self.frame_time = time.time()
                        self.is_connected = True
                        self.reconnect_attempts = 0
                        return True
                except:
                    time.sleep(0.1)

            self.disconnect()
            return False

        except:
            self.disconnect()
            return False

    def disconnect(self) -> bool:
        if not REALSENSE_AVAILABLE:
            return True

        try:
            if self.pipeline is not None:
                try:
                    self.pipeline.stop()
                except:
                    pass
                self.pipeline = None
                self.config = None
                self.profile = None
                self.is_connected = False
        except:
            pass

        return True

    def get_frame(self) -> Optional[np.ndarray]:
        if not REALSENSE_AVAILABLE or not self.is_connected or self.pipeline is None:
            return None

        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=500)
            color_frame = frames.get_color_frame()

            if not color_frame:
                return None

            frame = np.asanyarray(color_frame.get_data())

            with self.frame_lock:
                self.frame_time = time.time()
                self.latest_frame = frame

            return frame.copy()
        except Exception as e:
            if "device disconnected" in str(e).lower():
                self.is_connected = False

                if self.reconnect_attempts < self.max_reconnect_attempts:
                    self.reconnect_attempts += 1
                    if self.connect():
                        try:
                            frames = self.pipeline.wait_for_frames(timeout_ms=500)
                            color_frame = frames.get_color_frame()
                            if color_frame:
                                frame = np.asanyarray(color_frame.get_data())
                                with self.frame_lock:
                                    self.latest_frame = frame
                                    self.frame_time = time.time()
                                return frame.copy()
                        except:
                            pass

            return self.latest_frame.copy() if self.latest_frame is not None else None

    @staticmethod
    def list_available_cameras() -> List[Dict]:
        if not REALSENSE_AVAILABLE:
            return []

        available_cameras = []
        try:
            context = rs.context()
            devices = context.query_devices()

            for device in devices:
                try:
                    serial = device.get_info(rs.camera_info.serial_number)
                    name = device.get_info(rs.camera_info.name)

                    if ('depth' in name.lower() and
                        'rgb' not in name.lower()):
                        continue

                    camera_name = f"RealSense {name} ({serial})"

                    available_cameras.append({
                        'id': serial,
                        'name': camera_name,
                        'width': 1920,
                        'height': 1080,
                        'fps': 30,
                        'type': 'realsense'
                    })
                except:
                    pass
        except Exception as e:
            pass

        return available_cameras
