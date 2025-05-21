import time
import threading
import platform
from typing import Dict, List, Callable, Any, Optional

from .webcam_camera import WebcamCamera
from .realsense_camera import RealSenseCamera, REALSENSE_AVAILABLE

if platform.system() == 'Windows':
    from .windows_camera_detector import get_cameras_from_device_manager

class CameraDetector:
    def __init__(self):
        self.available_cameras = {}
        self.callbacks = []
        self.detection_thread = None
        self.detection_lock = threading.Lock()
        self.detection_complete = threading.Event()

    _detection_cache = None
    _last_detection_time = 0
    _detection_cache_valid_duration = 10

    def detect_cameras_once(self) -> None:
        current_time = time.time()

        if (self._detection_cache is not None and
            current_time - self._last_detection_time < self._detection_cache_valid_duration and
            not (self.detection_thread and self.detection_thread.is_alive())):

            with self.detection_lock:
                self.available_cameras = self._detection_cache.copy()

            self.detection_complete.set()
            self._notify_callbacks({}, {})
            return

        with self.detection_lock:
            self.available_cameras = {
                "detecting_0": {
                    'id': "detecting_0",
                    'name': "Detecting cameras...",
                    'width': 1920,
                    'height': 1080,
                    'fps': 30,
                    'type': 'placeholder',
                    'raw_id': 0
                }
            }

        self.detection_complete.clear()
        self.detection_thread = threading.Thread(
            target=self._detect_cameras_thread,
            daemon=True
        )
        self.detection_thread.start()

    def _detect_cameras_thread(self) -> None:
        try:
            self._detect_cameras()
        except Exception:
            pass
        finally:
            self.detection_complete.set()

    def _detect_cameras(self) -> None:
        try:
            available_cameras = {}

            if platform.system() == 'Windows':
                windows_cameras = get_cameras_from_device_manager()

                for camera in windows_cameras:
                    if ('depth' in camera['name'].lower() and
                        'rgb' not in camera['name'].lower()):
                        continue

                    camera_id = f"webcam_{camera['id']}"
                    available_cameras[camera_id] = {
                        'id': camera_id,
                        'name': camera['name'],
                        'width': 1920,
                        'height': 1080,
                        'fps': 30,
                        'type': 'webcam',
                        'raw_id': camera['id'],
                        'instance_id': camera.get('instance_id', '')
                    }
            else:
                webcams = WebcamCamera.list_available_cameras()

                for webcam in webcams:
                    camera_id = f"webcam_{webcam['id']}"
                    available_cameras[camera_id] = {
                        'id': camera_id,
                        'name': webcam['name'],
                        'width': 1920,
                        'height': 1080,
                        'fps': 30,
                        'type': 'webcam',
                        'raw_id': webcam['id']
                    }

            if REALSENSE_AVAILABLE:
                realsense_cameras = RealSenseCamera.list_available_cameras()

                for realsense in realsense_cameras:
                    if ('depth' in realsense['name'].lower() and
                        'rgb' not in realsense['name'].lower()):
                        continue

                    camera_id = f"realsense_{realsense['id']}"
                    available_cameras[camera_id] = {
                        'id': camera_id,
                        'name': realsense['name'],
                        'width': 1920,
                        'height': 1080,
                        'fps': 30,
                        'type': 'realsense',
                        'raw_id': realsense['id']
                    }

            added_cameras = {k: v for k, v in available_cameras.items() if k not in self.available_cameras}
            removed_cameras = {k: v for k, v in self.available_cameras.items() if k not in available_cameras}

            with self.detection_lock:
                self.available_cameras = available_cameras
                self._detection_cache = available_cameras.copy()
                self._last_detection_time = time.time()

            if added_cameras or removed_cameras:
                self._notify_callbacks(added_cameras, removed_cameras)
            elif len(available_cameras) > 0:
                self._notify_callbacks({}, {})

        except Exception:
            pass

    def get_available_cameras(self) -> List[Dict]:
        with self.detection_lock:
            return list(self.available_cameras.values())

    def register_callback(self, callback: Callable[[Dict[str, Dict], Dict[str, Dict]], None]) -> None:
        if callback not in self.callbacks:
            self.callbacks.append(callback)

    def unregister_callback(self, callback: Callable[[Dict[str, Dict], Dict[str, Dict]], None]) -> None:
        if callback in self.callbacks:
            self.callbacks.remove(callback)

    def _notify_callbacks(self, added_cameras: Dict[str, Dict], removed_cameras: Dict[str, Dict]) -> None:
        for callback in self.callbacks:
            try:
                callback(added_cameras, removed_cameras)
            except Exception:
                pass

    def create_camera(self, camera_id: str) -> Any:
        if self.detection_thread and self.detection_thread.is_alive() and not self.detection_complete.is_set():
            if not self.detection_complete.wait(timeout=1.0):
                pass

        with self.detection_lock:
            if camera_id not in self.available_cameras:
                return None

            camera_info = self.available_cameras[camera_id]

            if camera_info['type'] == 'placeholder':
                return None

            camera_type = camera_info['type']
            raw_id = camera_info['raw_id']
            instance_id = camera_info.get('instance_id', '')

        try:
            if camera_type == 'webcam':
                if isinstance(raw_id, str):
                    try:
                        raw_id = int(raw_id)
                    except ValueError:
                        return None

                if instance_id and platform.system() == 'Windows':
                    return WebcamCamera(
                        camera_id=raw_id,
                        name=camera_info['name'],
                        width=1920,
                        height=1080,
                        fps=30,
                        instance_id=instance_id
                    )
                else:
                    return WebcamCamera(
                        camera_id=raw_id,
                        name=camera_info['name'],
                        width=1920,
                        height=1080,
                        fps=30
                    )
            elif camera_type == 'realsense':
                return RealSenseCamera(
                    camera_id=raw_id,
                    name=camera_info['name'],
                    width=1920,
                    height=1080,
                    fps=30
                )
            else:
                return None
        except Exception:
            return None
