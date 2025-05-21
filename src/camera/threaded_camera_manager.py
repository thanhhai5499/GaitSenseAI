import time
import threading
from typing import Dict, List, Optional, Any, Callable

from .base_camera import BaseCamera
from .camera_detector import CameraDetector

class ThreadedCameraManager:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = ThreadedCameraManager()
        return cls._instance

    def __init__(self):
        if ThreadedCameraManager._instance is not None:
            raise RuntimeError("ThreadedCameraManager is a singleton class. Use get_instance() instead.")

        self.camera_detector = CameraDetector()
        self.active_cameras = {}
        self.camera_threads = {}
        self.camera_stop_events = {}
        self.camera_frames = {}
        self.camera_frame_locks = {}
        self.camera_change_callbacks = []
        self.ui_ready = False

        self.camera_detector.register_callback(self._on_camera_change)
        self.camera_detector.detect_cameras_once()

    def _on_camera_change(self, added_cameras, removed_cameras):
        for position, camera in list(self.active_cameras.items()):
            camera_id = camera.camera_id
            camera_type = 'webcam' if hasattr(camera, 'cap') else 'realsense'
            full_id = f"{camera_type}_{camera_id}"

            if full_id in removed_cameras:
                self.disconnect_camera(position)

        if self.ui_ready or added_cameras or removed_cameras:
            for callback in self.camera_change_callbacks:
                try:
                    callback()
                except Exception:
                    pass

    def set_ui_ready(self):
        self.ui_ready = True

        for callback in self.camera_change_callbacks:
            try:
                callback()
            except Exception:
                pass

    def register_camera_change_callback(self, callback: Callable[[], None]) -> None:
        if callback not in self.camera_change_callbacks:
            self.camera_change_callbacks.append(callback)

    def unregister_camera_change_callback(self, callback: Callable[[], None]) -> None:
        if callback in self.camera_change_callbacks:
            self.camera_change_callbacks.remove(callback)

    def get_available_cameras(self) -> List[Dict]:
        return self.camera_detector.get_available_cameras()

    def connect_camera(self, camera_id: str, position: str) -> bool:
        if position in self.active_cameras:
            return False

        for pos, cam in self.active_cameras.items():
            if camera_id == cam.camera_id:
                return False

        camera = self.camera_detector.create_camera(camera_id)
        if camera is None:
            return False

        camera.position = position

        if not camera.connect():
            return False

        test_frame = camera.get_frame()
        if test_frame is None:
            camera.disconnect()
            return False

        stop_event = threading.Event()
        frame_lock = threading.Lock()

        thread = threading.Thread(
            target=self._camera_thread,
            args=(camera, position, stop_event, frame_lock),
            daemon=True
        )
        thread.start()

        self.active_cameras[position] = camera
        self.camera_threads[position] = thread
        self.camera_stop_events[position] = stop_event
        self.camera_frame_locks[position] = frame_lock
        self.camera_frames[position] = test_frame

        return True

    def disconnect_camera(self, position: str) -> None:
        if position not in self.active_cameras:
            return

        camera = self.active_cameras[position]
        thread = self.camera_threads[position]
        stop_event = self.camera_stop_events[position]

        stop_event.set()
        thread.join(timeout=2.0)

        try:
            camera.disconnect()
        except:
            pass

        del self.active_cameras[position]
        del self.camera_threads[position]
        del self.camera_stop_events[position]

        with self.camera_frame_locks[position]:
            if position in self.camera_frames:
                del self.camera_frames[position]

        del self.camera_frame_locks[position]
        time.sleep(0.1)

    def disconnect_all_cameras(self) -> None:
        for position in list(self.active_cameras.keys()):
            self.disconnect_camera(position)

    def is_camera_active(self, position: str) -> bool:
        if position not in self.active_cameras:
            return False

        if position not in self.camera_frames:
            return False

        if self.camera_frames[position] is None:
            return False

        return True

    def get_frame(self, position: str) -> Optional[Any]:
        if position not in self.active_cameras or position not in self.camera_frames:
            return None

        with self.camera_frame_locks[position]:
            if self.camera_frames[position] is None:
                return None

            return self.camera_frames[position].copy()

    def get_active_cameras(self) -> Dict[str, str]:
        return {position: camera.camera_id for position, camera in self.active_cameras.items()}

    def get_active_camera_info(self, position: str) -> Optional[Dict]:
        if position not in self.active_cameras:
            return None

        return self.active_cameras[position].get_camera_info()

    def _camera_thread(self, camera: BaseCamera, position: str, stop_event: threading.Event, frame_lock: threading.Lock) -> None:
        consecutive_failures = 0
        last_successful_frame_time = time.time()
        reconnect_attempts = 0
        max_reconnect_attempts = 2
        camera_disconnected = False

        target_fps = getattr(camera, 'actual_fps', 30)
        if target_fps <= 0:
            target_fps = 30
        frame_interval = 1.0 / target_fps

        last_frame_time = time.time()

        while not stop_event.is_set():
            try:
                current_time = time.time()

                if camera_disconnected:
                    time.sleep(0.1)
                    continue

                time_since_last_frame = current_time - last_frame_time
                if time_since_last_frame < frame_interval:
                    sleep_time = max(0.001, frame_interval - time_since_last_frame)
                    time.sleep(sleep_time)
                    continue

                last_frame_time = current_time
                frame = camera.get_frame()

                if frame is not None:
                    consecutive_failures = 0
                    last_successful_frame_time = current_time
                    reconnect_attempts = 0

                    with frame_lock:
                        self.camera_frames[position] = frame
                else:
                    consecutive_failures += 1

                    if consecutive_failures > 10 or (current_time - last_successful_frame_time > 2.0):
                        if reconnect_attempts < max_reconnect_attempts:
                            camera.disconnect()
                            time.sleep(0.2)
                            if camera.connect():
                                consecutive_failures = 0
                                last_successful_frame_time = time.time()
                                reconnect_attempts += 1
                            else:
                                reconnect_attempts += 1
                                time.sleep(0.5)
                        else:
                            camera_disconnected = True

                            for callback in self.camera_change_callbacks:
                                try:
                                    callback()
                                except Exception:
                                    pass

            except Exception as e:
                consecutive_failures += 1
                time.sleep(0.1)

                if "device disconnected" in str(e).lower() or "device not found" in str(e).lower():
                    camera_disconnected = True

                    for callback in self.camera_change_callbacks:
                        try:
                            callback()
                        except Exception:
                            pass

        try:
            camera.disconnect()
        except:
            pass
