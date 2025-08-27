"""
Camera handler for RGB camera with 1920x1080@60fps support
"""

import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QImage, QPixmap
from config.settings import CAMERA_SETTINGS


class CameraHandler(QThread):
    """Handle camera operations in a separate thread"""
    
    frame_ready = pyqtSignal(np.ndarray)
    error_occurred = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.camera = None
        self.is_running = False
        self.width = CAMERA_SETTINGS["width"]
        self.height = CAMERA_SETTINGS["height"]
        self.fps = CAMERA_SETTINGS["fps"]
        self.device_id = CAMERA_SETTINGS["device_id"]
        
    def initialize_camera(self):
        """Initialize camera with specified settings - optimized for Brio300"""
        try:
            print(f"🎥 Initializing camera {self.device_id}")
            
            # Try different backends for better Brio300 compatibility
            backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
            
            for backend in backends:
                try:
                    print(f"Trying backend: {backend}")
                    self.camera = cv2.VideoCapture(self.device_id, backend)
                    
                    if self.camera.isOpened():
                        print(f"✅ Opened with backend {backend}")
                        break
                    else:
                        if self.camera:
                            self.camera.release()
                        self.camera = None
                except Exception as e:
                    print(f"Backend {backend} failed: {e}")
                    if self.camera:
                        self.camera.release()
                    self.camera = None
                    continue
            
            if not self.camera or not self.camera.isOpened():
                raise Exception(f"Cannot open camera {self.device_id} with any backend")
            
            # Set buffer size to prevent lag
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Set camera properties with error checking
            try:
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self.camera.set(cv2.CAP_PROP_FPS, self.fps)
            except Exception as e:
                print(f"⚠️ Warning setting properties: {e}")
            
            # Test frame reading with timeout
            import time
            start_time = time.time()
            ret, frame = None, None
            
            for _ in range(10):  # Try 10 times
                ret, frame = self.camera.read()
                if ret and frame is not None:
                    break
                time.sleep(0.1)
            
            if not ret or frame is None:
                raise Exception("Cannot read frames from camera")
            
            # Verify settings
            actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.camera.get(cv2.CAP_PROP_FPS)
            
            print(f"✅ Camera initialized: {actual_width}x{actual_height}@{actual_fps}fps")
            
            return True
            
        except Exception as e:
            error_msg = f"Camera initialization failed: {str(e)}"
            print(f"❌ {error_msg}")
            self.error_occurred.emit(error_msg)
            
            if self.camera:
                try:
                    self.camera.release()
                except:
                    pass
                self.camera = None
            
            return False
    
    def start_capture(self):
        """Start camera capture"""
        if self.initialize_camera():
            self.is_running = True
            self.start()
    
    def stop_capture(self):
        """Stop camera capture"""
        self.is_running = False
        if self.camera:
            self.camera.release()
            self.camera = None
    
    def run(self):
        """Main camera capture loop - optimized for Brio300"""
        frame_count = 0
        failed_reads = 0
        max_failed_reads = 10
        
        while self.is_running and self.camera and self.camera.isOpened():
            try:
                ret, frame = self.camera.read()
                if ret and frame is not None:
                    # Reset failed counter
                    failed_reads = 0
                    frame_count += 1
                    
                    # No flip - show camera as normal view
                    # frame = cv2.flip(frame, 1)  # Removed mirror effect
                    
                    # Add frame info for debugging
                    if frame_count % 30 == 0:  # Every 30 frames
                        print(f"📹 Frame {frame_count}: {frame.shape[1]}x{frame.shape[0]}")
                    
                    self.frame_ready.emit(frame)
                else:
                    failed_reads += 1
                    print(f"⚠️ Failed to read frame ({failed_reads}/{max_failed_reads})")
                    
                    if failed_reads >= max_failed_reads:
                        self.error_occurred.emit("Too many failed frame reads from camera")
                        break
                    
                    # Small delay before retry
                    import time
                    time.sleep(0.01)
                    
            except Exception as e:
                print(f"❌ Camera run error: {e}")
                self.error_occurred.emit(f"Camera error: {str(e)}")
                break
                
        print("🔌 Camera capture loop ended")
        self.stop_capture()


class CameraWidget:
    """Widget wrapper for camera display"""
    
    def __init__(self):
        self.current_frame = None
        
    def frame_to_qimage(self, frame):
        """Convert OpenCV frame to QImage"""
        if frame is None:
            return None
            
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        q_image = QImage(rgb_frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        return q_image
    
    def frame_to_pixmap(self, frame):
        """Convert frame to QPixmap maintaining original size"""
        q_image = self.frame_to_qimage(frame)
        if q_image is None:
            return None
            
        pixmap = QPixmap.fromImage(q_image)
        return pixmap
    
    def update_frame(self, frame):
        """Update current frame"""
        self.current_frame = frame

