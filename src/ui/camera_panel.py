"""
Camera panel component for displaying RGB camera feed
"""

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QPixmap, QPalette


class CameraStatusBar(QFrame):
    """Status bar showing camera information"""
    
    def __init__(self):
        super().__init__()
        self.setFixedHeight(40)
        self.setStyleSheet("""
            QFrame {
                background-color: #f0f0f0;
                border-bottom: 1px solid #cccccc;
            }
        """)
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup status bar UI"""
        layout = QHBoxLayout()
        layout.setContentsMargins(15, 5, 15, 5)
        
        # Camera status
        self.status_label = QLabel("Camera: Disconnected")
        self.status_label.setStyleSheet("color: #d73527; font-weight: bold;")
        
        # Resolution info - Updated for Brio300
        self.resolution_label = QLabel("1280x720@30fps")
        self.resolution_label.setStyleSheet("color: #333333;")
        
        # FPS counter
        self.fps_label = QLabel("FPS: 0")
        self.fps_label.setStyleSheet("color: #0078d4;")
        
        layout.addWidget(self.status_label)
        layout.addStretch()
        layout.addWidget(self.resolution_label)
        layout.addWidget(QLabel("|"))
        layout.addWidget(self.fps_label)
        
        self.setLayout(layout)
    
    def update_status(self, connected: bool):
        """Update camera connection status"""
        if connected:
            self.status_label.setText("Camera: Connected")
            self.status_label.setStyleSheet("color: #0078d4; font-weight: bold;")
        else:
            self.status_label.setText("Camera: Disconnected")
            self.status_label.setStyleSheet("color: #d73527; font-weight: bold;")
    
    def update_fps(self, fps: float):
        """Update FPS display"""
        self.fps_label.setText(f"FPS: {fps:.1f}")


class CameraDisplay(QLabel):
    """Widget for displaying camera feed with 16:9 aspect ratio"""
    
    def __init__(self):
        super().__init__()
        self.setStyleSheet("""
            QLabel {
                background-color: #f0f0f0;
                border: 1px solid #cccccc;
                color: #666666;
            }
        """)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setScaledContents(False)  # Don't stretch - maintain aspect ratio
        self.setText("Camera feed will appear here")
        self.setFont(QFont("Arial", 18))
        
        # Set minimum size to maintain 16:9 aspect ratio (smaller for more sidebar space)
        self.setMinimumSize(480, 270)  # 16:9 aspect ratio - smaller
        
        # Override size hint to maintain aspect ratio
        self._aspect_ratio = 16.0 / 9.0
        
    def update_frame(self, pixmap: QPixmap):
        """Update camera frame with proper aspect ratio - no distortion"""
        if pixmap:
            # Simply scale maintaining original aspect ratio
            scaled_pixmap = pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio, 
                Qt.TransformationMode.SmoothTransformation
            )
            self.setPixmap(scaled_pixmap)
        else:
            self.setText("No camera feed")
    
    def clear_frame(self):
        """Clear camera display"""
        self.clear()
        self.setText("Camera feed will appear here")
    
    def sizeHint(self):
        """Return preferred size maintaining 16:9 aspect ratio"""
        from PyQt6.QtCore import QSize
        return QSize(960, 540)  # 16:9 ratio
    
    def heightForWidth(self, width):
        """Maintain aspect ratio when resizing"""
        return int(width / self._aspect_ratio)
    
    def hasHeightForWidth(self):
        """Enable aspect ratio constraint"""
        return True


class CameraOverlay(QWidget):
    """Overlay widget for displaying additional information on camera feed"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setStyleSheet("background-color: transparent;")
        
        # Countdown timer
        self.countdown_timer = QTimer()
        self.countdown_timer.timeout.connect(self.update_countdown)
        self.remaining_seconds = 0
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup overlay UI"""
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Top overlay info
        self.info_label = QLabel()
        self.info_label.setStyleSheet("""
            QLabel {
                color: #ffffff;
                background-color: rgba(0, 120, 212, 180);
                padding: 5px;
                border-radius: 4px;
                font-weight: bold;
            }
        """)
        self.info_label.hide()
        
        # Countdown timer (top-left)
        self.countdown_label = QLabel("⏱️ 01:00")
        self.countdown_label.setStyleSheet("""
            QLabel {
                color: #ffffff;
                background-color: rgba(255, 69, 0, 220);
                padding: 10px 15px;
                border-radius: 8px;
                font-weight: bold;
                font-size: 18px;
                font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
                border: 2px solid rgba(255, 69, 0, 255);
            }
        """)
        self.countdown_label.setFixedHeight(40)
        self.countdown_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.countdown_label.hide()
        
        # Create horizontal layout for top items
        top_layout = QHBoxLayout()
        top_layout.addWidget(self.countdown_label)
        top_layout.addStretch()
        top_layout.addWidget(self.info_label)
        
        layout.addLayout(top_layout)
        layout.addStretch()
        
        self.setLayout(layout)
    
    def show_info(self, text: str):
        """Show overlay information"""
        self.info_label.setText(text)
        self.info_label.show()
    
    def hide_info(self):
        """Hide overlay information"""
        self.info_label.hide()
    
    def start_countdown(self, seconds: int):
        """Start countdown timer"""
        self.remaining_seconds = seconds
        self.countdown_label.show()
        self.countdown_label.raise_()  # Bring to front
        self.update_countdown()
        self.countdown_timer.start(1000)  # Update every second
    
    def stop_countdown(self):
        """Stop countdown timer"""
        self.countdown_timer.stop()
        self.countdown_label.hide()
    
    def show_countdown_test(self):
        """Show countdown for testing (always visible)"""
        self.countdown_label.setText("⏱️ 01:00")
        self.countdown_label.show()
        self.countdown_label.raise_()
    
    def update_countdown(self):
        """Update countdown display"""
        if self.remaining_seconds > 0:
            minutes = self.remaining_seconds // 60
            seconds = self.remaining_seconds % 60
            self.countdown_label.setText(f"⏱️ {minutes:02d}:{seconds:02d}")
            self.remaining_seconds -= 1
        else:
            self.stop_countdown()


class CameraPanel(QWidget):
    """Main camera panel widget"""
    
    frame_clicked = pyqtSignal()  # Signal when camera display is clicked
    
    def __init__(self):
        super().__init__()
        self.setStyleSheet("""
            QWidget {
                background-color: #ffffff;
            }
        """)
        
        self.frame_count = 0
        self.fps_counter = 0
        self.fps_timer_start = None
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup camera panel layout"""
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Status bar
        self.status_bar = CameraStatusBar()
        
        # Camera display container
        display_container = QFrame()
        display_container.setStyleSheet("""
            QFrame {
                background-color: #f8f8f8;
                border: none;
            }
        """)
        
        display_layout = QVBoxLayout()
        display_layout.setContentsMargins(5, 5, 5, 5)
        
        # Camera display
        self.camera_display = CameraDisplay()
        
        # Overlay
        self.overlay = CameraOverlay(self.camera_display)
        
        display_layout.addWidget(self.camera_display)
        display_container.setLayout(display_layout)
        
        layout.addWidget(self.status_bar)
        layout.addWidget(display_container, 1)  # Give camera display most space
        
        self.setLayout(layout)
    
    def update_camera_frame(self, pixmap: QPixmap):
        """Update camera display with new frame"""
        self.camera_display.update_frame(pixmap)
        
        # Update FPS counter
        self.frame_count += 1
        if self.frame_count % 30 == 0:  # Update FPS every 30 frames
            import time
            current_time = time.time()
            if self.fps_timer_start:
                elapsed = current_time - self.fps_timer_start
                if elapsed > 0:
                    fps = 30 / elapsed
                    self.status_bar.update_fps(fps)
            self.fps_timer_start = current_time
    
    def set_camera_connected(self, connected: bool):
        """Update camera connection status"""
        self.status_bar.update_status(connected)
        if not connected:
            self.camera_display.clear_frame()
    
    def show_overlay_info(self, text: str):
        """Show overlay information"""
        self.overlay.show_info(text)
    
    def hide_overlay_info(self):
        """Hide overlay information"""
        self.overlay.hide_info()
    
    def start_countdown(self, seconds: int):
        """Start countdown timer"""
        self.overlay.start_countdown(seconds)
    
    def stop_countdown(self):
        """Stop countdown timer"""
        self.overlay.stop_countdown()
    
    def show_countdown_test(self):
        """Show countdown for testing"""
        self.overlay.show_countdown_test()
    
    def resizeEvent(self, event):
        """Handle resize event to update overlay"""
        super().resizeEvent(event)
        if hasattr(self, 'overlay'):
            self.overlay.resize(self.camera_display.size())
            self.overlay.move(self.camera_display.pos())

