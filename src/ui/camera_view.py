from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QComboBox, QPushButton,
    QHBoxLayout, QFrame
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap

import cv2
import numpy as np
from typing import Optional, Dict, List, Any

from ..camera.threaded_camera_manager import ThreadedCameraManager as CameraManager


class CameraView(QWidget):
    camera_selected = pyqtSignal(str, object)

    def __init__(self, position: str):
        super().__init__()

        self.position = position
        self.camera_manager = CameraManager.get_instance()
        self.camera_id = None
        self.frame = None

        self.init_ui()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(33)  # ~30 fps

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        # Camera selection
        selection_layout = QHBoxLayout()

        position_label = QLabel(f"{self.position.capitalize()} Camera:")
        position_label.setStyleSheet("color: white; font-weight: bold;")

        self.camera_combo = QComboBox()
        self.camera_combo.setStyleSheet("""
            QComboBox {
                background-color: #2c3e50;
                color: white;
                border: 1px solid #34495e;
                padding: 5px;
                border-radius: 3px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: url(down_arrow.png);
                width: 12px;
                height: 12px;
            }
            QComboBox QAbstractItemView {
                background-color: #2c3e50;
                color: white;
                selection-background-color: #3498db;
            }
        """)

        self.connect_btn = QPushButton("Connect")
        self.connect_btn.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                border: none;
                padding: 5px 10px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #2ecc71;
            }
            QPushButton:pressed {
                background-color: #16a085;
            }
            QPushButton:disabled {
                background-color: #7f8c8d;
            }
        """)

        self.disconnect_btn = QPushButton("Disconnect")
        self.disconnect_btn.setStyleSheet("""
            QPushButton {
                background-color: #c0392b;
                color: white;
                border: none;
                padding: 5px 10px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #e74c3c;
            }
            QPushButton:pressed {
                background-color: #a93226;
            }
            QPushButton:disabled {
                background-color: #7f8c8d;
            }
        """)
        self.disconnect_btn.setEnabled(False)

        selection_layout.addWidget(position_label)
        selection_layout.addWidget(self.camera_combo, 1)
        selection_layout.addWidget(self.connect_btn)
        selection_layout.addWidget(self.disconnect_btn)

        # Camera display
        self.display_label = QLabel()
        self.display_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.display_label.setMinimumSize(400, 300)
        self.display_label.setStyleSheet("background-color: #1e272e; color: white;")
        self.display_label.setText(f"No {self.position} camera connected")

        # Info display
        self.info_label = QLabel()
        self.info_label.setStyleSheet("color: white; background-color: rgba(0, 0, 0, 128); padding: 5px;")
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom)
        self.info_label.setText(f"{self.position.capitalize()} Camera: Not connected")

        # Add widgets to layout
        layout.addLayout(selection_layout)
        layout.addWidget(self.display_label, 1)
        layout.addWidget(self.info_label)

        self.setLayout(layout)

        # Connect signals
        self.connect_btn.clicked.connect(self.connect_camera)
        self.disconnect_btn.clicked.connect(self.disconnect_camera)
        self.camera_combo.currentIndexChanged.connect(self.on_camera_selection_changed)

    def update_camera_list(self, active_cameras: Dict[str, Any]):
        current_text = self.camera_combo.currentText()

        self.camera_combo.clear()
        self.camera_combo.addItem("None", None)

        available_cameras = self.camera_manager.get_available_cameras()

        for camera in available_cameras:
            camera_id = camera['id']

            # Skip cameras that are already active in other positions
            skip = False
            for pos, cam_id in active_cameras.items():
                if pos != self.position and cam_id == camera_id:
                    skip = True
                    break

            if not skip:
                self.camera_combo.addItem(camera['name'], camera_id)

        # Try to restore previous selection
        index = self.camera_combo.findText(current_text)
        if index >= 0:
            self.camera_combo.setCurrentIndex(index)

    def on_camera_selection_changed(self, index):
        self.connect_btn.setEnabled(index > 0)  # Enable if not "None"

    def connect_camera(self):
        camera_id = self.camera_combo.currentData()

        if camera_id is not None:
            if self.camera_manager.connect_camera(camera_id, self.position):
                self.camera_id = camera_id
                self.connect_btn.setEnabled(False)
                self.disconnect_btn.setEnabled(True)
                self.camera_selected.emit(self.position, camera_id)

                # Update info label
                camera_info = self.camera_manager.get_active_camera_info(self.position)
                if camera_info:
                    # Safely get resolution information
                    resolution = camera_info.get('resolution', 'Unknown')
                    fps = camera_info.get('fps', 'Unknown')

                    # Use actual values if available
                    if 'actual_width' in camera_info and 'actual_height' in camera_info:
                        resolution = f"{camera_info['actual_width']}x{camera_info['actual_height']}"

                    if 'actual_fps' in camera_info:
                        fps = camera_info['actual_fps']

                    self.info_label.setText(
                        f"{self.position.capitalize()} Camera: {camera_info['name']} "
                        f"({resolution}@{fps}fps)"
                    )
            else:
                self.display_label.setText(f"Failed to connect to {self.position} camera")

    def disconnect_camera(self):
        self.camera_manager.disconnect_camera(self.position)
        self.camera_id = None
        self.frame = None
        self.display_label.setText(f"No {self.position} camera connected")
        self.info_label.setText(f"{self.position.capitalize()} Camera: Not connected")
        self.connect_btn.setEnabled(self.camera_combo.currentIndex() > 0)
        self.disconnect_btn.setEnabled(False)
        self.camera_selected.emit(self.position, None)

    def update_frame(self):
        if self.camera_id is not None and self.camera_manager.is_camera_active(self.position):
            frame = self.camera_manager.get_frame(self.position)

            if frame is not None:
                self.frame = frame
                self.display_frame()
        elif self.frame is not None:
            # Clear the display if camera is no longer active
            self.frame = None
            self.display_label.setText(f"No {self.position} camera connected")

    def display_frame(self):
        if self.frame is None:
            return

        # Convert frame to QPixmap and display it
        frame_rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w

        # Scale the image to fit the display while maintaining aspect ratio
        display_size = self.display_label.size()

        # Create QImage and QPixmap
        image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(image)

        # Scale pixmap to fit the display label while maintaining aspect ratio
        pixmap = pixmap.scaled(
            display_size.width(),
            display_size.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )

        self.display_label.setPixmap(pixmap)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.frame is not None:
            self.display_frame()
