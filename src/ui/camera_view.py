from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QComboBox, QPushButton,
    QHBoxLayout, QFrame, QMessageBox, QCheckBox
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap

import cv2
import numpy as np
from typing import Optional, Dict, List, Any

from ..camera.threaded_camera_manager import ThreadedCameraManager as CameraManager
from ..models.mediapipe_model import MediaPipePoseModel


class CameraView(QWidget):
    camera_selected = pyqtSignal(str, object)
    pose_detected = pyqtSignal(str, dict)  # Signal for pose detection results

    def __init__(self, position: str):
        super().__init__()

        self.position = position
        self.camera_manager = CameraManager.get_instance()
        self.camera_id = None
        self.frame = None
        self.processed_frame = None

        # MediaPipe pose model
        self.pose_model = MediaPipePoseModel(model_complexity=1)
        self.show_skeleton = True
        self.landmarks = {}

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

        # Skeleton visualization toggle
        skeleton_layout = QHBoxLayout()
        self.skeleton_checkbox = QCheckBox("Show Skeleton")
        self.skeleton_checkbox.setChecked(self.show_skeleton)
        self.skeleton_checkbox.setStyleSheet("color: white;")
        self.skeleton_checkbox.toggled.connect(self.toggle_skeleton)
        skeleton_layout.addWidget(self.skeleton_checkbox)
        skeleton_layout.addStretch()

        # Add widgets to layout
        layout.addLayout(selection_layout)
        layout.addWidget(self.display_label, 1)
        layout.addLayout(skeleton_layout)
        layout.addWidget(self.info_label)

        self.setLayout(layout)

        # Connect signals
        self.connect_btn.clicked.connect(self.connect_camera)
        self.disconnect_btn.clicked.connect(self.disconnect_camera)
        self.camera_combo.currentIndexChanged.connect(self.on_camera_selection_changed)

    def update_camera_list(self, active_cameras: Dict[str, Any]):
        """
        Update the camera dropdown list with available cameras.

        Args:
            active_cameras: Dictionary mapping positions to camera IDs
        """
        # Save current selection and state
        current_camera_id = self.camera_combo.currentData()
        was_connected = self.camera_id is not None and self.camera_manager.is_camera_active(self.position)

        # Clear the combo box
        self.camera_combo.clear()
        self.camera_combo.addItem("None", None)

        # Get all available cameras from the camera manager
        available_cameras = self.camera_manager.get_available_cameras()

        # Check if our current camera is still available
        current_camera_available = False
        if self.camera_id is not None:
            for camera in available_cameras:
                if camera['id'] == self.camera_id:
                    current_camera_available = True
                    break

        # If our camera was connected but is no longer available, it was disconnected
        if was_connected and not current_camera_available and self.camera_id is not None:
            # Camera was disconnected, update UI
            self.camera_id = None
            self.frame = None
            self.display_label.setText(f"Camera disconnected")
            self.info_label.setText(f"{self.position.capitalize()} Camera: Not connected")
            self.disconnect_btn.setEnabled(False)

        # Create a list of cameras that are actually active (connected and working)
        active_camera_ids = []
        for pos, cam_id in active_cameras.items():
            if self.camera_manager.is_camera_active(pos):
                active_camera_ids.append(cam_id)

        # Add all available cameras to the dropdown
        for camera in available_cameras:
            camera_id = camera['id']

            # Skip cameras that are actually active in other positions
            skip = False
            for pos, cam_id in active_cameras.items():
                # Only skip if the position is different, the camera ID matches, and the camera is actually active
                if pos != self.position and cam_id == camera_id and self.camera_manager.is_camera_active(pos):
                    skip = True
                    break

            if not skip:
                # Add camera to dropdown with name and resolution info
                camera_name = camera['name']
                resolution = f"{camera['width']}x{camera['height']}"
                fps = camera['fps']
                display_name = f"{camera_name} ({resolution}@{fps}fps)"

                self.camera_combo.addItem(display_name, camera_id)

        # Restore previous selection if possible
        if current_camera_id is not None:
            index = self.camera_combo.findData(current_camera_id)
            if index >= 0:
                self.camera_combo.setCurrentIndex(index)
            else:
                # If previous camera is no longer available, select "None"
                self.camera_combo.setCurrentIndex(0)

        # Enable/disable buttons based on current state
        self.connect_btn.setEnabled(self.camera_combo.currentIndex() > 0 and self.camera_id is None)
        self.disconnect_btn.setEnabled(self.camera_id is not None and self.camera_manager.is_camera_active(self.position))

    def on_camera_selection_changed(self, index):
        """
        Handle camera selection change in the dropdown.

        Args:
            index: New selected index
        """
        # Get the selected camera ID
        camera_id = self.camera_combo.currentData()

        # Only proceed if a camera is selected (not "None")
        if index > 0 and camera_id is not None:
            # Check if this camera is already in use in another position
            active_cameras = self.camera_manager.get_active_cameras()
            for pos, cam_id in active_cameras.items():
                if pos != self.position and cam_id == camera_id and self.camera_manager.is_camera_active(pos):
                    # Camera is already in use, show warning dialog
                    self.show_camera_in_use_warning(pos)
                    # Reset selection to previous or None
                    if self.camera_id is not None:
                        # Find index of current camera
                        prev_index = self.camera_combo.findData(self.camera_id)
                        if prev_index >= 0:
                            # Block signals to prevent recursion
                            self.camera_combo.blockSignals(True)
                            self.camera_combo.setCurrentIndex(prev_index)
                            self.camera_combo.blockSignals(False)
                        else:
                            # If previous camera is no longer available, select "None"
                            self.camera_combo.blockSignals(True)
                            self.camera_combo.setCurrentIndex(0)
                            self.camera_combo.blockSignals(False)
                    else:
                        # No previous camera, select "None"
                        self.camera_combo.blockSignals(True)
                        self.camera_combo.setCurrentIndex(0)
                        self.camera_combo.blockSignals(False)
                    return

        # Only enable connect button if a camera is selected and no camera is currently connected
        self.connect_btn.setEnabled(index > 0 and self.camera_id is None)

    def show_camera_in_use_warning(self, position):
        """
        Show a warning dialog when a camera is already in use.

        Args:
            position: Position where the camera is already in use
        """
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Icon.Warning)
        msg.setWindowTitle("Camera Already In Use")
        msg.setText(f"This camera is already in use as the {position} camera.")
        msg.setInformativeText("Please select a different camera or disconnect the camera from its current position first.")
        msg.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg.exec()

    def connect_camera(self):
        """
        Connect to the selected camera.
        """
        camera_id = self.camera_combo.currentData()

        if camera_id is not None:
            # Check if this camera is already in use in another position
            active_cameras = self.camera_manager.get_active_cameras()
            for pos, cam_id in active_cameras.items():
                if pos != self.position and cam_id == camera_id and self.camera_manager.is_camera_active(pos):
                    # Camera is already in use, show warning dialog
                    self.show_camera_in_use_warning(pos)
                    return

            # Disable both buttons during connection attempt to prevent multiple clicks
            self.connect_btn.setEnabled(False)
            self.disconnect_btn.setEnabled(False)

            # Update display to show connecting status
            self.display_label.setText(f"Connecting to {self.position} camera...")

            # Process events to update UI immediately
            from PyQt6.QtCore import QCoreApplication
            QCoreApplication.processEvents()

            # Try to connect to the camera
            if self.camera_manager.connect_camera(camera_id, self.position):
                self.camera_id = camera_id
                self.disconnect_btn.setEnabled(True)

                # Get the camera name from the dropdown
                camera_name = self.camera_combo.currentText()
                # Update the info label with the connected camera name
                self.info_label.setText(f"{self.position.capitalize()} Camera: Connected to {camera_name}")

                self.camera_selected.emit(self.position, camera_id)
            else:
                # Connection failed, re-enable connect button
                self.connect_btn.setEnabled(True)
                self.display_label.setText(f"Failed to connect to {self.position} camera")
                self.info_label.setText(f"{self.position.capitalize()} Camera: Connection failed")

    def disconnect_camera(self):
        """
        Disconnect from the current camera.
        """
        # Disable both buttons during disconnection to prevent multiple clicks
        self.connect_btn.setEnabled(False)
        self.disconnect_btn.setEnabled(False)

        # Update display to show disconnecting status
        self.display_label.setText(f"Disconnecting {self.position} camera...")

        # Process events to update UI immediately
        from PyQt6.QtCore import QCoreApplication
        QCoreApplication.processEvents()

        # Disconnect the camera
        self.camera_manager.disconnect_camera(self.position)

        # Update UI state
        self.camera_id = None
        self.frame = None
        self.landmarks = {}
        self.display_label.setText(f"No {self.position} camera connected")
        self.info_label.setText(f"{self.position.capitalize()} Camera: Not connected")

        # Re-enable connect button if a camera is selected in the dropdown
        self.connect_btn.setEnabled(self.camera_combo.currentIndex() > 0)
        self.disconnect_btn.setEnabled(False)

        # Notify that camera has been disconnected
        self.camera_selected.emit(self.position, None)

    def update_frame(self):
        """
        Update the camera frame display.
        Called periodically by the timer.
        """
        # Check if we have a camera ID and if it's still active
        if self.camera_id is not None:
            if self.camera_manager.is_camera_active(self.position):
                # Camera is active, try to get a frame
                frame = self.camera_manager.get_frame(self.position)

                if frame is not None:
                    # Got a valid frame, display it
                    self.frame = frame
                    self.display_frame()

                    # Make sure disconnect button is enabled
                    if not self.disconnect_btn.isEnabled():
                        self.disconnect_btn.setEnabled(True)

                    # Ensure the info label shows the connected camera
                    if not self.info_label.text().startswith(f"{self.position.capitalize()} Camera: Connected to"):
                        camera_name = self.camera_combo.currentText()
                        self.info_label.setText(f"{self.position.capitalize()} Camera: Connected to {camera_name}")
            else:
                # Camera is no longer active but we still have a camera ID
                # This means the camera was disconnected unexpectedly
                if self.frame is not None:
                    # Clear the display
                    self.frame = None
                    self.display_label.setText(f"Camera disconnected")
                    self.info_label.setText(f"{self.position.capitalize()} Camera: Not connected")

                    # Update button states
                    self.disconnect_btn.setEnabled(False)
                    self.connect_btn.setEnabled(self.camera_combo.currentIndex() > 0)

                    # Clear camera ID since it's no longer active
                    self.camera_id = None

    def toggle_skeleton(self, checked):
        """
        Toggle skeleton visualization.

        Args:
            checked: Whether the checkbox is checked
        """
        self.show_skeleton = checked

    def process_frame_with_pose(self, frame):
        """
        Process frame with MediaPipe pose detection.

        Args:
            frame: Input frame

        Returns:
            Processed frame with skeleton visualization
        """
        if frame is None:
            return None

        # Only process every other frame to improve performance
        if self.show_skeleton:
            try:
                # Detect pose and get landmarks
                self.landmarks, processed_frame = self.pose_model.detect_pose(frame, draw_full_body=False)

                # Emit pose detection results
                if self.landmarks:
                    self.pose_detected.emit(self.position, self.landmarks)

                return processed_frame
            except Exception as e:
                print(f"Error processing frame with MediaPipe: {e}")
                return frame

        return frame

    def display_frame(self):
        if self.frame is None:
            return

        # Process frame with pose detection if enabled
        if self.show_skeleton:
            display_frame = self.process_frame_with_pose(self.frame)
        else:
            display_frame = self.frame

        # Convert frame to QPixmap and display it
        frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
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
