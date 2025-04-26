from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QStatusBar, QSplitter, QTabWidget
)
from PyQt6.QtCore import Qt, QTimer, pyqtSlot
import threading

from ..camera.threaded_camera_manager import ThreadedCameraManager as CameraManager
from .camera_view import CameraView
from .gait_analysis_view import GaitAnalysisView


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.camera_manager = CameraManager.get_instance()
        self.active_cameras = self.camera_manager.get_active_cameras()

        self.init_ui()

        # Update camera dropdowns once at startup
        self.update_camera_dropdowns()

    def init_ui(self):
        self.setWindowTitle("Gait Analysis System")
        self.setFixedSize(1920, 1080)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.setGeometry(0, 0, 1920, 1080)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_splitter.setChildrenCollapsible(False)
        main_splitter.setHandleWidth(1)

        camera_container = QWidget()
        camera_container_layout = QVBoxLayout()
        camera_container_layout.setContentsMargins(0, 0, 0, 0)
        camera_container_layout.setSpacing(0)

        camera_title = QLabel("Camera Views")
        camera_title.setStyleSheet("font-weight: bold; font-size: 16px; color: white; background-color: #2c3e50; padding: 8px;")
        camera_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        camera_container_layout.addWidget(camera_title)

        camera_widget = QWidget()
        camera_widget.setStyleSheet("background-color: #34495e; border-radius: 5px;")
        camera_layout = QGridLayout()
        camera_layout.setContentsMargins(20, 20, 20, 20)
        camera_layout.setSpacing(40)
        camera_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.front_camera_view = CameraView("front")
        self.back_camera_view = CameraView("back")
        self.left_camera_view = CameraView("left")
        self.right_camera_view = CameraView("right")

        camera_layout.addWidget(self.front_camera_view, 0, 0)
        camera_layout.addWidget(self.back_camera_view, 0, 1)
        camera_layout.addWidget(self.left_camera_view, 1, 0)
        camera_layout.addWidget(self.right_camera_view, 1, 1)

        camera_layout.setColumnStretch(0, 1)
        camera_layout.setColumnStretch(1, 1)
        camera_layout.setRowStretch(0, 1)
        camera_layout.setRowStretch(1, 1)

        camera_widget.setLayout(camera_layout)
        camera_container_layout.addWidget(camera_widget)
        camera_container.setLayout(camera_container_layout)

        analysis_container = QWidget()
        analysis_container.setFixedWidth(350)
        analysis_container_layout = QVBoxLayout()
        analysis_container_layout.setContentsMargins(0, 0, 0, 0)
        analysis_container_layout.setSpacing(0)

        analysis_title = QLabel("Gait Analysis")
        analysis_title.setStyleSheet("font-weight: bold; font-size: 16px; color: white; background-color: #2c3e50; padding: 8px;")
        analysis_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        analysis_container_layout.addWidget(analysis_title)

        analysis_widget = QWidget()
        analysis_widget.setFixedWidth(350)
        analysis_widget.setStyleSheet("background-color: #34495e; border-radius: 5px;")
        analysis_layout = QVBoxLayout()
        analysis_layout.setContentsMargins(0, 0, 0, 0)

        self.gait_analysis_view = GaitAnalysisView()
        analysis_layout.addWidget(self.gait_analysis_view)

        # Connect buttons from GaitAnalysisView
        self.start_analysis_btn = self.gait_analysis_view.start_analysis_btn
        self.stop_analysis_btn = self.gait_analysis_view.stop_analysis_btn
        self.save_data_btn = self.gait_analysis_view.save_data_btn

        analysis_widget.setLayout(analysis_layout)
        analysis_container_layout.addWidget(analysis_widget)
        analysis_container.setLayout(analysis_container_layout)

        main_splitter.addWidget(camera_container)
        main_splitter.addWidget(analysis_container)

        camera_section_width = int(1920 * 0.60)
        analysis_section_width = int(1920 * 0.40)

        main_splitter.setSizes([camera_section_width, analysis_section_width])

        main_layout.addWidget(main_splitter)

        central_widget.setLayout(main_layout)

        self.status_bar = QStatusBar()
        self.status_bar.setStyleSheet("background-color: #2c3e50; color: white;")
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

        exit_button = QPushButton("Thoát")
        exit_button.setStyleSheet("""
            QPushButton {
                background-color: #c0392b;
                color: white;
                border: none;
                padding: 5px 10px;
                border-radius: 3px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #e74c3c;
            }
        """)
        exit_button.setFixedSize(60, 30)
        exit_button.clicked.connect(self.close)

        exit_button.setParent(self)
        exit_button.move(1840, 4)

        self.front_camera_view.camera_selected.connect(self.on_camera_selected)
        self.back_camera_view.camera_selected.connect(self.on_camera_selected)
        self.left_camera_view.camera_selected.connect(self.on_camera_selected)
        self.right_camera_view.camera_selected.connect(self.on_camera_selected)

        self.start_analysis_btn.clicked.connect(self.start_analysis)
        self.stop_analysis_btn.clicked.connect(self.stop_analysis)
        self.save_data_btn.clicked.connect(self.save_analysis_data)

    def update_camera_dropdowns(self):
        # Get the latest active cameras from the camera manager
        self.active_cameras = self.camera_manager.get_active_cameras()
        active_cameras_copy = self.active_cameras.copy()

        # Update camera dropdowns in all views
        self.front_camera_view.update_camera_list(active_cameras_copy)
        self.back_camera_view.update_camera_list(active_cameras_copy)
        self.left_camera_view.update_camera_list(active_cameras_copy)
        self.right_camera_view.update_camera_list(active_cameras_copy)

    @pyqtSlot(str, object)
    def on_camera_selected(self, position, camera_id):
        self.active_cameras[position] = camera_id

        if camera_id is not None:
            self.status_bar.showMessage(f"Camera {camera_id} selected for {position} position")
        else:
            self.status_bar.showMessage(f"No camera selected for {position} position")

    def start_analysis(self):
        self.status_bar.showMessage("Analysis started")

    def stop_analysis(self):
        self.status_bar.showMessage("Analysis stopped")

    def save_analysis_data(self):
        self.status_bar.showMessage("Saving analysis data...")

    def closeEvent(self, event):
        # Disconnect all cameras
        self.front_camera_view.disconnect_camera()
        self.back_camera_view.disconnect_camera()
        self.left_camera_view.disconnect_camera()
        self.right_camera_view.disconnect_camera()

        event.accept()
