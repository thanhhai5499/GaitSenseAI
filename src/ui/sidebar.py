"""
Sidebar component for displaying gait analysis metrics
"""

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QFrame, QPushButton, QScrollArea, QGroupBox)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QPalette


class MetricCard(QFrame):
    """Individual metric display card"""
    
    def __init__(self, title: str, value: str = "0", unit: str = ""):
        super().__init__()
        self.setFrameStyle(QFrame.Shape.Box)
        self.setStyleSheet("""
            QFrame {
                background-color: #2b2b2b;
                border: 1px solid #404040;
                border-radius: 8px;
                padding: 8px;
                margin: 4px;
            }
        """)
        
        layout = QVBoxLayout()
        layout.setSpacing(4)
        
        # Title
        self.title_label = QLabel(title)
        self.title_label.setFont(QFont("Arial", 12))
        self.title_label.setStyleSheet("color: #cccccc; font-weight: bold;")
        
        # Value
        self.value_label = QLabel(f"{value} {unit}")
        self.value_label.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        self.value_label.setStyleSheet("color: #00ff88;")
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        layout.addWidget(self.title_label)
        layout.addWidget(self.value_label)
        self.setLayout(layout)
        
    def update_value(self, value: str, unit: str = ""):
        """Update the metric value"""
        self.value_label.setText(f"{value} {unit}")


class GaitMetricsPanel(QGroupBox):
    """Panel containing all gait metrics"""
    
    def __init__(self):
        super().__init__("Thông số dáng đi")
        self.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #404040;
                border-radius: 8px;
                margin-top: 1ex;
                color: #ffffff;
                background-color: #1e1e1e;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the UI layout"""
        layout = QVBoxLayout()
        layout.setSpacing(8)
        
        # Create metric cards
        self.metrics = {
            "step_count": MetricCard("Số bước", "0", "bước"),
            "cadence": MetricCard("Tần số bước", "0.0", "bước/phút"),
            "stride_length": MetricCard("Độ dài bước", "0.0", "px"),
            "step_time": MetricCard("Thời gian bước", "0.00", "giây"),
            "walking_speed": MetricCard("Tốc độ đi", "0.0", "px/s"),
            "foot_angle_left": MetricCard("Góc chân trái", "0.0", "°"),
            "foot_angle_right": MetricCard("Góc chân phải", "0.0", "°"),
        }
        
        # Add all metric cards to layout
        for metric_card in self.metrics.values():
            layout.addWidget(metric_card)
        
        # Add stretch to push everything to top
        layout.addStretch()
        
        self.setLayout(layout)
    
    def update_metrics(self, metrics_dict):
        """Update all metrics with new values"""
        metric_mapping = {
            "Số bước": ("step_count", "bước"),
            "Tần số bước (bước/phút)": ("cadence", "bước/phút"),
            "Độ dài bước (px)": ("stride_length", "px"),
            "Thời gian bước (s)": ("step_time", "giây"),
            "Tốc độ đi (px/s)": ("walking_speed", "px/s"),
            "Góc chân trái (°)": ("foot_angle_left", "°"),
            "Góc chân phải (°)": ("foot_angle_right", "°"),
        }
        
        for display_name, value in metrics_dict.items():
            if display_name in metric_mapping:
                metric_key, unit = metric_mapping[display_name]
                if metric_key in self.metrics:
                    self.metrics[metric_key].update_value(str(value), unit)


class ControlPanel(QGroupBox):
    """Control panel for camera and analysis controls"""
    
    connect_camera = pyqtSignal()
    disconnect_camera = pyqtSignal()
    start_analysis = pyqtSignal()
    stop_analysis = pyqtSignal()
    reset_analysis = pyqtSignal()
    
    def __init__(self):
        super().__init__("Điều khiển")
        self.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #404040;
                border-radius: 8px;
                margin-top: 1ex;
                color: #ffffff;
                background-color: #1e1e1e;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        
        self.is_camera_connected = False
        self.is_analysis_running = False
        self.setup_ui()
        
    def setup_ui(self):
        """Setup control buttons"""
        layout = QVBoxLayout()
        layout.setSpacing(8)
        
        # Camera control buttons
        camera_group = QLabel("📹 Camera")
        camera_group.setStyleSheet("color: #00ff88; font-weight: bold; margin-bottom: 5px;")
        
        # Connect/Disconnect camera button
        self.camera_btn = QPushButton("Kết nối Camera")
        self.camera_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 6px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #1565C0;
            }
        """)
        self.camera_btn.clicked.connect(self.toggle_camera)
        
        # Analysis control buttons
        analysis_group = QLabel("🔬 Phân tích")
        analysis_group.setStyleSheet("color: #00ff88; font-weight: bold; margin-top: 10px; margin-bottom: 5px;")
        
        # Start/Stop analysis button
        self.analysis_btn = QPushButton("Bắt đầu phân tích")
        self.analysis_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 6px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
        """)
        self.analysis_btn.clicked.connect(self.toggle_analysis)
        self.analysis_btn.setEnabled(False)  # Disabled until camera connected
        
        # Reset button
        self.reset_btn = QPushButton("Đặt lại")
        self.reset_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 6px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
            QPushButton:pressed {
                background-color: #b71c1c;
            }
        """)
        self.reset_btn.clicked.connect(self.reset_analysis_clicked)
        
        layout.addWidget(camera_group)
        layout.addWidget(self.camera_btn)
        layout.addWidget(analysis_group)
        layout.addWidget(self.analysis_btn)
        layout.addWidget(self.reset_btn)
        layout.addStretch()
        
        self.setLayout(layout)
    
    def toggle_camera(self):
        """Toggle camera connection"""
        if self.is_camera_connected:
            self.disconnect_camera.emit()
        else:
            self.connect_camera.emit()
    
    def toggle_analysis(self):
        """Toggle between start and stop analysis"""
        if self.is_analysis_running:
            self.stop_analysis.emit()
        else:
            self.start_analysis.emit()
    
    def set_camera_connected(self, connected: bool):
        """Update camera connection state"""
        self.is_camera_connected = connected
        if connected:
            self.camera_btn.setText("Ngắt kết nối Camera")
            self.camera_btn.setStyleSheet("""
                QPushButton {
                    background-color: #ff9800;
                    color: white;
                    border: none;
                    padding: 10px;
                    border-radius: 6px;
                    font-size: 12px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #e68900;
                }
                QPushButton:pressed {
                    background-color: #cc7700;
                }
            """)
            self.analysis_btn.setEnabled(True)
        else:
            self.camera_btn.setText("Kết nối Camera")
            self.camera_btn.setStyleSheet("""
                QPushButton {
                    background-color: #2196F3;
                    color: white;
                    border: none;
                    padding: 10px;
                    border-radius: 6px;
                    font-size: 12px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #1976D2;
                }
                QPushButton:pressed {
                    background-color: #1565C0;
                }
            """)
            self.analysis_btn.setEnabled(False)
            # Also stop analysis if camera disconnected
            if self.is_analysis_running:
                self.set_analysis_running(False)
    
    def set_analysis_running(self, running: bool):
        """Update analysis running state"""
        self.is_analysis_running = running
        if running:
            self.analysis_btn.setText("Dừng phân tích")
            self.analysis_btn.setStyleSheet("""
                QPushButton {
                    background-color: #ff9800;
                    color: white;
                    border: none;
                    padding: 10px;
                    border-radius: 6px;
                    font-size: 12px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #e68900;
                }
                QPushButton:pressed {
                    background-color: #cc7700;
                }
            """)
        else:
            self.analysis_btn.setText("Bắt đầu phân tích")
            self.analysis_btn.setStyleSheet("""
                QPushButton {
                    background-color: #4CAF50;
                    color: white;
                    border: none;
                    padding: 10px;
                    border-radius: 6px;
                    font-size: 12px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #45a049;
                }
                QPushButton:pressed {
                    background-color: #3d8b40;
                }
            """)
    
    def reset_analysis_clicked(self):
        """Handle reset button click"""
        self.reset_analysis.emit()


class Sidebar(QWidget):
    """Main sidebar widget containing all panels"""
    
    connect_camera = pyqtSignal()
    disconnect_camera = pyqtSignal()
    start_analysis = pyqtSignal()
    stop_analysis = pyqtSignal()
    reset_analysis = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.setFixedWidth(350)
        self.setStyleSheet("""
            QWidget {
                background-color: #1a1a1a;
                color: #ffffff;
            }
        """)
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup sidebar layout"""
        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Title
        title = QLabel("GaitSenseAI")
        title.setFont(QFont("Arial", 22, QFont.Weight.Bold))
        title.setStyleSheet("color: #00ff88; margin-bottom: 10px;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Control panel
        self.control_panel = ControlPanel()
        self.control_panel.connect_camera.connect(self.connect_camera)
        self.control_panel.disconnect_camera.connect(self.disconnect_camera)
        self.control_panel.start_analysis.connect(self.start_analysis)
        self.control_panel.stop_analysis.connect(self.stop_analysis)
        self.control_panel.reset_analysis.connect(self.reset_analysis)
        
        # Metrics panel
        self.metrics_panel = GaitMetricsPanel()
        
        # Create scroll area for metrics
        scroll = QScrollArea()
        scroll.setWidget(self.metrics_panel)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: #1a1a1a;
            }
        """)
        
        layout.addWidget(title)
        layout.addWidget(self.control_panel)
        layout.addWidget(scroll, 1)  # Give scroll area most of the space
        
        self.setLayout(layout)
    
    def update_metrics(self, metrics_dict):
        """Update metrics display"""
        self.metrics_panel.update_metrics(metrics_dict)
    
    def set_camera_connected(self, connected: bool):
        """Update camera connection state in control panel"""
        self.control_panel.set_camera_connected(connected)
    
    def set_analysis_running(self, running: bool):
        """Update analysis running state in control panel"""
        self.control_panel.set_analysis_running(running)

