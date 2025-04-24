"""
Gait Analysis View Module
-----------------------
This module provides a widget for displaying gait analysis results.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QTabWidget, QFrame, QSplitter, QGridLayout,
    QPushButton, QProgressBar
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QFont, QColor

import numpy as np
from typing import Optional, Dict, List, Any

try:
    import matplotlib
    matplotlib.use('QtAgg')  # Use QtAgg instead of Qt5Agg for PyQt6
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class MatplotlibCanvas(FigureCanvas):
    """Matplotlib canvas for plotting."""

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        """
        Initialize the matplotlib canvas.
        """
        if not MATPLOTLIB_AVAILABLE:
            return

        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)

        super().__init__(self.fig)
        self.setParent(parent)

        # Set background color
        self.fig.patch.set_facecolor('#2c3e50')
        self.axes.set_facecolor('#2c3e50')

        # Set text color to white
        self.axes.tick_params(colors='white')
        self.axes.xaxis.label.set_color('white')
        self.axes.yaxis.label.set_color('white')
        self.axes.title.set_color('white')

        # Set grid color
        self.axes.grid(True, color='gray', alpha=0.3)

        # Set tight layout
        self.fig.tight_layout()


class SkeletonVisualizationWidget(QWidget):
    """Widget for visualizing skeleton."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(300, 300)
        self.setStyleSheet("background-color: #1e272e; border-radius: 5px;")

        layout = QVBoxLayout()

        label = QLabel("Skeleton Visualization")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet("color: white; font-size: 14px;")

        layout.addWidget(label)
        layout.addStretch()

        self.setLayout(layout)


class JointAngleWidget(QWidget):
    """Widget for displaying joint angles."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background-color: transparent;")

        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(2)

        # Title
        title = QLabel("Joint Angles (degrees)")
        title.setStyleSheet("color: white; font-size: 14px; font-weight: bold;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Joint angles grid
        angles_grid = QGridLayout()
        angles_grid.setSpacing(5)  # Reduced spacing between rows
        angles_grid.setContentsMargins(5, 5, 5, 5)

        # Left side
        left_hip_label = QLabel("Left Hip:")
        left_hip_label.setStyleSheet("color: white;")
        self.left_hip_value = QLabel("55.4°")
        self.left_hip_value.setStyleSheet("color: #3498db; font-weight: bold;")

        left_knee_label = QLabel("Left Knee:")
        left_knee_label.setStyleSheet("color: white;")
        self.left_knee_value = QLabel("34.7°")
        self.left_knee_value.setStyleSheet("color: #3498db; font-weight: bold;")

        left_ankle_label = QLabel("Left Ankle:")
        left_ankle_label.setStyleSheet("color: white;")
        self.left_ankle_value = QLabel("79.4°")
        self.left_ankle_value.setStyleSheet("color: #3498db; font-weight: bold;")

        left_shoulder_label = QLabel("Left Shoulder:")
        left_shoulder_label.setStyleSheet("color: white;")
        self.left_shoulder_value = QLabel("3.5°")
        self.left_shoulder_value.setStyleSheet("color: #3498db; font-weight: bold;")

        left_elbow_label = QLabel("Left Elbow:")
        left_elbow_label.setStyleSheet("color: white;")
        self.left_elbow_value = QLabel("2.3°")
        self.left_elbow_value.setStyleSheet("color: #3498db; font-weight: bold;")

        # Right side
        right_hip_label = QLabel("Right Hip:")
        right_hip_label.setStyleSheet("color: white;")
        self.right_hip_value = QLabel("33.5°")
        self.right_hip_value.setStyleSheet("color: #e74c3c; font-weight: bold;")

        right_knee_label = QLabel("Right Knee:")
        right_knee_label.setStyleSheet("color: white;")
        self.right_knee_value = QLabel("54.0°")
        self.right_knee_value.setStyleSheet("color: #e74c3c; font-weight: bold;")

        right_ankle_label = QLabel("Right Ankle:")
        right_ankle_label.setStyleSheet("color: white;")
        self.right_ankle_value = QLabel("46.0°")
        self.right_ankle_value.setStyleSheet("color: #e74c3c; font-weight: bold;")

        right_shoulder_label = QLabel("Right Shoulder:")
        right_shoulder_label.setStyleSheet("color: white;")
        self.right_shoulder_value = QLabel("6.8°")
        self.right_shoulder_value.setStyleSheet("color: #e74c3c; font-weight: bold;")

        right_elbow_label = QLabel("Right Elbow:")
        right_elbow_label.setStyleSheet("color: white;")
        self.right_elbow_value = QLabel("56.5°")
        self.right_elbow_value.setStyleSheet("color: #e74c3c; font-weight: bold;")

        # Add to grid
        angles_grid.addWidget(left_hip_label, 0, 0)
        angles_grid.addWidget(self.left_hip_value, 0, 1)
        angles_grid.addWidget(right_hip_label, 0, 2)
        angles_grid.addWidget(self.right_hip_value, 0, 3)

        angles_grid.addWidget(left_knee_label, 1, 0)
        angles_grid.addWidget(self.left_knee_value, 1, 1)
        angles_grid.addWidget(right_knee_label, 1, 2)
        angles_grid.addWidget(self.right_knee_value, 1, 3)

        angles_grid.addWidget(left_ankle_label, 2, 0)
        angles_grid.addWidget(self.left_ankle_value, 2, 1)
        angles_grid.addWidget(right_ankle_label, 2, 2)
        angles_grid.addWidget(self.right_ankle_value, 2, 3)

        angles_grid.addWidget(left_shoulder_label, 3, 0)
        angles_grid.addWidget(self.left_shoulder_value, 3, 1)
        angles_grid.addWidget(right_shoulder_label, 3, 2)
        angles_grid.addWidget(self.right_shoulder_value, 3, 3)

        angles_grid.addWidget(left_elbow_label, 4, 0)
        angles_grid.addWidget(self.left_elbow_value, 4, 1)
        angles_grid.addWidget(right_elbow_label, 4, 2)
        angles_grid.addWidget(self.right_elbow_value, 4, 3)

        # Set row and column stretches
        for i in range(5):
            angles_grid.setRowStretch(i, 1)

        angles_grid.setColumnStretch(0, 2)
        angles_grid.setColumnStretch(1, 1)
        angles_grid.setColumnStretch(2, 2)
        angles_grid.setColumnStretch(3, 1)

        layout.addLayout(angles_grid)
        self.setLayout(layout)

    def update_angles(self, angles: Dict[str, float]):
        """Update joint angles."""
        if 'left_hip' in angles:
            self.left_hip_value.setText(f"{angles['left_hip']:.1f}°")
        if 'right_hip' in angles:
            self.right_hip_value.setText(f"{angles['right_hip']:.1f}°")
        if 'left_knee' in angles:
            self.left_knee_value.setText(f"{angles['left_knee']:.1f}°")
        if 'right_knee' in angles:
            self.right_knee_value.setText(f"{angles['right_knee']:.1f}°")
        if 'left_ankle' in angles:
            self.left_ankle_value.setText(f"{angles['left_ankle']:.1f}°")
        if 'right_ankle' in angles:
            self.right_ankle_value.setText(f"{angles['right_ankle']:.1f}°")
        if 'left_shoulder' in angles:
            self.left_shoulder_value.setText(f"{angles['left_shoulder']:.1f}°")
        if 'right_shoulder' in angles:
            self.right_shoulder_value.setText(f"{angles['right_shoulder']:.1f}°")
        if 'left_elbow' in angles:
            self.left_elbow_value.setText(f"{angles['left_elbow']:.1f}°")
        if 'right_elbow' in angles:
            self.right_elbow_value.setText(f"{angles['right_elbow']:.1f}°")


class GaitParametersWidget(QWidget):
    """Widget for displaying gait parameters."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background-color: transparent;")

        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(2)

        # Title
        title = QLabel("Gait Parameters")
        title.setStyleSheet("color: white; font-size: 14px; font-weight: bold;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Parameters grid
        params_grid = QGridLayout()
        params_grid.setSpacing(5)  # Reduced spacing between rows
        params_grid.setContentsMargins(5, 5, 5, 5)

        stride_length_label = QLabel("Stride Length:")
        stride_length_label.setStyleSheet("color: white;")
        self.stride_length_value = QLabel("126.8 cm")
        self.stride_length_value.setStyleSheet("color: #2ecc71; font-weight: bold;")

        step_width_label = QLabel("Step Width:")
        step_width_label.setStyleSheet("color: white;")
        self.step_width_value = QLabel("8.0 cm")
        self.step_width_value.setStyleSheet("color: #2ecc71; font-weight: bold;")

        cadence_label = QLabel("Cadence:")
        cadence_label.setStyleSheet("color: white;")
        self.cadence_value = QLabel("105.1 steps/min")
        self.cadence_value.setStyleSheet("color: #2ecc71; font-weight: bold;")

        walking_speed_label = QLabel("Walking Speed:")
        walking_speed_label.setStyleSheet("color: white;")
        self.walking_speed_value = QLabel("0.94 m/s")
        self.walking_speed_value.setStyleSheet("color: #2ecc71; font-weight: bold;")

        stance_phase_label = QLabel("Stance Phase:")
        stance_phase_label.setStyleSheet("color: white;")
        self.stance_phase_value = QLabel("64.8 %")
        self.stance_phase_value.setStyleSheet("color: #2ecc71; font-weight: bold;")

        # Add to grid
        params_grid.addWidget(stride_length_label, 0, 0)
        params_grid.addWidget(self.stride_length_value, 0, 1)

        params_grid.addWidget(step_width_label, 1, 0)
        params_grid.addWidget(self.step_width_value, 1, 1)

        params_grid.addWidget(cadence_label, 2, 0)
        params_grid.addWidget(self.cadence_value, 2, 1)

        params_grid.addWidget(walking_speed_label, 3, 0)
        params_grid.addWidget(self.walking_speed_value, 3, 1)

        params_grid.addWidget(stance_phase_label, 4, 0)
        params_grid.addWidget(self.stance_phase_value, 4, 1)

        # Progress bar for visualization
        progress_layout = QHBoxLayout()
        progress_label = QLabel("0")
        progress_label.setStyleSheet("color: white;")
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #3498db;
                border-radius: 5px;
                background-color: #2c3e50;
                text-align: center;
                color: transparent;
            }
            QProgressBar::chunk {
                background-color: #3498db;
            }
        """)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(50)
        self.progress_bar.setFixedHeight(10)
        progress_layout.addWidget(progress_label)
        progress_layout.addWidget(self.progress_bar)

        # Set row and column stretches
        for i in range(5):
            params_grid.setRowStretch(i, 1)

        params_grid.setColumnStretch(0, 3)
        params_grid.setColumnStretch(1, 1)

        layout.addLayout(params_grid)
        layout.addLayout(progress_layout)
        self.setLayout(layout)

    def update_parameters(self, params: Dict[str, float]):
        """Update gait parameters."""
        if 'stride_length' in params:
            self.stride_length_value.setText(f"{params['stride_length']:.1f} cm")
        if 'step_width' in params:
            self.step_width_value.setText(f"{params['step_width']:.1f} cm")
        if 'cadence' in params:
            self.cadence_value.setText(f"{params['cadence']:.1f} steps/min")
        if 'walking_speed' in params:
            self.walking_speed_value.setText(f"{params['walking_speed']:.2f} m/s")
        if 'stance_phase' in params:
            self.stance_phase_value.setText(f"{params['stance_phase']:.1f} %")


class GaitChartWidget(QWidget):
    """Widget for displaying gait analysis charts."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background-color: transparent;")

        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)

        # # Title
        # title = QLabel("Gait Analysis Chart")
        # title.setStyleSheet("color: white; font-size: 14px; font-weight: bold;")
        # title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # layout.addWidget(title)

        if MATPLOTLIB_AVAILABLE:
            # Create matplotlib canvas for gait analysis chart
            self.chart_canvas = MatplotlibCanvas(self, width=6, height=8)  # Increased width and height

            # Adjust figure to give more space for title and labels
            self.chart_canvas.fig.subplots_adjust(top=0.9, bottom=0.12, left=0.12, right=0.95)

            self.chart_canvas.axes.set_title("Gait Cycle Analysis", color='white', fontsize=14, pad=10)
            self.chart_canvas.axes.set_xlabel("Time (s)", color='white', fontsize=12, labelpad=8)
            self.chart_canvas.axes.set_ylabel("Joint Angle (degrees)", color='white', fontsize=12, labelpad=8)

            # Generate sample data
            time = np.linspace(0, 10, 100)
            left_hip = np.sin(time) * 10 + 55
            right_hip = np.sin(time + np.pi/3) * 10 + 33
            left_knee = np.sin(time + np.pi/6) * 15 + 35
            right_knee = np.sin(time + np.pi/2) * 15 + 54

            # Plot data
            self.chart_canvas.axes.plot(time, left_hip, 'b-', label='Left Hip', linewidth=2)
            self.chart_canvas.axes.plot(time, right_hip, 'r-', label='Right Hip', linewidth=2)
            self.chart_canvas.axes.plot(time, left_knee, 'b--', label='Left Knee', linewidth=2)
            self.chart_canvas.axes.plot(time, right_knee, 'r--', label='Right Knee', linewidth=2)

            # Set y-axis limits to match the image
            self.chart_canvas.axes.set_ylim(20, 70)

            # Customize the chart appearance
            self.chart_canvas.axes.legend(loc='upper right', fontsize=10)
            self.chart_canvas.axes.grid(True, color='gray', alpha=0.3)

            # Set background color
            self.chart_canvas.fig.patch.set_facecolor('#2c3e50')
            self.chart_canvas.axes.set_facecolor('#2c3e50')

            # Set tick colors
            self.chart_canvas.axes.tick_params(axis='x', colors='white', labelsize=10)
            self.chart_canvas.axes.tick_params(axis='y', colors='white', labelsize=10)

            # Set spine colors
            for spine in self.chart_canvas.axes.spines.values():
                spine.set_color('white')

            layout.addWidget(self.chart_canvas)
        else:
            # Fallback if matplotlib is not available
            fallback_label = QLabel("Chart visualization requires matplotlib")
            fallback_label.setStyleSheet("color: white; font-style: italic;")
            fallback_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(fallback_label)

        self.setLayout(layout)

    def update_chart(self, data: Dict[str, List[float]]) -> None:
        """Update chart with new data."""
        if not MATPLOTLIB_AVAILABLE or not hasattr(self, 'chart_canvas'):
            return

        self.chart_canvas.axes.clear()

        # Adjust figure to give more space for title and labels
        self.chart_canvas.fig.subplots_adjust(top=0.9, bottom=0.12, left=0.12, right=0.95)

        self.chart_canvas.axes.set_title("Gait Cycle Analysis", color='white', fontsize=14, pad=10)
        self.chart_canvas.axes.set_xlabel("Time (s)", color='white', fontsize=12, labelpad=8)
        self.chart_canvas.axes.set_ylabel("Joint Angle (degrees)", color='white', fontsize=12, labelpad=8)

        if 'time' in data and 'left_hip' in data:
            self.chart_canvas.axes.plot(data['time'], data['left_hip'], 'b-', label='Left Hip', linewidth=2)
        if 'time' in data and 'right_hip' in data:
            self.chart_canvas.axes.plot(data['time'], data['right_hip'], 'r-', label='Right Hip', linewidth=2)
        if 'time' in data and 'left_knee' in data:
            self.chart_canvas.axes.plot(data['time'], data['left_knee'], 'b--', label='Left Knee', linewidth=2)
        if 'time' in data and 'right_knee' in data:
            self.chart_canvas.axes.plot(data['time'], data['right_knee'], 'r--', label='Right Knee', linewidth=2)

        # Set y-axis limits to match the image
        self.chart_canvas.axes.set_ylim(20, 70)

        # Customize the chart appearance
        self.chart_canvas.axes.legend(loc='upper right', fontsize=10)
        self.chart_canvas.axes.grid(True, color='gray', alpha=0.3)

        # Set background color
        self.chart_canvas.fig.patch.set_facecolor('#2c3e50')
        self.chart_canvas.axes.set_facecolor('#2c3e50')

        # Set tick colors
        self.chart_canvas.axes.tick_params(axis='x', colors='white', labelsize=10)
        self.chart_canvas.axes.tick_params(axis='y', colors='white', labelsize=10)

        # Set spine colors
        for spine in self.chart_canvas.axes.spines.values():
            spine.set_color('white')

        self.chart_canvas.draw()


class GaitAnalysisView(QWidget):
    """Widget for displaying gait analysis results."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        # Initialize data
        self.joint_angles = {
            'left_hip': 55.2,
            'right_hip': 33.5,
            'left_knee': 34.7,
            'right_knee': 54.1,
            'left_ankle': 79.5,
            'right_ankle': 46.2,
            'left_shoulder': 3.7,
            'right_shoulder': 6.8,
            'left_elbow': 2.6,
            'right_elbow': 56.5
        }

        self.gait_parameters = {
            'stride_length': 126.8,
            'step_width': 8.1,
            'cadence': 105.1,
            'walking_speed': 0.95,
            'stance_phase': 64.9
        }

        self.chart_data = {
            'time': np.linspace(0, 10, 100),
            'left_hip': np.sin(np.linspace(0, 10, 100)) * 10 + 55,
            'right_hip': np.sin(np.linspace(0, 10, 100) + np.pi/3) * 10 + 33,
            'left_knee': np.sin(np.linspace(0, 10, 100) + np.pi/6) * 15 + 35,
            'right_knee': np.sin(np.linspace(0, 10, 100) + np.pi/2) * 15 + 54
        }

        # Create UI
        self.init_ui()

        # Create timer for updating data
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_data)
        self.timer.start(1000)  # Update every second

    def init_ui(self) -> None:
        # Main layout
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        # Keep the current size of the widget
        self.setStyleSheet("background-color: #2c3e50;")

        # Control buttons
        control_layout = QHBoxLayout()
        control_layout.setContentsMargins(5, 5, 5, 5)
        control_layout.setSpacing(5)

        self.start_analysis_btn = QPushButton("Start Analysis")
        self.start_analysis_btn.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 8px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2ecc71;
            }
        """)

        self.stop_analysis_btn = QPushButton("Stop Analysis")
        self.stop_analysis_btn.setStyleSheet("""
            QPushButton {
                background-color: #c0392b;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 8px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #e74c3c;
            }
        """)

        self.save_data_btn = QPushButton("Save Data")
        self.save_data_btn.setStyleSheet("""
            QPushButton {
                background-color: #2980b9;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 8px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #3498db;
            }
        """)

        control_layout.addWidget(self.start_analysis_btn)
        control_layout.addWidget(self.stop_analysis_btn)
        control_layout.addWidget(self.save_data_btn)

        layout.addLayout(control_layout)

        # Create a splitter for the top and bottom sections
        main_splitter = QSplitter(Qt.Orientation.Vertical)

        # Top section with parameters
        top_widget = QWidget()
        top_layout = QVBoxLayout(top_widget)
        top_layout.setContentsMargins(5, 5, 5, 5)
        top_layout.setSpacing(0)  # Reduced spacing

        # Joint angles
        self.joint_angles_widget = JointAngleWidget()
        top_layout.addWidget(self.joint_angles_widget)

        # Gait parameters
        self.gait_params_widget = GaitParametersWidget()
        top_layout.addWidget(self.gait_params_widget)

        # Bottom section with chart
        bottom_widget = QWidget()
        bottom_layout = QVBoxLayout(bottom_widget)
        bottom_layout.setContentsMargins(5, 5, 5, 5)

        # Gait analysis chart
        self.chart_widget = GaitChartWidget()
        bottom_layout.addWidget(self.chart_widget)

        # Add widgets to splitter
        main_splitter.addWidget(top_widget)
        main_splitter.addWidget(bottom_widget)

        # Set initial sizes - give more space to the chart
        main_splitter.setSizes([250, 550])

        # Add splitter to main layout
        layout.addWidget(main_splitter)

        # Set layout
        self.setLayout(layout)

        # Update widgets with initial data
        self.joint_angles_widget.update_angles(self.joint_angles)
        self.gait_params_widget.update_parameters(self.gait_parameters)
        self.chart_widget.update_chart(self.chart_data)

    def update_data(self) -> None:
        """Update data with simulated values."""
        # Simulate changing joint angles
        for key in self.joint_angles:
            # Add small random variation
            self.joint_angles[key] += np.random.uniform(-0.1, 0.1)

        # Simulate changing gait parameters
        for key in self.gait_parameters:
            # Add small random variation
            if key == 'walking_speed':
                self.gait_parameters[key] += np.random.uniform(-0.01, 0.01)
            else:
                self.gait_parameters[key] += np.random.uniform(-0.1, 0.1)

        # Update chart data
        phase_shift = (self.timer.interval() / 1000) % (2 * np.pi)
        self.chart_data = {
            'time': np.linspace(0, 10, 100),
            'left_hip': np.sin(np.linspace(0, 10, 100) + phase_shift) * 10 + 55,
            'right_hip': np.sin(np.linspace(0, 10, 100) + phase_shift + np.pi/3) * 10 + 33,
            'left_knee': np.sin(np.linspace(0, 10, 100) + phase_shift + np.pi/6) * 15 + 35,
            'right_knee': np.sin(np.linspace(0, 10, 100) + phase_shift + np.pi/2) * 15 + 54
        }

        # Update widgets
        self.joint_angles_widget.update_angles(self.joint_angles)
        self.gait_params_widget.update_parameters(self.gait_parameters)
        self.chart_widget.update_chart(self.chart_data)

        # Update progress bar
        if hasattr(self.gait_params_widget, 'progress_bar'):
            current_value = self.gait_params_widget.progress_bar.value()
            new_value = (current_value + 1) % 100
            self.gait_params_widget.progress_bar.setValue(new_value)

    def update_joint_angles(self, joint_angles: Dict[str, float]) -> None:
        """Update joint angle data."""
        self.joint_angles.update(joint_angles)
        self.joint_angles_widget.update_angles(self.joint_angles)

    def update_gait_parameters(self, gait_parameters: Dict[str, float]) -> None:
        """Update gait parameter data."""
        self.gait_parameters.update(gait_parameters)
        self.gait_params_widget.update_parameters(self.gait_parameters)

    def clear_data(self) -> None:
        """Clear all data."""
        self.joint_angles = {
            'left_hip': 0.0,
            'right_hip': 0.0,
            'left_knee': 0.0,
            'right_knee': 0.0,
            'left_ankle': 0.0,
            'right_ankle': 0.0,
            'left_shoulder': 0.0,
            'right_shoulder': 0.0,
            'left_elbow': 0.0,
            'right_elbow': 0.0
        }

        self.gait_parameters = {
            'stride_length': 0.0,
            'step_width': 0.0,
            'cadence': 0.0,
            'walking_speed': 0.0,
            'stance_phase': 0.0
        }

        # Update widgets
        self.joint_angles_widget.update_angles(self.joint_angles)
        self.gait_params_widget.update_parameters(self.gait_parameters)
