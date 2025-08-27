"""
RealSense Style Control Panel - Professional Camera Interface
"""

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QFrame, QScrollArea, QGroupBox,
                             QComboBox, QCheckBox, QSlider, QSpinBox,
                             QTabWidget, QSplitter, QButtonGroup, QRadioButton,
                             QLineEdit, QListWidget, QListWidgetItem)
from PyQt6.QtCore import Qt, pyqtSignal, QSize
from PyQt6.QtGui import QFont, QIcon, QPalette


class ToggleButton(QPushButton):
    """Custom toggle button like RealSense"""
    
    def __init__(self, text=""):
        super().__init__(text)
        self.setCheckable(True)
        self.setFixedHeight(30)
        self.update_style()
        self.toggled.connect(self.update_style)
    
    def update_style(self):
        if self.isChecked():
            self.setStyleSheet("""
                QPushButton {
                    background-color: #0078d4;
                    color: white;
                    border: none;
                    border-radius: 4px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #106ebe;
                }
            """)
        else:
            self.setStyleSheet("""
                QPushButton {
                    background-color: #e0e0e0;
                    color: #333333;
                    border: 1px solid #cccccc;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #d0d0d0;
                }
            """)


class DropdownControl(QFrame):
    """RealSense style dropdown with label"""
    
    def __init__(self, label_text, items, current_index=0):
        super().__init__()
        self.setStyleSheet("""
            QFrame {
                background-color: transparent;
                border: none;
            }
        """)
        
        layout = QVBoxLayout()
        layout.setSpacing(4)
        layout.setContentsMargins(0, 0, 0, 8)
        
        # Label
        label = QLabel(label_text)
        label.setFont(QFont("Segoe UI", 12))
        label.setStyleSheet("color: #cccccc;")
        
        # Dropdown
        self.combo = QComboBox()
        self.combo.addItems(items)
        self.combo.setCurrentIndex(current_index)
        self.combo.setFixedHeight(25)
        self.combo.setStyleSheet("""
            QComboBox {
                background-color: #ffffff;
                border: 1px solid #cccccc;
                border-radius: 3px;
                padding: 4px 8px;
                color: #000000;
                font-size: 14px;
            }
            QComboBox:hover {
                border-color: #0078d4;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid #333333;
                margin-right: 5px;
            }
            QComboBox QAbstractItemView {
                background-color: #ffffff;
                border: 1px solid #cccccc;
                selection-background-color: #0078d4;
                color: #000000;
            }
        """)
        
        layout.addWidget(label)
        layout.addWidget(self.combo)
        self.setLayout(layout)


class SliderControl(QFrame):
    """RealSense style slider with value display"""
    
    def __init__(self, label_text, min_val=0, max_val=100, current_val=50):
        super().__init__()
        self.setStyleSheet("QFrame { background-color: transparent; border: none; }")
        
        layout = QVBoxLayout()
        layout.setSpacing(4)
        layout.setContentsMargins(0, 0, 0, 8)
        
        # Header with label and value
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        
        label = QLabel(label_text)
        label.setFont(QFont("Segoe UI", 12))
        label.setStyleSheet("color: #cccccc;")
        
        self.value_label = QLabel(str(current_val))
        self.value_label.setFont(QFont("Segoe UI", 12))
        self.value_label.setStyleSheet("color: #ffffff;")
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        
        header_layout.addWidget(label)
        header_layout.addWidget(self.value_label)
        
        # Slider
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(min_val)
        self.slider.setMaximum(max_val)
        self.slider.setValue(current_val)
        self.slider.setFixedHeight(20)
        self.slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #555555;
                height: 6px;
                background: #3c3c3c;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #0078d4;
                border: 1px solid #0078d4;
                width: 14px;
                height: 14px;
                border-radius: 7px;
                margin: -5px 0;
            }
            QSlider::handle:horizontal:hover {
                background: #106ebe;
            }
            QSlider::sub-page:horizontal {
                background: #0078d4;
                border-radius: 3px;
            }
        """)
        
        self.slider.valueChanged.connect(lambda v: self.value_label.setText(str(v)))
        
        layout.addLayout(header_layout)
        layout.addWidget(self.slider)
        self.setLayout(layout)


class CollapsibleSection(QFrame):
    """Collapsible section like RealSense modules"""
    
    def __init__(self, title, content_widget=None):
        super().__init__()
        self.setStyleSheet("""
            QFrame {
                background-color: #ffffff;
                border: 1px solid #dddddd;
                border-radius: 6px;
                margin: 2px 0px;
            }
        """)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Header
        self.header = QPushButton(f"▼ {title}")
        self.header.setCheckable(True)
        self.header.setChecked(True)
        self.header.setFixedHeight(35)
        self.header.setStyleSheet("""
            QPushButton {
                background-color: #f0f0f0;
                color: #000000;
                border: none;
                border-radius: 6px 6px 0px 0px;
                text-align: left;
                padding-left: 12px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
            }
        """)
        self.header.toggled.connect(self.toggle_content)
        
        # Content container
        self.content_frame = QFrame()
        self.content_frame.setStyleSheet("QFrame { border: none; background-color: transparent; }")
        self.content_layout = QVBoxLayout()
        self.content_layout.setContentsMargins(12, 12, 12, 12)
        self.content_layout.setSpacing(8)
        self.content_frame.setLayout(self.content_layout)
        
        if content_widget:
            self.content_layout.addWidget(content_widget)
        
        layout.addWidget(self.header)
        layout.addWidget(self.content_frame)
        self.setLayout(layout)
    
    def toggle_content(self, checked):
        self.content_frame.setVisible(checked)
        arrow = "▼" if checked else "▶"
        text = self.header.text()
        title = text[2:]  # Remove arrow
        self.header.setText(f"{arrow} {title}")
    
    def add_control(self, widget):
        self.content_layout.addWidget(widget)
    
    def set_content(self, widget):
        """Set the main content widget for this section"""
        # Clear existing content first
        while self.content_layout.count():
            child = self.content_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        # Add new content
        if widget:
            self.content_layout.addWidget(widget)


class RealSenseSidebar(QWidget):
    """RealSense Viewer style control panel"""
    
    # Signals
    connect_camera = pyqtSignal()
    disconnect_camera = pyqtSignal()
    start_analysis = pyqtSignal()
    stop_analysis = pyqtSignal()
    reset_analysis = pyqtSignal()
    view_history_requested = pyqtSignal(str)  # Emit file path when history item is selected
    
    def __init__(self):
        super().__init__()
        self.camera_connected = False
        self.analysis_running = False
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the RealSense style interface"""
        self.setFixedWidth(420)
        self.setStyleSheet("""
            QWidget {
                background-color: #f8f8f8;
                color: #000000;
                font-family: 'Segoe UI', Arial;
            }
        """)
        
        # Main layout
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)
        
        # Header
        header = self.create_header()
        layout.addWidget(header)
        
        # Scroll area for controls
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QScrollBar:vertical {
                background-color: #f0f0f0;
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background-color: #cccccc;
                border-radius: 6px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #aaaaaa;
            }
        """)
        
        # Controls container
        controls_widget = QWidget()
        controls_layout = QVBoxLayout()
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(6)
        
        # Patient Information Module (replaces Camera Module)
        patient_section = self.create_patient_info_section()
        controls_layout.addWidget(patient_section)
        
        # Analysis History Module
        history_section = self.create_history_section()
        controls_layout.addWidget(history_section)
        
        # Camera Module - ẩn đi
        # camera_section = self.create_camera_section()
        # controls_layout.addWidget(camera_section)
        
        # Pose Detection Module - ẩn đi, dùng default settings
        # pose_section = self.create_pose_section()
        # controls_layout.addWidget(pose_section)
        
        # Gait Metrics Module
        metrics_section = self.create_metrics_section()
        controls_layout.addWidget(metrics_section)
        
        # Advanced Controls - ẩn đi, dùng default settings
        # advanced_section = self.create_advanced_section()
        # controls_layout.addWidget(advanced_section)
        
        controls_layout.addStretch()
        controls_widget.setLayout(controls_layout)
        scroll.setWidget(controls_widget)
        
        layout.addWidget(scroll)
        self.setLayout(layout)
    
    def create_header(self):
        """Create header with main controls"""
        header = QFrame()
        header.setFixedHeight(50)
        header.setStyleSheet("""
            QFrame {
                background-color: #ffffff;
                border: 1px solid #dddddd;
                border-radius: 8px;
            }
        """)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(12, 8, 12, 8)
        
        # Title - ẩn đi
        # title = QLabel("GaitSenseAI")
        # title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        # title.setStyleSheet("color: #0078d4; border: none;")
        
        # Control buttons
        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(8)
        
        self.camera_btn = ToggleButton("Camera")
        self.camera_btn.clicked.connect(self.toggle_camera)
        
        self.analysis_btn = ToggleButton("Analysis")
        self.analysis_btn.clicked.connect(self.toggle_analysis)
        self.analysis_btn.setEnabled(False)
        
        buttons_layout.addWidget(self.camera_btn)
        buttons_layout.addWidget(self.analysis_btn)
        
        # layout.addWidget(title)
        layout.addLayout(buttons_layout)
        header.setLayout(layout)
        
        return header
    
    def create_patient_info_section(self):
        """Create patient information input section"""
        section = CollapsibleSection("Thông Tin Bệnh Nhân")
        section.header.setStyleSheet(section.header.styleSheet() + "font-size: 16px;")
        
        # Main container frame
        info_frame = QFrame()
        info_frame.setStyleSheet("""
            QFrame {
                background-color: #ffffff;
                border: 1px solid #dddddd;
                border-radius: 6px;
                padding: 8px;
            }
        """)
        
        info_layout = QVBoxLayout()
        info_layout.setSpacing(8)
        info_layout.setContentsMargins(8, 8, 8, 8)
        
        # Patient Name input
        name_label = QLabel("Họ và tên:")
        name_label.setStyleSheet("color: #000000; font-weight: bold; font-size: 14px;")
        
        self.patient_name_input = QLineEdit()
        self.patient_name_input.setPlaceholderText("Nhập họ và tên bệnh nhân...")
        self.patient_name_input.setStyleSheet("""
            QLineEdit {
                background-color: #ffffff;
                border: 1px solid #cccccc;
                border-radius: 4px;
                padding: 8px 12px;
                font-size: 14px;
                color: #000000;
            }
            QLineEdit:focus {
                border-color: #0078d4;
                outline: none;
            }
        """)
        
        # Patient Age input
        age_label = QLabel("Tuổi:")
        age_label.setStyleSheet("color: #000000; font-weight: bold; font-size: 14px;")
        
        self.patient_age_input = QLineEdit()
        self.patient_age_input.setPlaceholderText("Nhập tuổi (1-120)")
        self.patient_age_input.setText("25")  # Default value
        
        # Set input validator for numbers only
        from PyQt6.QtGui import QIntValidator
        age_validator = QIntValidator(1, 120)
        self.patient_age_input.setValidator(age_validator)
        
        self.patient_age_input.setStyleSheet("""
            QLineEdit {
                background-color: #ffffff;
                border: 1px solid #cccccc;
                border-radius: 4px;
                padding: 8px 12px;
                font-size: 14px;
                color: #000000;
            }
            QLineEdit:focus {
                border-color: #0078d4;
                outline: none;
            }
        """)
        
        # Gender selection
        gender_label = QLabel("Giới tính:")
        gender_label.setStyleSheet("color: #000000; font-weight: bold; font-size: 14px;")
        
        gender_layout = QHBoxLayout()
        gender_layout.setSpacing(10)
        
        self.gender_male = QRadioButton("Nam")
        self.gender_female = QRadioButton("Nữ")
        self.gender_male.setChecked(True)  # Default selection
        
        for radio in [self.gender_male, self.gender_female]:
            radio.setStyleSheet("""
                QRadioButton {
                    color: #000000;
                    font-size: 14px;
                }
                QRadioButton::indicator {
                    width: 16px;
                    height: 16px;
                }
                QRadioButton::indicator:unchecked {
                    background-color: #ffffff;
                    border: 2px solid #cccccc;
                    border-radius: 8px;
                }
                QRadioButton::indicator:checked {
                    background-color: #0078d4;
                    border: 2px solid #0078d4;
                    border-radius: 8px;
                }
            """)
        
        gender_layout.addWidget(self.gender_male)
        gender_layout.addWidget(self.gender_female)
        gender_layout.addStretch()
        
        # Additional notes section removed
        
        # Add all components to layout
        info_layout.addWidget(name_label)
        info_layout.addWidget(self.patient_name_input)
        info_layout.addWidget(age_label)
        info_layout.addWidget(self.patient_age_input)
        info_layout.addWidget(gender_label)
        info_layout.addLayout(gender_layout)
        
        info_frame.setLayout(info_layout)
        section.add_control(info_frame)
        
        # Connect validation
        self.patient_name_input.textChanged.connect(self.validate_inputs)
        self.patient_age_input.textChanged.connect(self.validate_inputs)
        
        return section
    
    def validate_inputs(self):
        """Validate patient inputs and highlight errors"""
        # Reset styles first
        self.patient_name_input.setStyleSheet("""
            QLineEdit {
                background-color: #ffffff;
                border: 1px solid #cccccc;
                border-radius: 4px;
                padding: 8px 12px;
                font-size: 14px;
                color: #000000;
            }
            QLineEdit:focus {
                border-color: #0078d4;
                outline: none;
            }
        """)
        
        self.patient_age_input.setStyleSheet("""
            QLineEdit {
                background-color: #ffffff;
                border: 1px solid #cccccc;
                border-radius: 4px;
                padding: 8px 12px;
                font-size: 14px;
                color: #000000;
            }
            QLineEdit:focus {
                border-color: #0078d4;
                outline: none;
            }
        """)
        
        # Check name
        name = self.patient_name_input.text().strip()
        if not name or len(name) < 2:
            self.patient_name_input.setStyleSheet("""
                QLineEdit {
                    background-color: #fff5f5;
                    border: 2px solid #ff4444;
                    border-radius: 4px;
                    padding: 8px 12px;
                    font-size: 14px;
                    color: #000000;
                }
                QLineEdit:focus {
                    border-color: #ff2222;
                    outline: none;
                }
            """)
        
        # Check age
        try:
            age = int(self.patient_age_input.text().strip())
            if age < 1 or age > 120:
                self.patient_age_input.setStyleSheet("""
                    QLineEdit {
                        background-color: #fff5f5;
                        border: 2px solid #ff4444;
                        border-radius: 4px;
                        padding: 8px 12px;
                        font-size: 14px;
                        color: #000000;
                    }
                    QLineEdit:focus {
                        border-color: #ff2222;
                        outline: none;
                    }
                """)
        except (ValueError, TypeError):
            self.patient_age_input.setStyleSheet("""
                QLineEdit {
                    background-color: #fff5f5;
                    border: 2px solid #ff4444;
                    border-radius: 4px;
                    padding: 8px 12px;
                    font-size: 14px;
                    color: #000000;
                }
                QLineEdit:focus {
                    border-color: #ff2222;
                    outline: none;
                }
            """)
    
    def get_patient_info(self):
        """Get current patient information"""
        # Get age from text input, default to 25 if invalid
        try:
            age = int(self.patient_age_input.text().strip())
            if age < 1 or age > 120:
                age = 25  # Default if out of range
        except (ValueError, TypeError):
            age = 25  # Default if not a valid number
            
        return {
            "name": self.patient_name_input.text().strip(),
            "age": age,
            "gender": "Nam" if self.gender_male.isChecked() else "Nữ",
            "notes": ""  # Notes field removed
        }
    
    def create_history_section(self):
        """Create analysis history section"""
        from PyQt6.QtWidgets import QListWidget, QPushButton, QVBoxLayout
        from PyQt6.QtCore import pyqtSignal
        import os
        import glob
        from datetime import datetime
        
        section = CollapsibleSection("📊 Lịch Sử Phân Tích")
        # Tăng kích thước font cho section header
        section.header.setStyleSheet(section.header.styleSheet() + "font-size: 16px;")
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(8, 8, 8, 8)
        content_layout.setSpacing(8)
        
        # Refresh button
        refresh_btn = QPushButton("🔄 Làm mới danh sách")
        refresh_btn.setStyleSheet("""
            QPushButton {
                background-color: #e0e0e0;
                border: 1px solid #cccccc;
                border-radius: 4px;
                padding: 8px 16px;
                color: #333333;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #d0d0d0;
            }
            QPushButton:pressed {
                background-color: #c0c0c0;
            }
        """)
        content_layout.addWidget(refresh_btn)
        
        # History list widget
        self.history_list = QListWidget()
        self.history_list.setStyleSheet("""
            QListWidget {
                background-color: #ffffff;
                border: 1px solid #cccccc;
                border-radius: 4px;
                font-size: 16px;
                padding: 6px;
            }
            QListWidget::item {
                padding: 12px;
                border-bottom: 1px solid #eeeeee;
                color: #333333;
            }
            QListWidget::item:selected {
                background-color: #0078d4;
                color: white;
            }
            QListWidget::item:hover {
                background-color: #f0f0f0;
            }
        """)
        self.history_list.setMaximumHeight(200)
        content_layout.addWidget(self.history_list)
        
        # View result button
        view_btn = QPushButton("👁️ Xem kết quả")
        view_btn.setStyleSheet("""
            QPushButton {
                background-color: #0078d4;
                border: none;
                border-radius: 4px;
                padding: 10px 16px;
                color: white;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #106ebe;
            }
            QPushButton:pressed {
                background-color: #005a9e;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        view_btn.setEnabled(False)
        content_layout.addWidget(view_btn)
        
        # Connect signals
        refresh_btn.clicked.connect(self.refresh_history_list)
        self.history_list.itemSelectionChanged.connect(
            lambda: view_btn.setEnabled(len(self.history_list.selectedItems()) > 0)
        )
        view_btn.clicked.connect(self.view_selected_history)
        
        # Store buttons for later use
        self.refresh_history_btn = refresh_btn
        self.view_history_btn = view_btn
        
        section.add_control(content)
        
        # Load initial history
        self.refresh_history_list()
        
        return section
    
    def refresh_history_list(self):
        """Refresh the analysis history list"""
        import os
        import glob
        from datetime import datetime
        
        self.history_list.clear()
        
        # Check if results directory exists
        results_dir = "results"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
            return
        
        # Get all .txt files in results directory
        pattern = os.path.join(results_dir, "*.txt")
        files = glob.glob(pattern)
        
        # Parse files and create display items
        history_items = []
        for file_path in files:
            try:
                filename = os.path.basename(file_path)
                # Parse filename format: Name_Age_Gender_YYYYMMDD_HHMMSS.txt
                parts = filename.replace('.txt', '').split('_')
                
                if len(parts) >= 5:
                    # Extract info
                    name_parts = []
                    age_part = None
                    gender_part = None
                    date_part = None
                    time_part = None
                    
                    # Find age, gender, date, time parts
                    for i, part in enumerate(parts):
                        if part.endswith('tuoi'):
                            age_part = part.replace('tuoi', '')
                            if i > 0:
                                name_parts = parts[:i]
                            if i + 1 < len(parts):
                                gender_part = parts[i + 1]
                            if i + 2 < len(parts):
                                date_part = parts[i + 2]
                            if i + 3 < len(parts):
                                time_part = parts[i + 3]
                            break
                    
                    if age_part and gender_part and date_part and time_part:
                        name = '_'.join(name_parts) if name_parts else "Unknown"
                        
                        # Parse date and time
                        try:
                            datetime_str = f"{date_part}_{time_part}"
                            dt = datetime.strptime(datetime_str, "%Y%m%d_%H%M%S")
                            
                            # Format display text
                            display_text = (f"👤 {name} ({age_part} tuổi, {gender_part})\n"
                                          f"📅 {dt.strftime('%d/%m/%Y')} - "
                                          f"🕒 {dt.strftime('%H:%M:%S')}")
                            
                            history_items.append({
                                'display': display_text,
                                'file_path': file_path,
                                'datetime': dt,
                                'name': name,
                                'age': age_part,
                                'gender': gender_part
                            })
                        except ValueError:
                            continue
                            
            except Exception as e:
                print(f"Error parsing file {file_path}: {e}")
                continue
        
        # Sort by datetime (newest first)
        history_items.sort(key=lambda x: x['datetime'], reverse=True)
        
        # Add items to list
        for item in history_items:
            list_item = QListWidgetItem(item['display'])
            list_item.setData(Qt.ItemDataRole.UserRole, item)
            self.history_list.addItem(list_item)
    
    def view_selected_history(self):
        """View the selected history item"""
        from PyQt6.QtWidgets import QListWidgetItem
        
        selected_items = self.history_list.selectedItems()
        if not selected_items:
            return
            
        item = selected_items[0]
        item_data = item.data(Qt.ItemDataRole.UserRole)
        
        if item_data:
            file_path = item_data['file_path']
            # Emit signal to main window to load and display history
            self.view_history_requested.emit(file_path)
    
    def create_camera_section(self):
        """Create camera controls section"""
        section = CollapsibleSection("RGB Camera")
        section.header.setStyleSheet(section.header.styleSheet() + "font-size: 16px;")
        
        # Fixed settings display
        settings_frame = QFrame()
        settings_frame.setStyleSheet("""
            QFrame {
                background-color: #ffffff;
                border: 1px solid #dddddd;
                border-radius: 6px;
                padding: 8px;
            }
        """)
        
        settings_layout = QVBoxLayout()
        settings_layout.setSpacing(4)
        
        # Resolution (fixed)
        resolution_label = QLabel("Resolution: 1280 x 720")
        resolution_label.setStyleSheet("color: #cccccc; font-weight: bold;")
        
        # Frame Rate (fixed)
        fps_label = QLabel("Frame Rate: 30 FPS")
        fps_label.setStyleSheet("color: #cccccc; font-weight: bold;")
        
        # Connection status
        self.camera_status_label = QLabel("Status: Disconnected")
        self.camera_status_label.setStyleSheet("color: #ff6b6b; font-weight: bold;")
        
        settings_layout.addWidget(resolution_label)
        settings_layout.addWidget(fps_label)
        settings_layout.addWidget(self.camera_status_label)
        
        settings_frame.setLayout(settings_layout)
        section.add_control(settings_frame)
        
        # Auto Exposure
        auto_exposure = QCheckBox("Enable Auto Exposure")
        auto_exposure.setChecked(True)
        auto_exposure.setStyleSheet("""
            QCheckBox {
                color: #cccccc;
                font-size: 14px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
            }
            QCheckBox::indicator:unchecked {
                background-color: #ffffff;
                border: 1px solid #cccccc;
                border-radius: 3px;
            }
            QCheckBox::indicator:checked {
                background-color: #0078d4;
                border: 1px solid #0078d4;
                border-radius: 3px;
            }
        """)
        section.add_control(auto_exposure)
        
        return section
    
    def create_pose_section(self):
        """Create pose detection controls"""
        section = CollapsibleSection("Pose Detection")
        section.header.setStyleSheet(section.header.styleSheet() + "font-size: 16px;")
        
        # Model selection
        model_dropdown = DropdownControl(
            "Model:",
            ["OpenPose Body_25", "OpenPose COCO", "MediaPipe"],
            0
        )
        section.add_control(model_dropdown)
        
        # Confidence threshold
        confidence_slider = SliderControl("Confidence Threshold:", 0, 100, 10)
        section.add_control(confidence_slider)
        
        # Processing size
        size_dropdown = DropdownControl(
            "Processing Size:",
            ["192x192 (Fast)", "256x256 (Balanced)", "368x368 (Accurate)"],
            0
        )
        section.add_control(size_dropdown)
        
        return section
    
    def create_metrics_section(self):
        """Create professional gait analysis metrics display"""
        section = CollapsibleSection("Gait Analysis")
        section.header.setStyleSheet(section.header.styleSheet() + "font-size: 16px;")
        
        # Main container - single frame without internal borders
        main_frame = QFrame()
        main_frame.setStyleSheet("""
            QFrame {
                background-color: #ffffff;
                border: 1px solid #dddddd;
                border-radius: 6px;
                padding: 8px;
                margin: 2px 0px;
            }
        """)
        
        main_layout = QVBoxLayout()
        main_layout.setSpacing(6)
        main_layout.setContentsMargins(4, 4, 4, 4)
        
        # Helper function to create compact rows
        def create_compact_row(label_text, value_text, value_color="#cccccc"):
            row_layout = QHBoxLayout()
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(0)
            
            label = QLabel(label_text)
            if label_text and any(char.isdigit() for char in label_text[:2]):  # If starts with number (1., 2., etc.)
                label.setStyleSheet("color: #0078d4; font-size: 16px; font-weight: bold; padding: 2px 0px;")
            else:
                label.setStyleSheet("color: #333333; font-size: 15px; padding: 2px 0px;")
            
            value = QLabel(value_text)
            value.setStyleSheet(f"color: {value_color}; font-weight: bold; font-size: 16px; padding: 2px 0px;")
            
            row_layout.addWidget(label)
            row_layout.addStretch()
            row_layout.addWidget(value)
            
            return row_layout, value
        
        # Helper function to create section headers
        def create_section_header(title):
            header = QLabel(title)
            header.setStyleSheet("color: #000000; font-weight: bold; font-size: 12px; margin: 4px 0px 2px 0px;")
            return header
        
        # 1. Thông số góc khớp
        main_layout.addWidget(create_section_header("Thông số góc khớp:"))
        
        # 1. Góc khớp gối (Knee Angle)
        knee_angle_row, self.knee_angle_value = create_compact_row("1. Góc khớp gối (Knee Angle):", "0°", "#0078d4")
        main_layout.addLayout(knee_angle_row)
        
        # 2. Góc khớp hông (Hip Angle)  
        hip_angle_row, self.hip_angle_value = create_compact_row("2. Góc khớp hông (Hip Angle):", "0°", "#0078d4")
        main_layout.addLayout(hip_angle_row)
        
        # 3. Góc khớp cổ chân (Ankle Angle)
        ankle_angle_row, self.ankle_angle_value = create_compact_row("3. Góc khớp cổ chân (Ankle Angle):", "0°", "#0078d4")
        main_layout.addLayout(ankle_angle_row)
        
        main_layout.addSpacing(4)
        
        # 2. Thông số vị trí và khoảng cách
        main_layout.addWidget(create_section_header("Thông số vị trí và khoảng cách:"))
        
        # 4. Chiều cao nâng chân (Foot Clearance)
        foot_clearance_row, self.foot_clearance_value = create_compact_row("4. Chiều cao nâng chân (Foot Clearance):", "0 cm", "#0078d4")
        main_layout.addLayout(foot_clearance_row)
        
        # 5. Chiều dài bước (Step Length)
        step_length_row, self.step_length_value = create_compact_row("5. Chiều dài bước (Step Length):", "0 cm", "#0078d4")
        main_layout.addLayout(step_length_row)
        
        # 6. Chiều rộng bước (Step Width)
        step_width_row, self.step_width_value = create_compact_row("6. Chiều rộng bước (Step Width):", "0 cm", "#0078d4")
        main_layout.addLayout(step_width_row)
        
        main_layout.addSpacing(4)
        
        # 3. Thông số về thời gian
        main_layout.addWidget(create_section_header("Thông số về thời gian:"))
        
        # 7. Pha tiếp xúc (Stance Phase)
        stance_phase_row, self.stance_phase_value = create_compact_row("7. Pha tiếp xúc (Stance Phase):", "0%", "#0078d4")
        main_layout.addLayout(stance_phase_row)
        
        # 8. Pha bay (Swing Phase)
        swing_phase_row, self.swing_phase_value = create_compact_row("8. Pha bay (Swing Phase):", "0%", "#0078d4")
        main_layout.addLayout(swing_phase_row)
        
        main_layout.addSpacing(4)
        
        # 4. Thông số đối xứng
        main_layout.addWidget(create_section_header("Thông số đối xứng:"))
        
        # 9. Symmetry Index
        symmetry_row, self.symmetry_value = create_compact_row("9. Symmetry Index:", "0%", "#0078d4")
        main_layout.addLayout(symmetry_row)
        
        main_frame.setLayout(main_layout)
        section.add_control(main_frame)
        
        # Reset button
        reset_btn = QPushButton("Reset Analysis")
        reset_btn.clicked.connect(self.reset_analysis.emit)
        reset_btn.setStyleSheet("""
            QPushButton {
                background-color: #d73527;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 6px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #e04030;
            }
        """)
        section.add_control(reset_btn)
        
        return section
    
    def create_advanced_section(self):
        """Create advanced controls"""
        section = CollapsibleSection("Advanced Controls")
        section.header.setStyleSheet(section.header.styleSheet() + "font-size: 16px;")
        
        # Mirror image
        mirror_checkbox = QCheckBox("Mirror Image")
        mirror_checkbox.setChecked(True)
        mirror_checkbox.setStyleSheet("""
            QCheckBox {
                color: #cccccc;
                font-size: 14px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
            }
            QCheckBox::indicator:unchecked {
                background-color: #ffffff;
                border: 1px solid #cccccc;
                border-radius: 3px;
            }
            QCheckBox::indicator:checked {
                background-color: #0078d4;
                border: 1px solid #0078d4;
                border-radius: 3px;
            }
        """)
        section.add_control(mirror_checkbox)
        
        # Show keypoint numbers
        numbers_checkbox = QCheckBox("Show Keypoint Numbers")
        numbers_checkbox.setChecked(True)
        section.add_control(numbers_checkbox)
        
        # Analysis frequency
        freq_slider = SliderControl("Analysis Rate (FPS):", 1, 30, 10)
        section.add_control(freq_slider)
        
        return section
    
    def toggle_camera(self):
        """Toggle camera connection"""
        if self.camera_connected:
            self.disconnect_camera.emit()
        else:
            self.connect_camera.emit()
    
    def toggle_analysis(self):
        """Toggle analysis"""
        if self.analysis_running:
            self.stop_analysis.emit()
        else:
            self.start_analysis.emit()
    
    def set_camera_connected(self, connected):
        """Update camera connection state"""
        self.camera_connected = connected
        self.camera_btn.setChecked(connected)
        self.analysis_btn.setEnabled(connected)
        
        # Update status label - commented out since camera section is hidden
        # if connected:
        #     self.camera_status_label.setText("Status: Connected")
        #     self.camera_status_label.setStyleSheet("color: #00ff88; font-weight: bold;")
        # else:
        #     self.camera_status_label.setText("Status: Disconnected")
        #     self.camera_status_label.setStyleSheet("color: #ff6b6b; font-weight: bold;")
        if not connected:
            self.set_analysis_running(False)
    
    def set_analysis_running(self, running):
        """Update analysis running state"""
        self.analysis_running = running
        self.analysis_btn.setChecked(running)
    
    def update_metrics(self, metrics):
        """Update professional gait analysis metrics display"""
        if not metrics:
            return
        
        # Update Joint Angles with real-time calculations
        knee_left = metrics.get('knee_angle_left', 0)
        knee_right = metrics.get('knee_angle_right', 0) 
        hip_left = metrics.get('hip_angle_left', 0)
        hip_right = metrics.get('hip_angle_right', 0)
        ankle_left = metrics.get('ankle_angle_left', 0)
        ankle_right = metrics.get('ankle_angle_right', 0)
        
        if knee_left > 0 or knee_right > 0:
            avg_knee = (knee_left + knee_right) / 2 if knee_left > 0 and knee_right > 0 else max(knee_left, knee_right)
            self.knee_angle_value.setText(f"{avg_knee:.1f}°")
        else:
            self.knee_angle_value.setText("0°")
        
        if hip_left > 0 or hip_right > 0:
            avg_hip = (hip_left + hip_right) / 2 if hip_left > 0 and hip_right > 0 else max(hip_left, hip_right)
            self.hip_angle_value.setText(f"{avg_hip:.1f}°")
        else:
            self.hip_angle_value.setText("0°")
        
        if ankle_left > 0 or ankle_right > 0:
            avg_ankle = (ankle_left + ankle_right) / 2 if ankle_left > 0 and ankle_right > 0 else max(ankle_left, ankle_right)
            self.ankle_angle_value.setText(f"{avg_ankle:.1f}°")
        else:
            self.ankle_angle_value.setText("0°")
        
        # Update Position & Distance metrics
        foot_clearance = metrics.get('foot_clearance_left', 0)
        if foot_clearance > 0:
            # Convert pixel to approximate cm (rough estimation)
            clearance_cm = foot_clearance / 30  # Rough pixel to cm conversion
            self.foot_clearance_value.setText(f"{clearance_cm:.1f} cm")
        else:
            self.foot_clearance_value.setText("0 cm")
        
        step_length = metrics.get('step_length', 0)
        if step_length > 0:
            length_cm = step_length / 10  # Rough conversion
            self.step_length_value.setText(f"{length_cm:.1f} cm")
        else:
            self.step_length_value.setText("0 cm")
        
        step_width = metrics.get('step_width', 0)
        if step_width > 0:
            width_cm = step_width / 10  # Rough conversion
            self.step_width_value.setText(f"{width_cm:.1f} cm")
        else:
            self.step_width_value.setText("0 cm")
        
        # Update Timing metrics
        stance_phase = metrics.get('stance_phase', 0)
        swing_phase = metrics.get('swing_phase', 0)
        
        if stance_phase > 0:
            self.stance_phase_value.setText(f"{stance_phase:.1f}%")
        else:
            self.stance_phase_value.setText("0%")
        
        if swing_phase > 0:
            self.swing_phase_value.setText(f"{swing_phase:.1f}%")
        else:
            self.swing_phase_value.setText("0%")
        
        # Update Symmetry metrics
        symmetry = metrics.get('symmetry_index', 0)
        if symmetry > 0:
            self.symmetry_value.setText(f"{symmetry:.1f}%")
        else:
            self.symmetry_value.setText("0%")
