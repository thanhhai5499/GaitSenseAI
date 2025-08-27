"""
GaitSenseAI - Main Application
Gait Analysis System with RGB Camera Support
"""

import sys
import os
import time
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QHBoxLayout, 
                             QVBoxLayout, QSplitter, QMenuBar, QStatusBar, QMessageBox)
from PyQt6.QtCore import Qt, QTimer, pyqtSlot, QThread, pyqtSignal
from PyQt6.QtGui import QAction, QIcon, QPixmap

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.ui.realsense_sidebar import RealSenseSidebar
from src.ui.camera_panel import CameraPanel
from src.core.camera_handler import CameraHandler, CameraWidget
from src.core.gait_analyzer import GaitAnalyzer
from config.settings import UI_SETTINGS, APP_SETTINGS


class AnalysisWorker(QThread):
    """Separate thread for pose analysis to maintain 30fps camera"""
    
    # Signals
    frame_analyzed = pyqtSignal(object)  # analyzed frame with pose
    analysis_complete = pyqtSignal(dict)  # analysis metrics
    error_occurred = pyqtSignal(str)
    
    def __init__(self, gait_analyzer):
        super().__init__()
        self.gait_analyzer = gait_analyzer
        self.current_frame = None
        self.is_running = False
        
    def update_frame(self, frame):
        """Update frame for analysis"""
        self.current_frame = frame.copy() if frame is not None else None
        
    def run(self):
        """Main analysis loop running at 10 FPS"""
        self.is_running = True
        
        while self.is_running:
            if self.current_frame is not None:
                try:
                    # Analyze frame
                    analyzed_frame, metrics = self.gait_analyzer.analyze_frame(
                        self.current_frame.copy()
                    )
                    
                    # Check for leg detection failure
                    if metrics == "LEG_DETECTION_FAILED":
                        self.error_occurred.emit("LEG_DETECTION_FAILED")
                        return
                    
                    # Emit results
                    self.frame_analyzed.emit(analyzed_frame)
                    
                    # Get metrics summary
                    metrics_summary = self.gait_analyzer.get_analysis_summary()
                    self.analysis_complete.emit(metrics_summary)
                    
                except Exception as e:
                    self.error_occurred.emit(str(e))
            
            # Sleep for 100ms (10 FPS analysis)
            self.msleep(100)
    
    def stop(self):
        """Stop analysis thread"""
        self.is_running = False
        self.quit()
        self.wait()


class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        
        # Initialize components
        self.camera_handler = CameraHandler()
        self.camera_widget = CameraWidget()
        self.gait_analyzer = GaitAnalyzer()
        
        # Analysis state
        self.is_analyzing = False
        self.analysis_start_time = None
        
        # Analysis timer for 1-minute automatic stop
        self.analysis_timer = QTimer()
        self.analysis_timer.setSingleShot(True)
        self.analysis_timer.timeout.connect(self.stop_analysis_after_timeout)
        
        # Analysis worker thread (separate from UI thread)
        self.analysis_worker = AnalysisWorker(self.gait_analyzer)
        self.analysis_worker.frame_analyzed.connect(self.on_frame_analyzed)
        self.analysis_worker.analysis_complete.connect(self.on_analysis_complete)
        self.analysis_worker.error_occurred.connect(self.on_analysis_error)
        
        # Setup UI
        self.setup_ui()
        self.setup_connections()
        # self.setup_menubar()  # Ẩn thanh menu không cần thiết
        self.menuBar().hide()  # Ẩn thanh menu bar
        self.setup_statusbar()
        
        # Apply light theme after all UI is set up
        self.apply_light_theme()
        
        # Show countdown test (always visible for demo)
        self.camera_panel.show_countdown_test()
        
    def setup_ui(self):
        """Setup main UI layout"""
        self.setWindowTitle(APP_SETTINGS["title"])
        self.setGeometry(100, 100, UI_SETTINGS["window_width"], UI_SETTINGS["window_height"])
        
        # Central widget
        central_widget = QWidget()
        central_widget.setStyleSheet("""
            QWidget {
                background-color: #ffffff;
                color: #000000;
            }
        """)
        self.setCentralWidget(central_widget)
        
        # Main horizontal layout
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Sidebar
        self.sidebar = RealSenseSidebar()
        # Sidebar đã có fixed width trong design
        
        # Camera panel
        self.camera_panel = CameraPanel()
        
        # Add to splitter
        splitter.addWidget(self.sidebar)
        splitter.addWidget(self.camera_panel)
        
        # Set splitter proportions (sidebar: camera = 1:2)
        splitter.setSizes([450, 900])
        splitter.setCollapsible(0, False)  # Don't allow sidebar to collapse
        splitter.setCollapsible(1, False)  # Don't allow camera panel to collapse
        
        main_layout.addWidget(splitter)
        central_widget.setLayout(main_layout)
    
    def setup_connections(self):
        """Setup signal connections"""
        # Sidebar signals
        self.sidebar.connect_camera.connect(self.connect_camera)
        self.sidebar.disconnect_camera.connect(self.disconnect_camera)
        self.sidebar.start_analysis.connect(self.start_analysis)
        self.sidebar.stop_analysis.connect(self.stop_analysis)
        self.sidebar.reset_analysis.connect(self.reset_analysis)
        self.sidebar.view_history_requested.connect(self.view_history_file)
        
        # Camera handler signals
        self.camera_handler.frame_ready.connect(self.on_frame_ready)
        self.camera_handler.error_occurred.connect(self.on_camera_error)
    
    def setup_menubar(self):
        """Setup application menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('File')
        
        exit_action = QAction('Exit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Camera menu
        camera_menu = menubar.addMenu('Camera')
        
        connect_action = QAction('Connect Camera', self)
        connect_action.triggered.connect(self.connect_camera)
        camera_menu.addAction(connect_action)
        
        disconnect_action = QAction('Disconnect Camera', self)
        disconnect_action.triggered.connect(self.disconnect_camera)
        camera_menu.addAction(disconnect_action)
        
        # Analysis menu
        analysis_menu = menubar.addMenu('Analysis')
        
        start_action = QAction('Start Analysis', self)
        start_action.setShortcut('Ctrl+S')
        start_action.triggered.connect(self.start_analysis)
        analysis_menu.addAction(start_action)
        
        stop_action = QAction('Stop Analysis', self)
        stop_action.setShortcut('Ctrl+T')
        stop_action.triggered.connect(self.stop_analysis)
        analysis_menu.addAction(stop_action)
        
        reset_action = QAction('Reset Analysis', self)
        reset_action.setShortcut('Ctrl+R')
        reset_action.triggered.connect(self.reset_analysis)
        analysis_menu.addAction(reset_action)
        
        # Help menu
        help_menu = menubar.addMenu('Help')
        
        about_action = QAction('About', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def setup_statusbar(self):
        """Setup status bar"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready - Connect camera to start analysis")
    
    def apply_light_theme(self):
        """Apply light theme to application"""
        # Apply app-wide light theme
        app_style = """
            QMainWindow {
                background-color: #ffffff;
                color: #000000;
            }
            QWidget {
                background-color: #ffffff;
                color: #000000;
            }
            QMenuBar {
                background-color: #f0f0f0;
                color: #000000;
                border-bottom: 1px solid #cccccc;
                padding: 4px;
            }
            QMenuBar::item {
                background-color: transparent;
                padding: 8px 12px;
                border-radius: 4px;
            }
            QMenuBar::item:selected {
                background-color: #e0e0e0;
            }
            QMenu {
                background-color: #ffffff;
                color: #000000;
                border: 1px solid #cccccc;
                border-radius: 4px;
            }
            QMenu::item {
                padding: 8px 25px;
            }
            QMenu::item:selected {
                background-color: #e0e0e0;
            }
            QStatusBar {
                background-color: #f0f0f0;
                color: #333333;
                border-top: 1px solid #cccccc;
                padding: 4px 8px;
            }
            QSplitter {
                background-color: #ffffff;
            }
            QSplitter::handle {
                background-color: #cccccc;
                width: 2px;
            }
            QSplitter::handle:hover {
                background-color: #0078d4;
            }
            QHBoxLayout, QVBoxLayout {
                margin: 0px;
                padding: 0px;
            }
        """
        
        # Apply to the main application
        QApplication.instance().setStyleSheet(app_style)
    
    @pyqtSlot()
    def connect_camera(self):
        """Connect to camera"""
        try:
            self.camera_handler.start_capture()
            self.camera_panel.set_camera_connected(True)
            self.sidebar.set_camera_connected(True)
            self.status_bar.showMessage("Camera connected - Ready for analysis")
        except Exception as e:
            QMessageBox.critical(self, "Camera Error", f"Failed to connect camera: {str(e)}")
            self.sidebar.set_camera_connected(False)
    
    @pyqtSlot()
    def disconnect_camera(self):
        """Disconnect camera"""
        self.camera_handler.stop_capture()
        self.camera_panel.set_camera_connected(False)
        self.sidebar.set_camera_connected(False)
        self.stop_analysis()
        self.status_bar.showMessage("Camera disconnected")
    
    @pyqtSlot()
    def start_analysis(self):
        """Start gait analysis - requires camera to be connected first"""
        if not self.camera_handler.camera:
            QMessageBox.warning(self, "Camera Required", 
                               "Vui lòng kết nối camera trước khi bắt đầu phân tích!")
            return
        
        # Validate patient information before starting analysis
        patient_info = self.sidebar.get_patient_info()
        validation_result = self.validate_patient_info(patient_info)
        
        if not validation_result["valid"]:
            QMessageBox.warning(self, "Thông Tin Thiếu", validation_result["message"])
            return
        
        # Reset analyzer for new session
        self.gait_analyzer.reset_analysis()
        
        self.is_analyzing = True
        self.analysis_start_time = time.time()
        
        # Start 1-minute timer (60,000 ms)
        self.analysis_timer.start(60000)
        
        self.analysis_worker.start()  # Start analysis thread
        self.status_bar.showMessage("Phân tích dáng đi đang chạy (1 phút)...")
        self.camera_panel.show_overlay_info("Phân tích dáng đi")
        self.camera_panel.start_countdown(60)  # Start 1-minute countdown
        self.sidebar.set_analysis_running(True)
    
    @pyqtSlot()
    def stop_analysis(self):
        """Stop gait analysis"""
        self.is_analyzing = False
        self.analysis_timer.stop()  # Stop timer if running
        self.analysis_worker.stop()  # Stop analysis thread
        self.sidebar.set_analysis_running(False)
        self.status_bar.showMessage("Analysis stopped")
        self.camera_panel.hide_overlay_info()
        self.camera_panel.stop_countdown()  # Stop countdown timer
    
    def stop_analysis_after_timeout(self):
        """Stop analysis after 1-minute timeout and show results"""
        self.stop_analysis()
        
        # Get patient info from sidebar
        patient_info = self.sidebar.get_patient_info()
        
        # Create session name from patient info
        session_name = self.create_session_name(patient_info)
        
        # Export session data to file
        exported_file = self.gait_analyzer.export_session_data(session_name, patient_info)
        
        if exported_file:
            # Generate comprehensive diagnosis from exported file
            diagnosis = self.gait_analyzer.load_and_diagnose(exported_file)
            
            if diagnosis:
                # Show comprehensive diagnosis results
                self.show_comprehensive_diagnosis(diagnosis, exported_file)
            else:
                # Fallback to basic results
                final_results = self.gait_analyzer.get_analysis_summary()
                comparison_results = self.gait_analyzer.get_bilateral_comparison()
                self.show_analysis_results(final_results, comparison_results)
        else:
            # Fallback if export failed
            final_results = self.gait_analyzer.get_analysis_summary()
            comparison_results = self.gait_analyzer.get_bilateral_comparison()
            self.show_analysis_results(final_results, comparison_results)
        
        self.status_bar.showMessage("Phân tích hoàn tất - Dữ liệu đã lưu và phân tích")
        
        # Refresh history list to show new analysis
        self.refresh_history_list()
    
    @pyqtSlot()
    def reset_analysis(self):
        """Reset gait analysis"""
        self.gait_analyzer.reset_analysis()
        self.sidebar.update_metrics({})
        self.status_bar.showMessage("Analysis reset")
    
    @pyqtSlot(object)
    def on_frame_ready(self, frame):
        """Handle new camera frame"""
        # Always update display with raw camera feed if not analyzing
        if not self.is_analyzing:
            pixmap = self.camera_widget.frame_to_pixmap(frame)
            self.camera_panel.update_camera_frame(pixmap)
        
        # Feed frame to analysis worker for processing
        if self.is_analyzing:
            self.analysis_worker.update_frame(frame)
        
        # Store frame for fallback
        self.camera_widget.update_frame(frame)
    
    @pyqtSlot(str)
    def on_camera_error(self, error_message):
        """Handle camera errors"""
        QMessageBox.warning(self, "Camera Error", error_message)
        self.camera_panel.set_camera_connected(False)
        self.stop_analysis()
        self.status_bar.showMessage(f"Camera error: {error_message}")
    
    @pyqtSlot(object)
    def on_frame_analyzed(self, analyzed_frame):
        """Handle analyzed frame from worker thread"""
        if self.is_analyzing and analyzed_frame is not None:
            # Update display with analyzed frame (with pose landmarks)
            pixmap = self.camera_widget.frame_to_pixmap(analyzed_frame)
            self.camera_panel.update_camera_frame(pixmap)
    
    @pyqtSlot(dict)
    def on_analysis_complete(self, metrics):
        """Handle analysis metrics from worker thread"""
        if self.is_analyzing:
            self.sidebar.update_metrics(metrics)
    
    @pyqtSlot(str)
    def on_analysis_error(self, error_message):
        """Handle analysis errors from worker thread"""
        print(f"❌ Analysis error: {error_message}")
        
        # Handle leg detection failure
        if error_message == "LEG_DETECTION_FAILED":
            self.stop_analysis()
            QMessageBox.warning(self, "Lỗi Phát Hiện Chân", 
                               "Không thể phát hiện đủ điểm chân trong thời gian dài.\n\n"
                               "Vui lòng:\n"
                               "• Đảm bảo toàn bộ chân (từ hông xuống) hiển thị trong camera\n"
                               "• Đứng ở nơi có ánh sáng đủ\n"
                               "• Mặc quần áo tương phản với nền\n"
                               "• Thử lại phân tích")
            self.status_bar.showMessage("Phân tích dừng - Không phát hiện được chân")
    
    def show_analysis_results(self, final_results, comparison_results):
        """Show analysis results dialog after 2-minute session"""
        result_text = "📊 KẾT QUẢ PHÂN TÍCH DÁNG ĐI (2 PHÚT)\n" + "="*50 + "\n\n"
        
        # Basic metrics summary
        result_text += "📈 THÔNG SỐ TỔNG QUAN:\n"
        result_text += f"• Cadence: {final_results.get('cadence', 0):.1f} bước/phút\n"
        result_text += f"• Tốc độ đi: {final_results.get('walking_speed', 0):.1f} cm/s\n"
        result_text += f"• Chỉ số đối xứng: {final_results.get('symmetry_index', 0):.1f}%\n\n"
        
        # Bilateral comparison
        result_text += "⚖️ SO SÁNH GIỮA HAI CHÂN:\n"
        
        # Joint angles comparison
        knee_diff = abs(comparison_results.get('knee_angle_difference', 0))
        hip_diff = abs(comparison_results.get('hip_angle_difference', 0))
        ankle_diff = abs(comparison_results.get('ankle_angle_difference', 0))
        
        result_text += f"• Chênh lệch góc gối: {knee_diff:.1f}° "
        if knee_diff > 10:
            result_text += "❌ (Lệch nhiều)\n"
        elif knee_diff > 5:
            result_text += "⚠️ (Lệch vừa)\n"
        else:
            result_text += "✅ (Bình thường)\n"
            
        result_text += f"• Chênh lệch góc hông: {hip_diff:.1f}° "
        if hip_diff > 8:
            result_text += "❌ (Lệch nhiều)\n"
        elif hip_diff > 4:
            result_text += "⚠️ (Lệch vừa)\n"
        else:
            result_text += "✅ (Bình thường)\n"
            
        result_text += f"• Chênh lệch góc cổ chân: {ankle_diff:.1f}° "
        if ankle_diff > 6:
            result_text += "❌ (Lệch nhiều)\n"
        elif ankle_diff > 3:
            result_text += "⚠️ (Lệch vừa)\n"
        else:
            result_text += "✅ (Bình thường)\n\n"
        
        # Overall assessment
        result_text += "🏥 ĐÁNH GIÁ TỔNG THỂ:\n"
        total_issues = 0
        if knee_diff > 5: total_issues += 1
        if hip_diff > 4: total_issues += 1  
        if ankle_diff > 3: total_issues += 1
        
        if total_issues == 0:
            result_text += "✅ Dáng đi cân đối, không có dấu hiệu bất thường.\n"
        elif total_issues == 1:
            result_text += "⚠️ Có một số bất cân xứng nhẹ, nên theo dõi thêm.\n"
        else:
            result_text += "❌ Phát hiện bất cân xứng rõ rệt, khuyên nên tham khảo ý kiến chuyên gia.\n"
        
        result_text += "\n💡 Lưu ý: Kết quả chỉ mang tính chất tham khảo."
        
        # Show results in message box with scroll
        msg = QMessageBox(self)
        msg.setWindowTitle("Kết Quả Phân Tích Dáng Đi")
        msg.setText(result_text)
        msg.setIcon(QMessageBox.Icon.Information)
        msg.exec()
    
    def show_comprehensive_diagnosis(self, diagnosis, data_file):
        """Show comprehensive medical-style diagnosis"""
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QTextEdit, QPushButton, QHBoxLayout, QLabel
        from PyQt6.QtCore import Qt
        
        # Create dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Báo Cáo Chuẩn Đoán Dáng Đi - GaitSenseAI")
        dialog.setModal(True)
        dialog.resize(800, 600)
        
        layout = QVBoxLayout()
        
        # Header
        header = QLabel("📋 BÁO CÁO CHUẨN ĐOÁN DÁNG ĐI")
        header.setStyleSheet("font-size: 24px; font-weight: bold; color: #0078d4; padding: 15px;")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(header)
        
        # Content area
        content = QTextEdit()
        content.setReadOnly(True)
        content.setStyleSheet("""
            QTextEdit {
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 14px;
                line-height: 1.6;
                background-color: #ffffff;
                border: 1px solid #cccccc;
                padding: 15px;
            }
        """)
        
        # Generate comprehensive report text
        report_text = self.generate_diagnosis_report(diagnosis, data_file)
        content.setHtml(report_text)
        
        layout.addWidget(content)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        # Open data file button
        open_file_btn = QPushButton("📄 Mở File Dữ Liệu")
        open_file_btn.clicked.connect(lambda: self.open_data_file(data_file))
        open_file_btn.setStyleSheet("""
            QPushButton {
                background-color: #0078d4;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 12px 20px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #106ebe;
            }
        """)
        
        # Close button
        close_btn = QPushButton("✅ Đóng")
        close_btn.clicked.connect(dialog.accept)
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 12px 20px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #218838;
            }
        """)
        
        button_layout.addWidget(open_file_btn)
        button_layout.addStretch()
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
        dialog.setLayout(layout)
        
        # Show dialog
        dialog.exec()
    
    def generate_diagnosis_report(self, diagnosis, data_file):
        """Generate HTML diagnosis report"""
        severity_colors = {
            0: "#28a745",  # Green - Normal
            1: "#ffc107",  # Yellow - Mild
            2: "#fd7e14",  # Orange - Moderate
            3: "#dc3545"   # Red - Severe
        }
        
        severity_labels = {
            0: "BÌNH THƯỜNG",
            1: "NHẸ",
            2: "VỪA PHẢI", 
            3: "NGHIÊM TRỌNG"
        }
        
        severity_score = diagnosis.get('severity_score', 0)
        severity_color = severity_colors.get(severity_score, "#6c757d")
        severity_label = severity_labels.get(severity_score, "KHÔNG XÁC ĐỊNH")
        
        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; font-size: 16px; }}
                .header {{ background-color: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 25px; }}
                .severity {{ color: {severity_color}; font-weight: bold; font-size: 20px; }}
                .section {{ margin-bottom: 25px; }}
                .section-title {{ color: #0078d4; font-weight: bold; font-size: 18px; margin-bottom: 15px; }}
                .finding {{ margin-bottom: 15px; padding: 15px; border-left: 4px solid #0078d4; background-color: #f8f9fa; font-size: 14px; }}
                .status-normal {{ border-left-color: #28a745; }}
                .status-mild {{ border-left-color: #ffc107; }}
                .status-moderate {{ border-left-color: #fd7e14; }}
                .status-severe {{ border-left-color: #dc3545; }}
                .recommendation {{ background-color: #e7f3ff; padding: 12px; border-radius: 6px; margin-top: 8px; font-size: 14px; }}
                .data-info {{ background-color: #f1f3f4; padding: 15px; border-radius: 6px; font-size: 14px; }}
                h2 {{ font-size: 22px; }}
                h3 {{ font-size: 18px; }}
                p {{ font-size: 16px; margin: 8px 0; }}
                strong {{ font-weight: bold; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h2>📋 BÁO CÁO PHÂN TÍCH DÁNG ĐI</h2>
                <p><strong>👤 Bệnh nhân:</strong> {diagnosis.get('patient_name', 'N/A')}</p>
                <p><strong>🎂 Tuổi:</strong> {diagnosis.get('patient_age', 'N/A')} tuổi</p>
                <p><strong>⚥ Giới tính:</strong> {diagnosis.get('patient_gender', 'N/A')}</p>
                {f"<p><strong>📝 Ghi chú:</strong> {diagnosis.get('patient_notes')}</p>" if diagnosis.get('patient_notes') else ""}
                <p><strong>📅 Ngày phân tích:</strong> {diagnosis.get('session_date', 'N/A')}</p>
                <hr style="margin: 15px 0;">
                <p><strong>🎯 Tình trạng tổng thể:</strong> <span class="severity">{severity_label}</span></p>
                <p><strong>📊 Đánh giá:</strong> {diagnosis.get('assessment_summary', 'N/A')}</p>
                <p><strong>👨‍⚕️ Cần theo dõi:</strong> {'Có' if diagnosis.get('follow_up_needed', False) else 'Không'}</p>
            </div>
            
            <div class="section">
                <div class="section-title">🔍 CHI TIẾT PHÂN TÍCH TỪNG KHỚP</div>
        """
        
        # Detailed findings for each joint
        findings = diagnosis.get('detailed_findings', {})
        joint_icons = {
            'Khớp Gối': '🦵',
            'Khớp Hông': '🏃', 
            'Khớp Cổ Chân': '🦶'
        }
        
        for joint, data in findings.items():
            status = data.get('status', 'UNKNOWN')
            difference = data.get('difference', 0)
            asymmetry = data.get('asymmetry', 0)
            recommendation = data.get('recommendation', 'N/A')
            
            # Map Vietnamese status to CSS class
            status_css_map = {
                'BÌNH THƯỜNG': 'normal',
                'NHẸ': 'mild',
                'TRUNG BÌNH': 'moderate', 
                'NẶNG': 'severe'
            }
            status_class = f"status-{status_css_map.get(status, 'unknown')}"
            icon = joint_icons.get(joint, '⚕️')
            
            # Check if this is normative data or traditional assessment
            if 'deviation_index' in data:
                # Normative data assessment
                deviation_index = data.get('deviation_index', 0)
                position_percentage = data.get('position_percentage', 50)
                norm_mean = data.get('normative_mean', 0)
                norm_std = data.get('normative_std', 0)
                
                # Tạo giải thích vị trí so sánh
                if position_percentage >= 95:
                    position_explanation = "Cao hơn hầu hết mọi người"
                elif position_percentage >= 75:
                    position_explanation = "Cao hơn 3/4 dân số"
                elif position_percentage >= 50:
                    position_explanation = "Trên mức trung bình"
                elif position_percentage >= 25:
                    position_explanation = "Dưới mức trung bình"
                else:
                    position_explanation = "Thấp hơn hầu hết mọi người"
                
                html += f"""
                <div class="finding {status_class}">
                    <h4>{icon} {joint}</h4>
                    <p><strong>Trạng thái:</strong> {status}</p>
                    <p><strong>Chỉ Số Lệch Chuẩn:</strong> {deviation_index}</p>
                    <p><strong>Vị Trí So Sánh:</strong> {position_percentage:.1f}% ({position_explanation})</p>
                    <p><strong>Chỉ Số Chuẩn:</strong> {norm_mean:.1f} ± {norm_std:.1f}</p>
                    <div class="recommendation">
                        <strong>Đánh giá:</strong> {recommendation}
                    </div>
                </div>
                """
            elif 'asymmetry_percent' in data:
                # Asymmetry assessment
                asymmetry = data.get('asymmetry_percent', 0)
                html += f"""
                <div class="finding {status_class}">
                    <h4>{icon} {joint}</h4>
                    <p><strong>Trạng thái:</strong> {status}</p>
                    <p><strong>Mức bất cân xứng:</strong> {asymmetry:.1f}%</p>
                    <div class="recommendation">
                        <strong>Đánh giá:</strong> {recommendation}
                    </div>
                </div>
                """
            else:
                # Traditional assessment
                difference = data.get('difference', 0)
                asymmetry = data.get('asymmetry', 0)
                html += f"""
                <div class="finding {status_class}">
                    <h4>{icon} {joint}</h4>
                    <p><strong>Trạng thái:</strong> {status}</p>
                    <p><strong>Chênh lệch góc:</strong> {difference:.1f}°</p>
                    <p><strong>Mức bất cân xứng:</strong> {asymmetry:.1f}%</p>
                    <div class="recommendation">
                        <strong>Khuyến nghị:</strong> {recommendation}
                    </div>
                </div>
                """
        
        # Recommendations
        recommendations = diagnosis.get('recommendations', [])
        if recommendations:
            html += """
                </div>
                <div class="section">
                    <div class="section-title">💡 KHUYẾN NGHỊ ĐIỀU TRỊ</div>
                    <ul>
            """
            for rec in recommendations:
                html += f"<li>{rec}</li>"
            html += "</ul></div>"
        
        # Data file info
        html += f"""
            <div class="section">
                <div class="section-title">📊 THÔNG TIN DỮ LIỆU</div>
                <div class="data-info">
                    <p><strong>File dữ liệu:</strong> {data_file}</p>
                    <p><strong>Lưu ý:</strong> File dữ liệu chứa tất cả thông số chi tiết và có thể được sử dụng cho phân tích thêm hoặc tham khảo y tế.</p>
                </div>
            </div>
            
            <div style="margin-top: 30px; padding: 15px; background-color: #fff3cd; border-radius: 5px; border-left: 4px solid #ffc107;">
                <p><strong>⚠️ Lưu ý quan trọng:</strong></p>
                <p>Kết quả này chỉ mang tính chất tham khảo và không thay thế cho việc khám bệnh chuyên khoa. 
                Vui lòng tham khảo ý kiến bác sĩ chuyên khoa để được tư vấn và điều trị phù hợp.</p>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def create_session_name(self, patient_info):
        """Create session name from patient information"""
        import re
        
        name = patient_info.get('name', '').strip()
        age = patient_info.get('age', 0)
        gender = patient_info.get('gender', 'Nam')
        
        # Clean name for filename (remove special characters)
        if name:
            # Remove diacritics and special characters for filename safety
            clean_name = re.sub(r'[^\w\s-]', '', name)
            clean_name = re.sub(r'\s+', '_', clean_name)  # Replace spaces with underscores
            session_name = f"{clean_name}_{age}tuoi_{gender}"
        else:
            session_name = f"BenhNhan_{age}tuoi_{gender}"
        
        return session_name
    
    def open_data_file(self, filename):
        """Open data file with default system application"""
        import os
        import subprocess
        import platform
        
        try:
            if platform.system() == 'Windows':
                os.startfile(filename)
            elif platform.system() == 'Darwin':  # macOS
                subprocess.call(['open', filename])
            else:  # Linux
                subprocess.call(['xdg-open', filename])
        except Exception as e:
            QMessageBox.warning(self, "Lỗi", f"Không thể mở file: {str(e)}")
    
    @pyqtSlot(str)
    def view_history_file(self, file_path):
        """View historical analysis file"""
        try:
            # Load and diagnose the historical file
            diagnosis = self.gait_analyzer.load_and_diagnose(file_path)
            
            if diagnosis:
                # Show historical diagnosis results
                self.show_comprehensive_diagnosis(diagnosis, file_path)
            else:
                QMessageBox.warning(self, "Lỗi", 
                                   f"Không thể đọc file lịch sử: {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Lỗi", 
                               f"Lỗi khi mở file lịch sử:\n{str(e)}")
    
    def refresh_history_list(self):
        """Refresh the history list in sidebar"""
        if hasattr(self.sidebar, 'refresh_history_list'):
            self.sidebar.refresh_history_list()
    
    def validate_patient_info(self, patient_info):
        """Validate patient information before starting analysis"""
        errors = []
        
        # Check name
        name = patient_info.get("name", "").strip()
        if not name:
            errors.append("• Họ và tên không được để trống")
        elif len(name) < 2:
            errors.append("• Họ và tên phải có ít nhất 2 ký tự")
        
        # Check age
        age = patient_info.get("age", 0)
        if age < 1 or age > 120:
            errors.append("• Tuổi phải từ 1 đến 120")
        
        if errors:
            message = "Vui lòng nhập đầy đủ thông tin bệnh nhân:\n\n" + "\n".join(errors)
            return {"valid": False, "message": message}
        
        return {"valid": True, "message": "Thông tin hợp lệ"}
    
    def show_about(self):
        """Show about dialog"""
        QMessageBox.about(self, "About GaitSenseAI", 
                         f"{APP_SETTINGS['title']} v{APP_SETTINGS['version']}\n\n"
                         "A real-time gait analysis system using computer vision.\n"
                         "Features RGB camera support and pose detection.")
    
    def closeEvent(self, event):
        """Handle application close"""
        self.stop_analysis()
        self.disconnect_camera()
        event.accept()


def main():
    """Main application entry point"""
    # Enable high DPI scaling for better text readability (PyQt6)
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    
    app = QApplication(sys.argv)
    app.setApplicationName("GaitSenseAI")
    app.setApplicationVersion(APP_SETTINGS["version"])
    
    # Set global font scaling for better readability
    font = app.font()
    font.setPointSize(font.pointSize() + 4)  # Tăng thêm 4 points cho rõ hơn
    app.setFont(font)
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    # Start event loop
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

