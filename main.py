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
        
        # Set splitter proportions (sidebar: camera = increased sidebar width)
        splitter.setSizes([600, 750])
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
            from src.core.gait_analyzer import GaitAnalyzer
            diagnosis = GaitAnalyzer.load_and_diagnose(exported_file)
            
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
        result_text += f"• Tốc độ đi: {final_results.get('walking_speed', 0):.2f} m/s\n"
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
        dialog.resize(1200, 800)
        
        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Header
        header = QLabel("BÁO CÁO CHUẨN ĐOÁN DÁNG ĐI")
        header.setStyleSheet("""
            font-size: 28px; 
            font-weight: bold; 
            color: #0078d4; 
            padding: 20px; 
            background-color: #f8f9fa; 
            border-radius: 8px; 
            margin-bottom: 10px;
        """)
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(header)
        
        # Content area
        content = QTextEdit()
        content.setReadOnly(True)
        content.setStyleSheet("""
            QTextEdit {
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 16px;
                line-height: 1.8;
                background-color: #ffffff;
                border: 2px solid #e0e0e0;
                border-radius: 8px;
                padding: 25px;
                margin: 10px;
            }
        """)
        
        # Generate comprehensive report text
        report_text = self.generate_diagnosis_report(diagnosis, data_file)
        content.setHtml(report_text)
        
        layout.addWidget(content)
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(15)
        button_layout.setContentsMargins(10, 20, 10, 10)
        
        # Open data file button
        open_file_btn = QPushButton("Mở File Dữ Liệu")
        open_file_btn.clicked.connect(lambda: self.open_data_file(data_file))
        open_file_btn.setStyleSheet("""
            QPushButton {
                background-color: #0078d4;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 15px 25px;
                font-weight: bold;
                font-size: 16px;
                min-width: 180px;
                min-height: 45px;
            }
            QPushButton:hover {
                background-color: #106ebe;
                transform: translateY(-1px);
            }
            QPushButton:pressed {
                background-color: #005a9e;
            }
        """)
        
        # Close button
        close_btn = QPushButton("Đóng Báo Cáo")
        close_btn.clicked.connect(dialog.accept)
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 15px 25px;
                font-weight: bold;
                font-size: 16px;
                min-width: 180px;
                min-height: 45px;
            }
            QPushButton:hover {
                background-color: #218838;
                transform: translateY(-1px);
            }
            QPushButton:pressed {
                background-color: #1e7e34;
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
        """Generate HTML diagnosis report focusing on asymmetry analysis"""
        
        # Lấy dữ liệu chi tiết từ diagnosis
        findings = diagnosis.get('detailed_findings', {})
        bilateral_comparison = diagnosis.get('bilateral_comparison', {})
        
        # Tính toán độ lệch so với chuẩn
        def calculate_deviation_percentage(measured, norm_mean, norm_std):
            if norm_std == 0:
                return 0
            deviation = abs(measured - norm_mean) / norm_std
            return min(deviation * 100, 999)  # Cap at 999%
        
        def get_status_color(status):
            colors = {
                'BÌNH THƯỜNG': '#28a745',
                'NHẸ': '#ffc107', 
                'TRUNG BÌNH': '#fd7e14',
                'NẶNG': '#dc3545'
            }
            return colors.get(status, '#6c757d')
        
        # Lấy dữ liệu chi tiết từ diagnosis
        findings = diagnosis.get('detailed_findings', {})
        
        # Tính toán độ lệch so với chuẩn
        def calculate_deviation_percentage(measured, norm_mean, norm_std):
            if norm_std == 0:
                return 0
            deviation = abs(measured - norm_mean) / norm_std
            return min(deviation * 100, 999)  # Cap at 999%
        
        html = f"""<!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Báo Cáo Phân Tích Dáng Đi</title>
            <style>
                body {{ 
                    font-family: 'Segoe UI', Arial, sans-serif; 
                    line-height: 1.6; 
                    font-size: 16px; 
                    margin: 0; 
                    padding: 20px;
                    background-color: #f8f9fa;
                    color: #333;
                }}
                .header {{ 
                    background: linear-gradient(135deg, #0078d4 0%, #106ebe 100%); 
                    color: white;
                    padding: 30px; 
                    border-radius: 15px; 
                    margin-bottom: 30px; 
                    box-shadow: 0 4px 15px rgba(0,120,212,0.3);
                    text-align: center;
                }}
                .patient-info {{
                    margin-top: 20px;
                    font-size: 18px;
                    color: #212529;
                    background: rgba(255,255,255,0.95);
                    padding: 15px;
                    border-radius: 10px;
                    text-align: left;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }}
                .patient-info strong {{
                    color: #0078d4;
                    font-weight: bold;
                }}
                .section {{ 
                    margin: 20px 0; 
                    padding: 25px; 
                    border-radius: 12px; 
                    background-color: #ffffff; 
                    box-shadow: 0 3px 10px rgba(0,0,0,0.1);
                    border: 1px solid #e9ecef;
                }}
                .asymmetry-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
                    gap: 20px;
                    margin: 20px 0;
                }}
                .joint-card {{
                    background: #ffffff;
                    border-radius: 12px;
                    padding: 20px;
                    border: 2px solid #e9ecef;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
                    position: relative;
                }}
                .joint-title {{
                    font-size: 18px;
                    font-weight: bold;
                    color: #0078d4;
                    margin-bottom: 15px;
                    display: flex;
                    align-items: center;
                }}
                .angle-comparison {{
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin: 8px 0;
                    padding: 12px;
                    background: #f8f9fa;
                    border-radius: 8px;
                    border: 1px solid #e9ecef;
                }}
                .deviation-visual {{
                    width: 100%;
                    height: 30px;
                    background: #e9ecef;
                    border-radius: 15px;
                    position: relative;
                    margin: 10px 0;
                    overflow: hidden;
                }}
                .deviation-fill {{
                    height: 100%;
                    border-radius: 15px;
                    transition: all 0.3s ease;
                    position: relative;
                }}
                .section-title {{ 
                    color: #0078d4; 
                    font-weight: bold; 
                    font-size: 20px; 
                    margin-bottom: 20px; 
                    padding-bottom: 10px;
                    border-bottom: 3px solid #0078d4;
                    display: flex;
                    align-items: center;
                }}
                .metric-value {{
                    font-size: 20px;
                    font-weight: bold;
                    color: #0078d4;
                }}
                .comparison-table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 15px 0;
                    border: 1px solid #dee2e6;
                    border-radius: 8px;
                    overflow: hidden;
                }}
                .comparison-table th, .comparison-table td {{
                    padding: 12px;
                    text-align: center;
                    border-bottom: 1px solid #dee2e6;
                }}
                .comparison-table th {{
                    background: #f8f9fa;
                    font-weight: bold;
                    color: #495057;
                }}
                .summary-card {{
                    background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
                    border-radius: 12px;
                    padding: 20px;
                    margin: 20px 0;
                    border-left: 5px solid #2196f3;
                }}
                .overall-status {{
                    font-size: 18px;
                    font-weight: bold;
                    padding: 15px;
                    border-radius: 10px;
                    text-align: center;
                    margin: 20px 0;
                }}
                h2 {{ font-size: 24px; margin: 15px 0; color: #212529; text-shadow: 0 2px 4px rgba(0,0,0,0.2); font-weight: bold; }}
                h3 {{ font-size: 18px; margin: 12px 0; color: #495057; }}
                strong {{ font-weight: bold; color: #212529; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h2>BÁO CÁO PHÂN TÍCH DÁNG ĐI CHI TIẾT</h2>
                <div class="patient-info">
                    <p><strong>Người Đo:</strong> {diagnosis.get('patient_name', 'N/A')} - {diagnosis.get('patient_age', 'N/A')} tuổi ({diagnosis.get('patient_gender', 'N/A')})</p>
                    <p><strong>Thời gian:</strong> {diagnosis.get('session_date', 'N/A')}</p>
                </div>
            </div>
        
            <!-- Tóm tắt tổng quan -->
            <div class="section">
                <div class="section-title">📊 TỔNG QUAN TÌNH TRẠNG</div>"""
        
        # Tóm tắt tổng quan
        overall_score = diagnosis.get('severity_score', 0)
        status_colors = {0: '#28a745', 1: '#ffc107', 2: '#fd7e14', 3: '#dc3545'}
        status_texts = {0: 'BÌNH THƯỜNG', 1: 'CẦN CHÚ Ý', 2: 'CẦN ĐIỀU TRỊ', 3: 'NGHIÊM TRỌNG'}
        
        html += f"""
                <div class="overall-status" style="background: {status_colors.get(overall_score, '#6c757d')}; color: white;">
                    {status_texts.get(overall_score, 'KHÔNG XÁC ĐỊNH')} - Điểm số: {overall_score}/3
                </div>
                <div class="summary-card">
                    <h3>📋 Đánh giá tổng thể:</h3>
                    <p>{diagnosis.get('assessment_summary', 'Không có đánh giá')}</p>
                    <p><strong>Cần theo dõi:</strong> {'✅ Có' if diagnosis.get('follow_up_needed', False) else '❌ Không'}</p>
                </div>
            </div>
        
            <!-- Phân tích bất đối xứng các khớp -->
            <div class="section">
                <div class="section-title">⚖️ PHÂN TÍCH BẤT ĐỐI XỨNG CÁC KHỚP</div>
                <div class="asymmetry-grid">"""
        
        # Các khớp cần phân tích
        joints = [
            ('Bất Cân Xứng Gối', '🦵', 'Khớp gối giữa chân trái và chân phải'),
            ('Bất Cân Xứng Hông', '🦴', 'Khớp hông giữa chân trái và chân phải'), 
            ('Bất Cân Xứng Cổ Chân', '🦶', 'Khớp cổ chân giữa chân trái và chân phải')
        ]
        
        def get_status_color(status):
            colors = {
                'BÌNH THƯỜNG': '#28a745',
                'NHẸ': '#ffc107', 
                'TRUNG BÌNH': '#fd7e14',
                'NẶNG': '#dc3545'
            }
            return colors.get(status, '#6c757d')
        
        for joint_name, emoji, description in joints:
            if joint_name in findings:
                data = findings[joint_name]
                status = data.get('status', 'KHÔNG RÕ')
                color = get_status_color(status)
                asymmetry_percent = data.get('asymmetry_percent', 0)
                recommendation = data.get('recommendation', 'Không có khuyến nghị')
                
                # Tính độ rộng của thanh deviation (max 100%)
                bar_width = min(asymmetry_percent * 10, 100)  # Scale for visualization
                
                html += f"""
                    <div class="joint-card" style="border-left: 5px solid {color};">
                        <div class="joint-title">{emoji} {joint_name}</div>
                        <p style="color: #666; font-style: italic; margin-bottom: 15px;">{description}</p>
                        
                        <div class="angle-comparison">
                            <span><strong>Mức bất cân xứng:</strong></span>
                            <span class="metric-value" style="color: {color};">{asymmetry_percent:.1f}%</span>
                        </div>
                        
                        <div class="deviation-visual">
                            <div class="deviation-fill" style="width: {bar_width}%; background: {color};">
                            </div>
                        </div>
                        
                        <div style="background: #f8f9fa; padding: 12px; border-radius: 8px; margin-top: 10px;">
                            <p style="margin: 0; font-weight: bold; color: {color};">📊 Đánh giá: {status}</p>
                            <p style="margin: 5px 0 0 0; font-size: 14px;"><strong>Giải thích:</strong> 
                                {'Cân xứng tốt' if asymmetry_percent <= 3 else 
                                 'Hơi lệch' if asymmetry_percent <= 6 else
                                 'Lệch rõ rệt' if asymmetry_percent <= 10 else 'Lệch nghiêm trọng'}
                            </p>
                        </div>
                        
                        <div style="background: #e9ecef; padding: 10px; border-radius: 6px; margin-top: 10px;">
                            <strong>💡 Khuyến nghị:</strong> {recommendation}
                        </div>
                    </div>
                """
        
        html += """
                </div>
            </div>
            
            <!-- Thông số chung -->
            <div class="section">
                <div class="section-title">📏 THÔNG SỐ CHUNG</div>
                <table class="comparison-table">
                    <thead>
                        <tr>
                            <th>Thông số</th>
                            <th>Giá trị đo được</th>
                            <th>Giá trị chuẩn</th>
                            <th>Độ lệch</th>
                            <th>Đánh giá</th>
                        </tr>
                    </thead>
                    <tbody>"""
        
        # Các thông số chung cần hiển thị
        general_params = [
            ('Tốc Độ Đi', 'm/s'),
            ('Chiều Dài Bước', 'cm'), 
            ('Thời Gian Đặt Chân', '%'),
            ('Chiều Cao Nâng Chân', 'cm'),
            ('Chiều Rộng Bước', 'cm')
        ]
        
        for param_name, unit in general_params:
            if param_name in findings:
                data = findings[param_name]
                measured = data.get('measured_value', 0)
                norm_mean = data.get('normative_mean', 0)
                norm_std = data.get('normative_std', 0)
                status = data.get('status', 'KHÔNG RÕ')
                color = get_status_color(status)
                
                # Tính độ lệch %
                if norm_mean > 0:
                    deviation_percent = abs(measured - norm_mean) / norm_mean * 100
                else:
                    deviation_percent = 0
                    
                html += f"""
                    <tr>
                        <td><strong>{param_name}</strong></td>
                        <td style="color: #0078d4; font-weight: bold;">{measured:.1f} {unit}</td>
                        <td>{norm_mean:.1f} ± {norm_std:.1f} {unit}</td>
                        <td style="color: {color}; font-weight: bold;">{deviation_percent:.1f}%</td>
                        <td style="color: {color}; font-weight: bold;">{status}</td>
                    </tr>
                """

        # Bỏ hàng Góc Hông khỏi THÔNG SỐ CHUNG theo yêu cầu
        
        html += """
                    </tbody>
                </table>
            </div>"""
        
        # Khuyến nghị
        recommendations = diagnosis.get('recommendations', [])
        if recommendations:
            html += """
                <div class="section">
                    <div class="section-title">💊 KHUYẾN NGHỊ ĐIỀU TRỊ</div>
                    <ul style="list-style-type: none; padding: 0;">"""
            for i, rec in enumerate(recommendations, 1):
                html += f"<li style='margin: 10px 0; padding: 10px; background: #f8f9fa; border-left: 4px solid #0078d4; border-radius: 4px;'><strong>{i}.</strong> {rec}</li>"
            html += "</ul></div>"
        
        # Lưu ý quan trọng
        html += f"""
            <div class="section" style="background: #fff3cd; border: 2px solid #ffc107;">
                <div class="section-title" style="color: #856404; border-bottom-color: #856404;">⚠️ LƯU Ý QUAN TRỌNG</div>
                <div style="color: #856404;">
                    <p><strong>📋 Kết quả này chỉ mang tính chất tham khảo</strong> và không thay thế cho việc khám bệnh chuyên khoa.</p>
                    <p><strong>👨‍⚕️ Vui lòng tham khảo ý kiến bác sĩ</strong> chuyên khoa cơ xương khớp hoặc phục hồi chức năng để được tư vấn và điều trị phù hợp.</p>
                    <p><strong>🔄 Nên thực hiện phân tích nhiều lần</strong> trong các điều kiện khác nhau để có kết quả chính xác nhất.</p>
                    <p><strong>📄 File dữ liệu:</strong> {data_file}</p>
                </div>
            </div>
        </body>
        </html>"""
        
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
            # Convert relative path to absolute path if needed
            if not os.path.isabs(filename):
                # Get the current working directory and join with filename
                current_dir = os.getcwd()
                absolute_path = os.path.join(current_dir, filename)
            else:
                absolute_path = filename
            
            # Check if file exists
            if not os.path.exists(absolute_path):
                QMessageBox.warning(self, "Lỗi", f"File không tồn tại: {absolute_path}")
                return
            
            if platform.system() == 'Windows':
                os.startfile(absolute_path)
            elif platform.system() == 'Darwin':  # macOS
                subprocess.call(['open', absolute_path])
            else:  # Linux
                subprocess.call(['xdg-open', absolute_path])
        except Exception as e:
            QMessageBox.warning(self, "Lỗi", f"Không thể mở file: {str(e)}")
    
    @pyqtSlot(str)
    def view_history_file(self, file_path):
        """View historical analysis file"""
        try:
            # Import GaitAnalyzer class to call static method directly
            from src.core.gait_analyzer import GaitAnalyzer
            
            # Load and diagnose the historical file using static method
            diagnosis = GaitAnalyzer.load_and_diagnose(file_path)
            
            if diagnosis:
                # Show the diagnosis dialog
                self.show_comprehensive_diagnosis(diagnosis, file_path)
            else:
                QMessageBox.warning(self, "Lỗi", "Không thể tải file lịch sử phân tích")
        except Exception as e:
            QMessageBox.warning(self, "Lỗi", f"Lỗi khi xem file lịch sử: {str(e)}")
    
    def validate_patient_info(self, patient_info):
        """Validate patient information"""
        errors = []
        
        # Check name
        name = patient_info.get("name", "").strip()
        if not name:
            errors.append("• Tên người đo không được để trống")
        
        # Check age
        age = patient_info.get("age", 0)
        if age < 1 or age > 120:
            errors.append("• Tuổi phải từ 1 đến 120")
        
        if errors:
            message = "Vui lòng nhập đầy đủ thông tin người đo:\n\n" + "\n".join(errors)
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
        self.camera_thread.stop()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
