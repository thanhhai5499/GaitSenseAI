"""
Dark theme styles for GaitSenseAI application
"""

DARK_THEME_STYLESHEET = """
/* Main Application */
QMainWindow {
    background-color: #1a1a1a;
    color: #ffffff;
}

/* Menu Bar */
QMenuBar {
    background-color: #2b2b2b;
    color: #ffffff;
    border-bottom: 1px solid #404040;
    font-size: 16px;
}

QMenuBar::item {
    background-color: transparent;
    padding: 8px 12px;
    margin: 2px;
    border-radius: 4px;
}

QMenuBar::item:selected {
    background-color: #404040;
}

QMenuBar::item:pressed {
    background-color: #505050;
}

/* Menu */
QMenu {
    background-color: #2b2b2b;
    color: #ffffff;
    border: 1px solid #404040;
    border-radius: 6px;
    padding: 4px;
}

QMenu::item {
    padding: 8px 25px;
    border-radius: 4px;
    margin: 2px;
}

QMenu::item:selected {
    background-color: #404040;
    color: #00ff88;
}

QMenu::separator {
    height: 1px;
    background-color: #404040;
    margin: 4px 8px;
}

/* Status Bar */
QStatusBar {
    background-color: #2b2b2b;
    color: #cccccc;
    border-top: 1px solid #404040;
    font-size: 16px;
    padding: 4px;
}

/* Splitter */
QSplitter::handle {
    background-color: #404040;
    width: 3px;
    margin: 0px;
}

QSplitter::handle:hover {
    background-color: #00ff88;
}

QSplitter::handle:pressed {
    background-color: #00cc6a;
}

/* Scroll Area */
QScrollArea {
    border: none;
    background-color: transparent;
}

QScrollBar:vertical {
    background-color: #2b2b2b;
    width: 12px;
    border-radius: 6px;
    margin: 0px;
}

QScrollBar::handle:vertical {
    background-color: #505050;
    border-radius: 6px;
    min-height: 20px;
    margin: 2px;
}

QScrollBar::handle:vertical:hover {
    background-color: #606060;
}

QScrollBar::handle:vertical:pressed {
    background-color: #707070;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
    background: none;
}

/* Group Box */
QGroupBox {
    font-weight: bold;
    font-size: 18px;
    border: 2px solid #404040;
    border-radius: 8px;
    margin-top: 1ex;
    color: #ffffff;
    background-color: #1e1e1e;
    padding-top: 10px;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 8px 0 8px;
    color: #00ff88;
    font-weight: bold;
    background-color: #1e1e1e;
}

/* Buttons */
QPushButton {
    background-color: #4CAF50;
    color: white;
    border: none;
    padding: 12px 20px;
    border-radius: 6px;
    font-size: 16px;
    font-weight: bold;
    min-width: 100px;
}

QPushButton:hover {
    background-color: #45a049;
    transform: translateY(-1px);
}

QPushButton:pressed {
    background-color: #3d8b40;
    transform: translateY(1px);
}

QPushButton:disabled {
    background-color: #555555;
    color: #888888;
}

/* Stop Button Style */
QPushButton[objectName="stop_button"] {
    background-color: #ff9800;
}

QPushButton[objectName="stop_button"]:hover {
    background-color: #e68900;
}

QPushButton[objectName="stop_button"]:pressed {
    background-color: #cc7700;
}

/* Reset Button Style */
QPushButton[objectName="reset_button"] {
    background-color: #f44336;
}

QPushButton[objectName="reset_button"]:hover {
    background-color: #da190b;
}

QPushButton[objectName="reset_button"]:pressed {
    background-color: #b71c1c;
}

/* Labels */
QLabel {
    color: #ffffff;
    font-size: 16px;
}

/* Frames */
QFrame {
    background-color: #2b2b2b;
    border: 1px solid #404040;
    border-radius: 8px;
}

/* Metric Cards */
QFrame[objectName="metric_card"] {
    background-color: #2b2b2b;
    border: 1px solid #404040;
    border-radius: 8px;
    padding: 8px;
    margin: 4px;
}

QFrame[objectName="metric_card"]:hover {
    border-color: #00ff88;
    background-color: #323232;
}

/* Value Labels in Metric Cards */
QLabel[objectName="metric_value"] {
    color: #00ff88;
    font-size: 20px;
    font-weight: bold;
}

QLabel[objectName="metric_title"] {
    color: #cccccc;
    font-size: 14px;
    font-weight: normal;
}

/* Camera Display */
QLabel[objectName="camera_display"] {
    background-color: #000000;
    border: 1px solid #404040;
    color: #cccccc;
    font-size: 18px;
}

/* Status Bar Elements */
QLabel[objectName="status_connected"] {
    color: #00ff88;
    font-weight: bold;
}

QLabel[objectName="status_disconnected"] {
    color: #ff4444;
    font-weight: bold;
}

QLabel[objectName="fps_display"] {
    color: #00ff88;
    font-family: monospace;
}

/* Overlay */
QWidget[objectName="overlay"] {
    background-color: transparent;
}

QLabel[objectName="overlay_info"] {
    color: #00ff88;
    background-color: rgba(0, 0, 0, 180);
    padding: 10px 15px;
    border-radius: 6px;
    font-weight: bold;
    font-size: 16px;
    border: 1px solid #00ff88;
}

/* Tool Tips */
QToolTip {
    background-color: #2b2b2b;
    color: #ffffff;
    border: 1px solid #404040;
    border-radius: 4px;
    padding: 10px;
    font-size: 14px;
}

/* Message Box */
QMessageBox {
    background-color: #2b2b2b;
    color: #ffffff;
}

QMessageBox QPushButton {
    min-width: 90px;
    padding: 8px 16px;
    font-size: 16px;
}
"""

# Color constants
COLORS = {
    'primary': '#00ff88',
    'secondary': '#404040',
    'background': '#1a1a1a',
    'surface': '#2b2b2b',
    'error': '#ff4444',
    'warning': '#ff9800',
    'success': '#4CAF50',
    'text_primary': '#ffffff',
    'text_secondary': '#cccccc',
    'text_disabled': '#888888'
}

# Font settings - Tăng kích thước cho dễ đọc
FONTS = {
    'default_size': 16,     # Tăng từ 12 lên 16
    'title_size': 22,       # Tăng từ 16 lên 22
    'small_size': 14,       # Tăng từ 10 lên 14
    'large_size': 24,       # Tăng từ 18 lên 24
    'metric_value_size': 20 # Tăng từ 16 lên 20
}

