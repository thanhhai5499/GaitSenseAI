"""
Utility functions and helpers for GaitSenseAI
"""

import os
import json
import csv
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
from PyQt6.QtCore import QStandardPaths
from PyQt6.QtGui import QPixmap, QImage
import cv2


def get_app_data_dir() -> str:
    """Get application data directory"""
    app_data = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppDataLocation)
    app_data_dir = os.path.join(app_data, "GaitSenseAI")
    os.makedirs(app_data_dir, exist_ok=True)
    return app_data_dir


def get_timestamp() -> str:
    """Get current timestamp string"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_analysis_data(data: Dict[str, Any], filename: Optional[str] = None) -> str:
    """Save analysis data to JSON file"""
    if filename is None:
        filename = f"gait_analysis_{get_timestamp()}.json"
    
    filepath = os.path.join(get_app_data_dir(), filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    return filepath


def load_analysis_data(filepath: str) -> Dict[str, Any]:
    """Load analysis data from JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def export_metrics_to_csv(metrics_history: List[Dict], filename: Optional[str] = None) -> str:
    """Export metrics history to CSV file"""
    if not metrics_history:
        raise ValueError("No metrics data to export")
    
    if filename is None:
        filename = f"gait_metrics_{get_timestamp()}.csv"
    
    filepath = os.path.join(get_app_data_dir(), filename)
    
    # Get all unique keys from metrics
    all_keys = set()
    for metrics in metrics_history:
        all_keys.update(metrics.keys())
    
    fieldnames = ['timestamp'] + sorted(all_keys)
    
    with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for i, metrics in enumerate(metrics_history):
            row = {'timestamp': i}
            row.update(metrics)
            writer.writerow(row)
    
    return filepath


def calculate_statistics(values: List[float]) -> Dict[str, float]:
    """Calculate basic statistics for a list of values"""
    if not values:
        return {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'median': 0}
    
    arr = np.array(values)
    return {
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr)),
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
        'median': float(np.median(arr))
    }


def normalize_angle(angle: float) -> float:
    """Normalize angle to -180 to 180 degrees range"""
    while angle > 180:
        angle -= 360
    while angle < -180:
        angle += 360
    return angle


def calculate_distance(point1: Dict[str, float], point2: Dict[str, float]) -> float:
    """Calculate Euclidean distance between two points"""
    if 'x' not in point1 or 'y' not in point1 or 'x' not in point2 or 'y' not in point2:
        return 0.0
    
    dx = point1['x'] - point2['x']
    dy = point1['y'] - point2['y']
    return np.sqrt(dx*dx + dy*dy)


def calculate_angle(point1: Dict[str, float], point2: Dict[str, float], point3: Dict[str, float]) -> float:
    """Calculate angle between three points (point2 is the vertex)"""
    if not all('x' in p and 'y' in p for p in [point1, point2, point3]):
        return 0.0
    
    # Vector from point2 to point1
    v1 = np.array([point1['x'] - point2['x'], point1['y'] - point2['y']])
    # Vector from point2 to point3
    v2 = np.array([point3['x'] - point2['x'], point3['y'] - point2['y']])
    
    # Calculate angle between vectors
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Ensure valid range for arccos
    angle = np.arccos(cos_angle)
    
    return np.degrees(angle)


def smooth_signal(signal: List[float], window_size: int = 5) -> List[float]:
    """Apply moving average smoothing to signal"""
    if len(signal) < window_size:
        return signal
    
    smoothed = []
    for i in range(len(signal)):
        start = max(0, i - window_size // 2)
        end = min(len(signal), i + window_size // 2 + 1)
        smoothed.append(np.mean(signal[start:end]))
    
    return smoothed


def detect_peaks_simple(signal: List[float], min_height: Optional[float] = None, 
                       min_distance: int = 1) -> List[int]:
    """Simple peak detection algorithm"""
    if len(signal) < 3:
        return []
    
    peaks = []
    
    for i in range(1, len(signal) - 1):
        # Check if current point is higher than neighbors
        if signal[i] > signal[i-1] and signal[i] > signal[i+1]:
            # Check minimum height requirement
            if min_height is None or signal[i] >= min_height:
                # Check minimum distance from previous peak
                if not peaks or i - peaks[-1] >= min_distance:
                    peaks.append(i)
    
    return peaks


def frame_to_base64(frame: np.ndarray) -> str:
    """Convert OpenCV frame to base64 string"""
    import base64
    
    # Encode frame to JPEG
    _, buffer = cv2.imencode('.jpg', frame)
    # Convert to base64
    frame_base64 = base64.b64encode(buffer).decode('utf-8')
    return frame_base64


def base64_to_frame(base64_string: str) -> np.ndarray:
    """Convert base64 string to OpenCV frame"""
    import base64
    
    # Decode base64
    image_data = base64.b64decode(base64_string)
    # Convert to numpy array
    nparr = np.frombuffer(image_data, np.uint8)
    # Decode image
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return frame


def validate_camera_settings(width: int, height: int, fps: int) -> bool:
    """Validate camera settings"""
    # Common resolutions
    valid_resolutions = [
        (640, 480), (800, 600), (1024, 768), (1280, 720),
        (1280, 960), (1600, 1200), (1920, 1080), (2560, 1440)
    ]
    
    # Check resolution
    if (width, height) not in valid_resolutions:
        return False
    
    # Check FPS (common values)
    if fps not in [15, 24, 30, 60, 120]:
        return False
    
    return True


def create_error_image(width: int, height: int, message: str) -> QPixmap:
    """Create an error image with text"""
    # Create black image
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Add text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (255, 255, 255)
    thickness = 2
    
    # Calculate text size and position
    text_size = cv2.getTextSize(message, font, font_scale, thickness)[0]
    text_x = (width - text_size[0]) // 2
    text_y = (height + text_size[1]) // 2
    
    cv2.putText(image, message, (text_x, text_y), font, font_scale, color, thickness)
    
    # Convert to QPixmap
    height, width, channel = image.shape
    bytes_per_line = 3 * width
    q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
    q_image = q_image.rgbSwapped()  # Convert BGR to RGB
    
    return QPixmap.fromImage(q_image)


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human readable format"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.1f}s"


def clamp(value: float, min_value: float, max_value: float) -> float:
    """Clamp value between min and max"""
    return max(min_value, min(max_value, value))

