"""
Configuration settings for GaitSenseAI application
"""

# Camera settings
CAMERA_SETTINGS = {
    "width": 1280,      # Brio300 tương thích tốt với 1280x720
    "height": 720,
    "fps": 30,          # 30fps cho ổn định
    "device_id": 0      # Camera ID được phát hiện
}

# Pose detection settings
POSE_SETTINGS = {
    "model_path": "models/pose/body_25",
    "confidence_threshold": 0.5,
    "min_detection_confidence": 0.7,
    "min_tracking_confidence": 0.5
}

# UI settings
UI_SETTINGS = {
    "window_width": 1400,
    "window_height": 900,
    "sidebar_width": 350,
    "dark_theme": True
}

# Gait analysis settings
GAIT_ANALYSIS = {
    "sampling_rate": 60,  # Hz
    "step_detection_threshold": 0.3,
    "stride_length_estimation": True,
    "cadence_calculation": True
}

# Application settings
APP_SETTINGS = {
    "title": "GaitSenseAI - Gait Analysis System",
    "version": "1.0.0",
    "auto_save": True,
    "export_formats": ["csv", "json", "xlsx"]
}

