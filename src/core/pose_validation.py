"""
Pose Validation Module - Kiểm tra độ chính xác số liệu pose vs realtime khi camera cách 2m
"""

import cv2
import numpy as np
import time
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
from collections import deque


@dataclass
class ValidationMetrics:
    """Metrics để so sánh pose data với realtime data"""
    timestamp: float
    pose_stride_length: float = 0.0
    realtime_stride_length: float = 0.0
    pose_speed: float = 0.0
    realtime_speed: float = 0.0
    pose_cadence: float = 0.0
    realtime_cadence: float = 0.0
    pose_knee_angle_left: float = 0.0
    pose_knee_angle_right: float = 0.0
    distance_error: float = 0.0
    speed_error: float = 0.0
    angle_error: float = 0.0
    confidence_score: float = 0.0


class CameraDistanceCalibrator:
    """Hiệu chỉnh cho camera đặt cách 2 mét"""
    
    def __init__(self, camera_distance_meters=2.0):
        self.camera_distance = camera_distance_meters
        self.pixel_to_meter_ratio = None
        self.calibration_reference = None
        self.frame_width = None
        self.frame_height = None
        
    def calibrate_with_reference_object(self, frame, known_height_meters=1.7):
        """
        Hiệu chỉnh tỷ lệ pixel/mét bằng chiều cao người (1.7m trung bình)
        Cải thiện để đo chính xác hơn với camera 2m
        """
        self.frame_height, self.frame_width = frame.shape[:2]
        
        # Improved calibration for 2m distance
        # Camera 2m away, person height 1.7m -> should appear as specific pixel height
        
        # Calculate based on camera geometry
        camera_fov_vertical = 60  # degrees (typical webcam)
        distance_to_person = self.camera_distance
        
        # Real height in frame based on perspective
        real_height_in_frame = 2 * distance_to_person * np.tan(np.radians(camera_fov_vertical/2))
        pixels_per_meter_theoretical = self.frame_height / real_height_in_frame
        
        # Practical adjustment for 2m distance
        # Person typically occupies 50-70% of frame height at 2m
        person_frame_ratio = 0.65  # 65% for 2m distance
        estimated_person_height_pixels = self.frame_height * person_frame_ratio
        
        # Use more conservative ratio to avoid under-measurement
        self.pixel_to_meter_ratio = estimated_person_height_pixels / known_height_meters
        
        # Apply correction factor for foot height measurements
        self.height_correction_factor = 1.8  # Increase measurements by 80%
        
        print(f"📏 Enhanced camera calibration for 2m distance:")
        print(f"   Frame size: {self.frame_width}x{self.frame_height}")
        print(f"   Theoretical pixels/meter: {pixels_per_meter_theoretical:.2f}")
        print(f"   Practical pixel/meter ratio: {self.pixel_to_meter_ratio:.2f}")
        print(f"   Height correction factor: {self.height_correction_factor:.1f}")
        print(f"   Expected person height: {estimated_person_height_pixels:.1f} pixels")
        
        return True
    
    def pixel_to_meters(self, pixel_distance):
        """Chuyển đổi pixel sang mét có bù perspective - cải thiện cho foot height"""
        if self.pixel_to_meter_ratio is None:
            return pixel_distance * 0.001  # Fallback estimate
        
        meters = pixel_distance / self.pixel_to_meter_ratio
        
        # Enhanced perspective correction cho camera cách 2m
        # Camera nhìn từ góc 15-30 độ từ trên xuống
        perspective_correction = 1.25  # Base correction
        
        # Additional height-specific correction
        if hasattr(self, 'height_correction_factor'):
            meters *= self.height_correction_factor
        
        return meters * perspective_correction
    
    def get_angle_correction_factor(self, y_position):
        """Hệ số hiệu chỉnh góc dựa trên vị trí Y trong frame"""
        if self.frame_height is None:
            return 1.0
        
        # Vị trí càng gần đáy frame (người xa camera) càng cần hiệu chỉnh nhiều
        normalized_y = y_position / self.frame_height
        correction_factor = 1.0 + (normalized_y * 0.3)  # Tăng 30% ở đáy frame
        
        return correction_factor


class PoseValidator:
    """Kiểm tra tính chính xác của pose detection so với realtime measurements"""
    
    def __init__(self, camera_distance=2.0):
        self.calibrator = CameraDistanceCalibrator(camera_distance)
        self.validation_history = deque(maxlen=1000)
        self.pose_measurements = deque(maxlen=100)
        self.realtime_measurements = deque(maxlen=100)
        
        # Thresholds cho validation
        self.distance_threshold = 0.1  # 10cm
        self.speed_threshold = 0.2     # 0.2 m/s
        self.angle_threshold = 10.0    # 10 degrees
        
        # Tracking variables
        self.last_pose_positions = None
        self.last_realtime_positions = None
        self.step_count = 0
        
    def validate_pose_accuracy(self, frame, pose_keypoints, realtime_data=None):
        """
        So sánh pose data với realtime measurements
        """
        timestamp = time.time()
        
        if pose_keypoints is None or len(pose_keypoints) < 25:
            return None
        
        # Calibrate nếu chưa có
        if self.calibrator.pixel_to_meter_ratio is None:
            self.calibrator.calibrate_with_reference_object(frame)
        
        # Extract pose measurements
        pose_metrics = self._extract_pose_metrics(pose_keypoints, timestamp)
        
        # Extract realtime measurements (nếu có)
        realtime_metrics = self._extract_realtime_metrics(realtime_data, timestamp)
        
        # So sánh và tạo validation metrics
        validation = self._compare_measurements(pose_metrics, realtime_metrics, timestamp)
        
        # Lưu vào history
        if validation:
            self.validation_history.append(validation)
        
        return validation
    
    def _extract_pose_metrics(self, keypoints, timestamp):
        """Trích xuất metrics từ pose keypoints"""
        metrics = {}
        
        # Lấy key points
        left_ankle = keypoints[14]   # LAnkle
        right_ankle = keypoints[11]  # RAnkle
        left_knee = keypoints[13]    # LKnee
        right_knee = keypoints[10]   # RKnee
        left_hip = keypoints[12]     # LHip
        right_hip = keypoints[9]     # RHip
        
        # Kiểm tra validity
        if all(kp[2] > 0.1 for kp in [left_ankle, right_ankle, left_knee, right_knee]):
            
            # 1. Stride length từ pose
            if self.last_pose_positions:
                current_center_x = (left_ankle[0] + right_ankle[0]) / 2
                last_center_x = self.last_pose_positions
                
                pixel_distance = abs(current_center_x - last_center_x)
                stride_length = self.calibrator.pixel_to_meters(pixel_distance)
                metrics['stride_length'] = stride_length
            
            # 2. Speed từ pose (cần timestamp previous)
            if hasattr(self, 'last_pose_timestamp') and 'stride_length' in metrics:
                time_diff = timestamp - self.last_pose_timestamp
                if time_diff > 0:
                    metrics['speed'] = metrics['stride_length'] / time_diff
            
            # 3. Knee angles
            metrics['knee_angle_left'] = self._calculate_knee_angle(left_hip, left_knee, left_ankle)
            metrics['knee_angle_right'] = self._calculate_knee_angle(right_hip, right_knee, right_ankle)
            
            # 4. Cadence (nếu detect được step)
            if self._detect_step_from_pose(keypoints):
                self.step_count += 1
                if hasattr(self, 'first_step_time'):
                    time_elapsed = timestamp - self.first_step_time
                    if time_elapsed > 0:
                        metrics['cadence'] = (self.step_count / time_elapsed) * 60  # steps/minute
                else:
                    self.first_step_time = timestamp
            
            # Update last positions
            self.last_pose_positions = (left_ankle[0] + right_ankle[0]) / 2
            self.last_pose_timestamp = timestamp
        
        return metrics
    
    def _extract_realtime_metrics(self, realtime_data, timestamp):
        """Trích xuất metrics từ realtime data (nếu có)"""
        if realtime_data is None:
            return {}
        
        # Giả sử realtime_data có format:
        # {'stride_length': float, 'speed': float, 'cadence': float, ...}
        return realtime_data
    
    def _calculate_knee_angle(self, hip, knee, ankle):
        """Tính góc knee từ 3 điểm hip-knee-ankle"""
        if any(point[2] < 0.1 for point in [hip, knee, ankle]):
            return 0.0
        
        # Vector from knee to hip
        v1 = np.array([hip[0] - knee[0], hip[1] - knee[1]])
        # Vector from knee to ankle  
        v2 = np.array([ankle[0] - knee[0], ankle[1] - knee[1]])
        
        # Tính góc
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle) * 180 / np.pi
        
        return angle
    
    def _detect_step_from_pose(self, keypoints):
        """Detect step từ pose data"""
        left_ankle = keypoints[14]
        right_ankle = keypoints[11]
        
        if left_ankle[2] < 0.1 or right_ankle[2] < 0.1:
            return False
        
        # Simple step detection: ankle height change
        ankle_height_diff = abs(left_ankle[1] - right_ankle[1])
        
        if not hasattr(self, 'last_ankle_height_diff'):
            self.last_ankle_height_diff = ankle_height_diff
            return False
        
        # Detect peak in height difference
        if (self.last_ankle_height_diff < ankle_height_diff and 
            ankle_height_diff > 20):  # 20 pixel threshold
            self.last_ankle_height_diff = ankle_height_diff
            return True
        
        self.last_ankle_height_diff = ankle_height_diff
        return False
    
    def _compare_measurements(self, pose_metrics, realtime_metrics, timestamp):
        """So sánh pose vs realtime measurements"""
        if not pose_metrics:
            return None
        
        validation = ValidationMetrics(timestamp=timestamp)
        
        # So sánh stride length
        if 'stride_length' in pose_metrics:
            validation.pose_stride_length = pose_metrics['stride_length']
            if 'stride_length' in realtime_metrics:
                validation.realtime_stride_length = realtime_metrics['stride_length']
                validation.distance_error = abs(validation.pose_stride_length - validation.realtime_stride_length)
        
        # So sánh speed
        if 'speed' in pose_metrics:
            validation.pose_speed = pose_metrics['speed']
            if 'speed' in realtime_metrics:
                validation.realtime_speed = realtime_metrics['speed']
                validation.speed_error = abs(validation.pose_speed - validation.realtime_speed)
        
        # So sánh cadence
        if 'cadence' in pose_metrics:
            validation.pose_cadence = pose_metrics['cadence']
            if 'cadence' in realtime_metrics:
                validation.realtime_cadence = realtime_metrics['cadence']
        
        # So sánh knee angles
        if 'knee_angle_left' in pose_metrics:
            validation.pose_knee_angle_left = pose_metrics['knee_angle_left']
        if 'knee_angle_right' in pose_metrics:
            validation.pose_knee_angle_right = pose_metrics['knee_angle_right']
        
        # Tính confidence score
        validation.confidence_score = self._calculate_confidence_score(validation)
        
        return validation
    
    def _calculate_confidence_score(self, validation):
        """Tính confidence score dựa trên độ chênh lệch"""
        score = 100.0  # Start with perfect score
        
        # Penalize based on errors
        if validation.distance_error > 0:
            distance_penalty = min(50, validation.distance_error * 100)  # Max 50 points
            score -= distance_penalty
        
        if validation.speed_error > 0:
            speed_penalty = min(30, validation.speed_error * 50)  # Max 30 points
            score -= speed_penalty
        
        if validation.angle_error > 0:
            angle_penalty = min(20, validation.angle_error)  # Max 20 points
            score -= angle_penalty
        
        return max(0, score)
    
    def get_validation_summary(self, last_n=50):
        """Lấy summary của validation trong N measurements gần nhất"""
        if not self.validation_history:
            return None
        
        recent_validations = list(self.validation_history)[-last_n:]
        
        summary = {
            'total_measurements': len(recent_validations),
            'avg_confidence': np.mean([v.confidence_score for v in recent_validations]),
            'avg_distance_error': np.mean([v.distance_error for v in recent_validations if v.distance_error > 0]),
            'avg_speed_error': np.mean([v.speed_error for v in recent_validations if v.speed_error > 0]),
            'accuracy_percentage': len([v for v in recent_validations if v.confidence_score > 80]) / len(recent_validations) * 100,
            'last_update': time.strftime('%H:%M:%S', time.localtime(recent_validations[-1].timestamp))
        }
        
        return summary
    
    def draw_validation_overlay(self, frame, current_validation=None):
        """Vẽ thông tin validation lên frame"""
        if not self.validation_history and current_validation is None:
            return frame
        
        overlay = frame.copy()
        
        # Background cho text
        cv2.rectangle(overlay, (10, 10), (400, 200), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        y_offset = 35
        line_height = 25
        
        # Title
        cv2.putText(frame, "POSE VALIDATION (2m distance)", (15, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        y_offset += line_height
        
        # Current validation
        if current_validation:
            confidence = current_validation.confidence_score
            color = (0, 255, 0) if confidence > 80 else (0, 255, 255) if confidence > 60 else (0, 0, 255)
            
            cv2.putText(frame, f"Confidence: {confidence:.1f}%", (15, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_offset += line_height
            
            if current_validation.pose_stride_length > 0:
                cv2.putText(frame, f"Stride: {current_validation.pose_stride_length:.2f}m", (15, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += line_height
            
            if current_validation.pose_speed > 0:
                cv2.putText(frame, f"Speed: {current_validation.pose_speed:.2f}m/s", (15, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += line_height
        
        # Summary statistics
        summary = self.get_validation_summary()
        if summary:
            cv2.putText(frame, f"Accuracy: {summary['accuracy_percentage']:.1f}%", (15, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += line_height
            
            cv2.putText(frame, f"Avg Error: {summary['avg_distance_error']:.3f}m", (15, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def save_validation_report(self, filename="validation_report.json"):
        """Lưu validation report"""
        if not self.validation_history:
            return False
        
        report_data = {
            'camera_distance_meters': self.calibrator.camera_distance,
            'pixel_to_meter_ratio': self.calibrator.pixel_to_meter_ratio,
            'total_validations': len(self.validation_history),
            'summary': self.get_validation_summary(),
            'detailed_data': [asdict(v) for v in self.validation_history]
        }
        
        with open(filename, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"✅ Validation report saved to {filename}")
        return True
