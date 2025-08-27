"""
Gait analysis module for detecting and analyzing walking patterns
"""

import cv2
import numpy as np
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.signal import find_peaks
from config.settings import POSE_SETTINGS, GAIT_ANALYSIS


@dataclass
class GaitMetrics:
    """Data class for professional gait analysis metrics"""
    # Basic metrics
    step_count: int = 0
    cadence: float = 0.0  # steps per minute
    stride_length: float = 0.0  # estimated in pixels
    step_time: float = 0.0  # average step time in seconds
    stance_time: float = 0.0  # stance phase duration
    swing_time: float = 0.0  # swing phase duration
    walking_speed: float = 0.0  # estimated walking speed
    left_step_length: float = 0.0
    right_step_length: float = 0.0
    step_width: float = 0.0
    foot_angle_left: float = 0.0
    foot_angle_right: float = 0.0
    
    # Professional gait analysis metrics
    # 1. Joint Angles
    knee_angle_left: float = 0.0
    knee_angle_right: float = 0.0
    hip_angle_left: float = 0.0 
    hip_angle_right: float = 0.0
    ankle_angle_left: float = 0.0
    ankle_angle_right: float = 0.0
    
    # 2. Position & Distance
    foot_clearance_left: float = 0.0
    foot_clearance_right: float = 0.0
    step_length_current: float = 0.0
    step_width_current: float = 0.0
    
    # 3. Timing
    stance_phase_percent: float = 0.0
    swing_phase_percent: float = 0.0
    
    # 4. Symmetry
    symmetry_index: float = 0.0


class OpenPosePoseDetector:
    """Pose detection using OpenPose Body_25 model"""
    
    def __init__(self):
        self.net = None
        self.setup_openpose()
        
        # Body_25 keypoint indices (as shown in your image)
        self.body_25_keypoints = {
            0: "Nose",
            1: "Neck", 
            2: "RShoulder",
            3: "RElbow",
            4: "RWrist",
            5: "LShoulder", 
            6: "LElbow",
            7: "LWrist",
            8: "MidHip",
            9: "RHip",
            10: "RKnee",
            11: "RAnkle",
            12: "LHip",
            13: "LKnee",
            14: "LAnkle",
            15: "REye",
            16: "LEye",
            17: "REar",
            18: "LEar",
            19: "LBigToe",
            20: "LSmallToe", 
            21: "LHeel",
            22: "RBigToe",
            23: "RSmallToe",
            24: "RHeel"
        }
        
        # Body_25 connections - Only lower body (from hip down)
        self.body_25_connections = [
            # Hip to legs connection
            (8, 9), (8, 12),  # MidHip to RHip and LHip
            # Right leg
            (9, 10), (10, 11),  # RHip -> RKnee -> RAnkle
            # Left leg
            (12, 13), (13, 14),  # LHip -> LKnee -> LAnkle
            # Feet
            (11, 22), (11, 24), (22, 23),  # Right foot
            (14, 19), (14, 21), (19, 20)   # Left foot
        ]
        
        # Key lower body keypoints for detection validation
        self.lower_body_keypoints = [8, 9, 10, 11, 12, 13, 14]  # MidHip, RHip, RKnee, RAnkle, LHip, LKnee, LAnkle
        
        # Track consecutive failed leg detections
        self.failed_leg_detection_count = 0
        self.max_failed_detections = 30  # Stop after 30 consecutive failures (~3 seconds at 10 FPS)
        
    def setup_openpose(self):
        """Setup OpenPose network"""
        try:
            # Paths to model files
            model_path = POSE_SETTINGS["model_path"]
            prototxt_path = os.path.join(model_path, "pose_deploy.prototxt")
            weights_path = os.path.join(model_path, "pose_iter_584000.caffemodel")
            
            print(f"🔍 Loading OpenPose model from: {model_path}")
            print(f"   Prototxt: {prototxt_path}")
            print(f"   Weights: {weights_path}")
            
            if not os.path.exists(prototxt_path):
                raise FileNotFoundError(f"Prototxt not found: {prototxt_path}")
            if not os.path.exists(weights_path):
                raise FileNotFoundError(f"Weights not found: {weights_path}")
            
            # Load the network
            self.net = cv2.dnn.readNetFromCaffe(prototxt_path, weights_path)
            print("✅ OpenPose model loaded successfully")
            
        except Exception as e:
            print(f"❌ Failed to load OpenPose model: {e}")
            print("📝 Using fallback pose detection...")
            self.net = None
        
    def detect_pose(self, frame):
        """Detect pose landmarks using OpenPose"""
        if self.net is None:
            return None
            
        try:
            height, width = frame.shape[:2]
            
            # Prepare input blob - smaller size for better performance
            input_blob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (192, 192), (0, 0, 0), swapRB=False, crop=False)
            self.net.setInput(input_blob)
            
            # Forward pass
            output = self.net.forward()
            
            # Parse keypoints - improved detection
            keypoints = []
            H, W = output.shape[2], output.shape[3]
            
            for i in range(min(25, output.shape[1])):  # Body_25 has 25 keypoints
                # Get heatmap for keypoint i
                prob_map = output[0, i, :, :]
                _, confidence, _, point = cv2.minMaxLoc(prob_map)
                
                # Scale coordinates back to original image size
                x = int((width * point[0]) / W)
                y = int((height * point[1]) / H)
                
                # More lenient threshold for better detection
                if confidence > 0.1:  # Lower threshold
                    keypoints.append((x, y, confidence))
                else:
                    keypoints.append((0, 0, 0))  # Invalid keypoint
            
            # Pad to 25 keypoints if less detected
            while len(keypoints) < 25:
                keypoints.append((0, 0, 0))
            
            return keypoints
            
        except Exception as e:
            print(f"❌ Pose detection error: {e}")
            return None
    
    def draw_pose(self, frame, keypoints):
        """Draw pose skeleton on frame with Body_25 style - optimized"""
        if keypoints is None or len(keypoints) != 25:
            return frame
        
        # Check if any valid keypoints exist
        valid_points = [(i, kp) for i, kp in enumerate(keypoints) if kp[2] > 0.1]
        if not valid_points:
            return frame
        
        # Minimal logging for performance
        
        # Body_25 standard colors (BGR format for OpenCV)
        colors = {
            'head': (255, 0, 255),      # Magenta for head/face
            'neck_torso': (255, 255, 0), # Cyan for neck/torso  
            'right_arm': (0, 0, 255),   # Red for right arm
            'left_arm': (0, 255, 255),  # Yellow for left arm
            'right_leg': (0, 255, 0),   # Green for right leg
            'left_leg': (0, 165, 255),  # Orange for left leg
        }
        
        # Body_25 keypoint groups for coloring
        keypoint_colors = {
            0: colors['neck_torso'],    # Nose
            1: colors['neck_torso'],    # Neck
            2: colors['right_arm'],     # RShoulder
            3: colors['right_arm'],     # RElbow
            4: colors['right_arm'],     # RWrist
            5: colors['left_arm'],      # LShoulder
            6: colors['left_arm'],      # LElbow
            7: colors['left_arm'],      # LWrist
            8: colors['neck_torso'],    # MidHip
            9: colors['right_leg'],     # RHip
            10: colors['right_leg'],    # RKnee
            11: colors['right_leg'],    # RAnkle
            12: colors['left_leg'],     # LHip
            13: colors['left_leg'],     # LKnee
            14: colors['left_leg'],     # LAnkle
            15: colors['head'],         # REye
            16: colors['head'],         # LEye
            17: colors['head'],         # REar
            18: colors['head'],         # LEar
            19: colors['left_leg'],     # LBigToe
            20: colors['left_leg'],     # LSmallToe
            21: colors['left_leg'],     # LHeel
            22: colors['right_leg'],    # RBigToe
            23: colors['right_leg'],    # RSmallToe
            24: colors['right_leg'],    # RHeel
        }
        
        # Essential connections only for performance
        essential_connections = [
            # Main body structure
            (1, 8),   # Neck to MidHip
            (1, 2), (1, 5),  # Neck to shoulders
            # Arms
            (2, 3), (3, 4),  # Right arm
            (5, 6), (6, 7),  # Left arm
            # Legs  
            (8, 9), (9, 10), (10, 11),  # Right leg
            (8, 12), (12, 13), (13, 14),  # Left leg
        ]
        
        # Draw connections (bones) - essential only
        for connection in essential_connections:
            pt1_idx, pt2_idx = connection
            if pt1_idx < len(keypoints) and pt2_idx < len(keypoints):
                pt1 = keypoints[pt1_idx]
                pt2 = keypoints[pt2_idx]
                
                # Only draw if both points are valid
                if pt1[2] > 0.1 and pt2[2] > 0.1:
                    # Quick color assignment
                    if pt1_idx in [2, 3, 4] or pt2_idx in [2, 3, 4]:
                        color = colors['right_arm']
                    elif pt1_idx in [5, 6, 7] or pt2_idx in [5, 6, 7]:
                        color = colors['left_arm']
                    elif pt1_idx in [9, 10, 11] or pt2_idx in [9, 10, 11]:
                        color = colors['right_leg']
                    elif pt1_idx in [12, 13, 14] or pt2_idx in [12, 13, 14]:
                        color = colors['left_leg']
                    else:
                        color = colors['neck_torso']
                    
                    cv2.line(frame, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), color, 6)
        
        # Draw all valid keypoints with Body_25 colors
        # Draw keypoints - only lower body (8-14 and feet 19-24)
        lower_body_indices = [8, 9, 10, 11, 12, 13, 14, 19, 20, 21, 22, 23, 24]
        for i in lower_body_indices:
            if i < len(keypoints):
                point = keypoints[i]
                if point[2] > 0.1:  # Valid keypoint
                    x, y, confidence = point
                    
                    # Get color for this keypoint
                    color = keypoint_colors.get(i, (255, 255, 255))  # Default white
                    
                    # Draw BIGGER circle for keypoint
                    cv2.circle(frame, (int(x), int(y)), 12, color, -1)
                    
                    # Draw keypoint number in white with larger font
                    cv2.putText(frame, str(i), (int(x) - 15, int(y) - 15), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
        

        
        # Add info overlay with bigger text
        cv2.putText(frame, f"POSE: {len(valid_points)}/25", (10, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        
        return frame


class GaitAnalyzer:
    """Main gait analysis class with OpenPose Body_25 - Professional Edition"""
    
    def __init__(self):
        self.pose_detector = OpenPosePoseDetector()
        self.landmark_history = []
        self.metrics = GaitMetrics()
        self.frame_count = 0
        self.fps = GAIT_ANALYSIS["sampling_rate"]
        
        # Bilateral comparison tracking
        self.left_joint_angles = {"knee": [], "hip": [], "ankle": []}
        self.right_joint_angles = {"knee": [], "hip": [], "ankle": []}
        
        # Leg detection tracking
        self.failed_leg_detection_count = 0
        self.max_failed_detections = 30  # Stop after 30 consecutive failures (~3 seconds at 10 FPS)
        
        # Key landmarks for gait analysis using Body_25 indices
        self.key_landmarks = {
            'left_ankle': 14,    # LAnkle
            'right_ankle': 11,   # RAnkle
            'left_knee': 13,     # LKnee
            'right_knee': 10,    # RKnee
            'left_hip': 12,      # LHip
            'right_hip': 9,      # RHip
            'left_heel': 21,     # LHeel
            'right_heel': 24,    # RHeel
            'left_toe': 19,      # LBigToe
            'right_toe': 22,     # RBigToe
            'neck': 1,           # Neck
            'mid_hip': 8,        # MidHip
            'left_shoulder': 5,  # LShoulder
            'right_shoulder': 2, # RShoulder
        }
    
    def calculate_angle(self, p1, p2, p3):
        """Calculate angle between three points (p1-p2-p3)"""
        try:
            # Vector from p2 to p1
            v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
            # Vector from p2 to p3  
            v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
            
            # Calculate angle using dot product
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Ensure valid range
            angle = np.degrees(np.arccos(cos_angle))
            
            return angle
        except:
            return 0.0
    
    def calculate_distance(self, p1, p2):
        """Calculate Euclidean distance between two points"""
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def analyze_joint_angles(self, keypoints):
        """Analyze joint angles from pose keypoints"""
        angles = {}
        
        # Extract key points with confidence check
        points = {}
        for name, idx in self.key_landmarks.items():
            if idx < len(keypoints) and keypoints[idx][2] > 0.1:
                points[name] = keypoints[idx][:2]  # x, y only
        
        # Calculate joint angles if points are available
        # 1. Knee Angles (hip-knee-ankle)
        if all(k in points for k in ['left_hip', 'left_knee', 'left_ankle']):
            angles['knee_left'] = self.calculate_angle(
                points['left_hip'], points['left_knee'], points['left_ankle']
            )
        
        if all(k in points for k in ['right_hip', 'right_knee', 'right_ankle']):
            angles['knee_right'] = self.calculate_angle(
                points['right_hip'], points['right_knee'], points['right_ankle']
            )
        
        # 2. Hip Angles (shoulder-hip-knee) 
        if all(k in points for k in ['left_shoulder', 'left_hip', 'left_knee']):
            angles['hip_left'] = self.calculate_angle(
                points['left_shoulder'], points['left_hip'], points['left_knee']
            )
        
        if all(k in points for k in ['right_shoulder', 'right_hip', 'right_knee']):
            angles['hip_right'] = self.calculate_angle(
                points['right_shoulder'], points['right_hip'], points['right_knee']
            )
        
        # 3. Ankle Angles (knee-ankle-toe)
        if all(k in points for k in ['left_knee', 'left_ankle', 'left_toe']):
            angles['ankle_left'] = self.calculate_angle(
                points['left_knee'], points['left_ankle'], points['left_toe']
            )
        
        if all(k in points for k in ['right_knee', 'right_ankle', 'right_toe']):
            angles['ankle_right'] = self.calculate_angle(
                points['right_knee'], points['right_ankle'], points['right_toe']
            )
        
        return angles
    
    def analyze_distances(self, keypoints):
        """Analyze distances and clearances"""
        distances = {}
        
        # Extract key points
        points = {}
        for name, idx in self.key_landmarks.items():
            if idx < len(keypoints) and keypoints[idx][2] > 0.1:
                points[name] = keypoints[idx][:2]
        
        # Calculate step width (distance between ankles)
        if 'left_ankle' in points and 'right_ankle' in points:
            distances['step_width'] = self.calculate_distance(
                points['left_ankle'], points['right_ankle']
            )
        
        # Calculate foot clearance (approximate - ankle height)
        # Note: This is simplified - in real application would need ground plane reference
        if 'left_ankle' in points:
            distances['foot_clearance_left'] = points['left_ankle'][1]  # Y coordinate (higher = lower foot)
        
        if 'right_ankle' in points:
            distances['foot_clearance_right'] = points['right_ankle'][1]
        
        return distances
        
    def analyze_frame(self, frame):
        """Analyze a single frame for professional gait patterns using OpenPose"""
        keypoints = self.pose_detector.detect_pose(frame)

        if keypoints is not None:
            # Check if leg detection is sufficient
            legs_detected = self.check_leg_detection(keypoints)
            
            if legs_detected:
                # Reset failed counter if legs are detected
                self.failed_leg_detection_count = 0
                
                # Professional analysis
                self._analyze_professional_metrics(keypoints)

                # Traditional analysis for compatibility
                landmarks = self._extract_landmarks_from_keypoints(keypoints, frame.shape)
                self.landmark_history.append(landmarks)

                # Keep only last 5 seconds of data
                max_frames = self.fps * 5
                if len(self.landmark_history) > max_frames:
                    self.landmark_history = self.landmark_history[-max_frames:]

                # Update traditional metrics if we have enough data
                if len(self.landmark_history) > self.fps:  # At least 1 second of data
                    self._update_metrics()

                # Draw pose skeleton
                frame = self.pose_detector.draw_pose(frame, keypoints)

                # Add frame info
                valid_count = sum(1 for kp in keypoints if kp[2] > 0)
                cv2.putText(frame, f"OpenPose: {valid_count}/25", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                # Increment failed counter
                self.failed_leg_detection_count += 1
                
                # Add warning message
                cv2.putText(frame, f"Insufficient leg detection ({self.failed_leg_detection_count}/{self.max_failed_detections})", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                
                # Check if we should stop analysis
                if self.failed_leg_detection_count >= self.max_failed_detections:
                    # Return error signal
                    return frame, "LEG_DETECTION_FAILED"
        else:
            # Increment failed counter for no pose detected
            self.failed_leg_detection_count += 1
            
            # Add "no pose detected" message
            cv2.putText(frame, f"No pose detected ({self.failed_leg_detection_count}/{self.max_failed_detections})", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Check if we should stop analysis
            if self.failed_leg_detection_count >= self.max_failed_detections:
                return frame, "LEG_DETECTION_FAILED"

        self.frame_count += 1
        return frame, self.metrics
    
    def _analyze_professional_metrics(self, keypoints):
        """Analyze professional gait metrics from current frame"""
        # 1. Joint Angles Analysis
        angles = self.analyze_joint_angles(keypoints)
        if angles:
            self.metrics.knee_angle_left = angles.get('knee_left', 0.0)
            self.metrics.knee_angle_right = angles.get('knee_right', 0.0)
            self.metrics.hip_angle_left = angles.get('hip_left', 0.0)
            self.metrics.hip_angle_right = angles.get('hip_right', 0.0)
            self.metrics.ankle_angle_left = angles.get('ankle_left', 0.0)
            self.metrics.ankle_angle_right = angles.get('ankle_right', 0.0)
        
        # 2. Distance Analysis
        distances = self.analyze_distances(keypoints)
        if distances:
            self.metrics.step_width_current = distances.get('step_width', 0.0)
            self.metrics.foot_clearance_left = distances.get('foot_clearance_left', 0.0)
            self.metrics.foot_clearance_right = distances.get('foot_clearance_right', 0.0)
        
        # 3. Timing Analysis (simplified for now)
        # This would require more sophisticated temporal analysis
        self.metrics.stance_phase_percent = 60.0  # Example value
        self.metrics.swing_phase_percent = 40.0   # Example value
        
        # Store angles for bilateral comparison (only if valid)
        if angles:
            knee_left = angles.get('knee_left', 0.0)
            knee_right = angles.get('knee_right', 0.0)
            hip_left = angles.get('hip_left', 0.0)
            hip_right = angles.get('hip_right', 0.0)
            ankle_left = angles.get('ankle_left', 0.0)
            ankle_right = angles.get('ankle_right', 0.0)
            
            if knee_left > 0:
                self.left_joint_angles["knee"].append(knee_left)
            if knee_right > 0:
                self.right_joint_angles["knee"].append(knee_right)
            if hip_left > 0:
                self.left_joint_angles["hip"].append(hip_left)
            if hip_right > 0:
                self.right_joint_angles["hip"].append(hip_right)
            if ankle_left > 0:
                self.left_joint_angles["ankle"].append(ankle_left)
            if ankle_right > 0:
                self.right_joint_angles["ankle"].append(ankle_right)
        
        # 4. Symmetry Analysis
        if self.metrics.knee_angle_left > 0 and self.metrics.knee_angle_right > 0:
            angle_diff = abs(self.metrics.knee_angle_left - self.metrics.knee_angle_right)
            self.metrics.symmetry_index = (angle_diff / max(self.metrics.knee_angle_left, self.metrics.knee_angle_right)) * 100
    
    def _extract_landmarks_from_keypoints(self, keypoints, frame_shape):
        """Extract key landmarks from OpenPose keypoints"""
        landmarks = {}
        h, w = frame_shape[:2]
        
        for name, idx in self.key_landmarks.items():
            if idx < len(keypoints):
                x, y, confidence = keypoints[idx]
                if confidence > 0:  # Valid keypoint
                    landmarks[name] = {
                        'x': x,
                        'y': y,
                        'confidence': confidence
                    }
        
        return landmarks
    
    def _update_metrics(self):
        """Update gait metrics based on landmark history"""
        if len(self.landmark_history) < 2:
            return
            
        # Calculate step detection based on ankle movement
        self._detect_steps()
        self._calculate_cadence()
        self._estimate_stride_length()
        self._calculate_step_timing()
        self._calculate_foot_angles()
        self._estimate_walking_speed()
    
    def _detect_steps(self):
        """Detect steps based on ankle vertical movement"""
        left_ankle_y = [frame.get('left_ankle', {}).get('y', 0) for frame in self.landmark_history]
        right_ankle_y = [frame.get('right_ankle', {}).get('y', 0) for frame in self.landmark_history]
        
        # Find peaks (local maxima) in ankle height - indicates step
        left_peaks, _ = find_peaks(left_ankle_y, height=np.mean(left_ankle_y), distance=self.fps//2)
        right_peaks, _ = find_peaks(right_ankle_y, height=np.mean(right_ankle_y), distance=self.fps//2)
        
        self.metrics.step_count = len(left_peaks) + len(right_peaks)
    
    def _calculate_cadence(self):
        """Calculate cadence (steps per minute)"""
        if len(self.landmark_history) > 0:
            time_window = len(self.landmark_history) / self.fps  # seconds
            if time_window > 0:
                self.metrics.cadence = (self.metrics.step_count / time_window) * 60
    
    def _estimate_stride_length(self):
        """Estimate stride length based on hip movement"""
        if len(self.landmark_history) < 2:
            return
            
        # Calculate horizontal displacement of hips
        left_hip_x = [frame.get('left_hip', {}).get('x', 0) for frame in self.landmark_history]
        right_hip_x = [frame.get('right_hip', {}).get('x', 0) for frame in self.landmark_history]
        
        if left_hip_x and right_hip_x:
            left_displacement = max(left_hip_x) - min(left_hip_x)
            right_displacement = max(right_hip_x) - min(right_hip_x)
            self.metrics.stride_length = (left_displacement + right_displacement) / 2
    
    def _calculate_step_timing(self):
        """Calculate step timing parameters"""
        if self.metrics.cadence > 0:
            self.metrics.step_time = 60 / self.metrics.cadence  # seconds per step
    
    def _calculate_foot_angles(self):
        """Calculate foot angles relative to ground"""
        if len(self.landmark_history) > 0:
            latest_frame = self.landmark_history[-1]
            
            # Left foot angle (using heel and toe)
            if 'left_heel' in latest_frame and 'left_toe' in latest_frame:
                heel = latest_frame['left_heel']
                toe = latest_frame['left_toe']
                self.metrics.foot_angle_left = np.degrees(
                    np.arctan2(toe['y'] - heel['y'], toe['x'] - heel['x'])
                )
            
            # Right foot angle (using heel and toe)
            if 'right_heel' in latest_frame and 'right_toe' in latest_frame:
                heel = latest_frame['right_heel']
                toe = latest_frame['right_toe']
                self.metrics.foot_angle_right = np.degrees(
                    np.arctan2(toe['y'] - heel['y'], toe['x'] - heel['x'])
                )
    
    def _estimate_walking_speed(self):
        """Estimate walking speed"""
        if self.metrics.cadence > 0 and self.metrics.stride_length > 0:
            # Very rough estimation - would need calibration in real application
            steps_per_second = self.metrics.cadence / 60
            self.metrics.walking_speed = steps_per_second * self.metrics.stride_length
    
    def get_analysis_summary(self) -> Dict:
        """Get professional gait analysis summary"""
        return {
            # Joint Angles
            "knee_angle_left": self.metrics.knee_angle_left,
            "knee_angle_right": self.metrics.knee_angle_right,
            "hip_angle_left": self.metrics.hip_angle_left,
            "hip_angle_right": self.metrics.hip_angle_right,
            "ankle_angle_left": self.metrics.ankle_angle_left,
            "ankle_angle_right": self.metrics.ankle_angle_right,
            
            # Position & Distance
            "foot_clearance_left": self.metrics.foot_clearance_left,
            "foot_clearance_right": self.metrics.foot_clearance_right,
            "step_width": self.metrics.step_width_current,
            "step_length": self.metrics.stride_length,
            
            # Timing
            "stance_phase": self.metrics.stance_phase_percent,
            "swing_phase": self.metrics.swing_phase_percent,
            
            # Symmetry
            "symmetry_index": self.metrics.symmetry_index,
            
            # Traditional metrics for compatibility
            "cadence": self.metrics.cadence,
            "walking_speed": self.metrics.walking_speed,
        }
    
    def reset_analysis(self):
        """Reset analysis data"""
        self.landmark_history = []
        self.metrics = GaitMetrics()
        self.frame_count = 0
        
        # Reset tracking for bilateral comparison
        self.left_joint_angles = {"knee": [], "hip": [], "ankle": []}
        self.right_joint_angles = {"knee": [], "hip": [], "ankle": []}
        
        # Reset leg detection counter
        self.failed_leg_detection_count = 0
    
    def get_bilateral_comparison(self):
        """Get bilateral comparison results after analysis session"""
        import numpy as np
        
        results = {}
        
        # Calculate average angles for each joint on each side
        for joint in ["knee", "hip", "ankle"]:
            left_angles = self.left_joint_angles[joint]
            right_angles = self.right_joint_angles[joint]
            
            if left_angles and right_angles:
                left_avg = np.mean(left_angles)
                right_avg = np.mean(right_angles)
                difference = abs(left_avg - right_avg)
                
                results[f"{joint}_angle_left_avg"] = left_avg
                results[f"{joint}_angle_right_avg"] = right_avg
                results[f"{joint}_angle_difference"] = difference
                
                # Calculate asymmetry percentage
                max_angle = max(left_avg, right_avg)
                if max_angle > 0:
                    asymmetry_percent = (difference / max_angle) * 100
                    results[f"{joint}_asymmetry_percent"] = asymmetry_percent
            else:
                results[f"{joint}_angle_left_avg"] = 0.0
                results[f"{joint}_angle_right_avg"] = 0.0
                results[f"{joint}_angle_difference"] = 0.0
                results[f"{joint}_asymmetry_percent"] = 0.0
        
        # Overall bilateral symmetry score
        total_asymmetry = (
            results.get("knee_asymmetry_percent", 0) +
            results.get("hip_asymmetry_percent", 0) +
            results.get("ankle_asymmetry_percent", 0)
        ) / 3
        
        results["overall_asymmetry_percent"] = total_asymmetry
        
        return results
    
    def export_session_data(self, session_name="gait_session", patient_info=None):
        """Export all session data to text file"""
        import json
        from datetime import datetime
        import os
        
        # Create results directory if not exists
        results_dir = "results"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{results_dir}/{session_name}_{timestamp}.txt"
        
        # Prepare data for export
        export_data = {
            "patient_info": patient_info or {
                "name": "Unknown",
                "age": 0,
                "gender": "Unknown",
                "notes": ""
            },
            "session_info": {
                "timestamp": datetime.now().isoformat(),
                "total_frames": self.frame_count,
                "session_duration_seconds": 60,  # 1 minute
                "fps": self.fps
            },
            "raw_joint_angles": {
                "left_knee": self.left_joint_angles["knee"],
                "right_knee": self.right_joint_angles["knee"],
                "left_hip": self.left_joint_angles["hip"],
                "right_hip": self.right_joint_angles["hip"],
                "left_ankle": self.left_joint_angles["ankle"],
                "right_ankle": self.right_joint_angles["ankle"]
            },
            "final_metrics": self.get_analysis_summary(),
            "bilateral_comparison": self.get_bilateral_comparison()
        }
        
        # Write to file
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("=" * 60 + "\n")
                f.write("GAITSENSE AI - SESSION DATA EXPORT\n")
                f.write("=" * 60 + "\n\n")
                
                # Patient information
                patient = export_data['patient_info']
                f.write("👤 THÔNG TIN BỆNH NHÂN:\n")
                f.write(f"Họ và tên: {patient['name']}\n")
                f.write(f"Tuổi: {patient['age']} tuổi\n")
                f.write(f"Giới tính: {patient['gender']}\n")
                if patient['notes']:
                    f.write(f"Ghi chú: {patient['notes']}\n")
                f.write("\n")
                
                # Session info
                f.write("📊 SESSION INFORMATION:\n")
                f.write(f"Timestamp: {export_data['session_info']['timestamp']}\n")
                f.write(f"Total Frames Analyzed: {export_data['session_info']['total_frames']}\n")
                f.write(f"Session Duration: {export_data['session_info']['session_duration_seconds']} seconds\n")
                f.write(f"Analysis FPS: {export_data['session_info']['fps']}\n\n")
                
                # Raw data statistics
                f.write("📈 RAW DATA STATISTICS:\n")
                for joint, angles in export_data['raw_joint_angles'].items():
                    if angles:
                        import numpy as np
                        f.write(f"{joint.replace('_', ' ').title()}:\n")
                        f.write(f"  - Sample Count: {len(angles)}\n")
                        f.write(f"  - Average: {np.mean(angles):.2f}°\n")
                        f.write(f"  - Min: {np.min(angles):.2f}°\n")
                        f.write(f"  - Max: {np.max(angles):.2f}°\n")
                        f.write(f"  - Std Dev: {np.std(angles):.2f}°\n\n")
                
                # Final metrics
                f.write("🎯 FINAL ANALYSIS METRICS:\n")
                metrics = export_data['final_metrics']
                for key, value in metrics.items():
                    f.write(f"{key.replace('_', ' ').title()}: {value}\n")
                f.write("\n")
                
                # Bilateral comparison
                f.write("⚖️ BILATERAL COMPARISON:\n")
                comparison = export_data['bilateral_comparison']
                for key, value in comparison.items():
                    if isinstance(value, float):
                        f.write(f"{key.replace('_', ' ').title()}: {value:.2f}\n")
                    else:
                        f.write(f"{key.replace('_', ' ').title()}: {value}\n")
                f.write("\n")
                
                # Raw angle data (for detailed analysis)
                f.write("📊 DETAILED RAW ANGLE DATA:\n")
                f.write("(For advanced analysis and research)\n\n")
                
                for joint, angles in export_data['raw_joint_angles'].items():
                    if angles:
                        f.write(f"{joint.upper()}_ANGLES = {angles}\n\n")
                
                # JSON data for machine reading
                f.write("\n" + "=" * 60 + "\n")
                f.write("JSON DATA (Machine Readable):\n")
                f.write("=" * 60 + "\n")
                f.write(json.dumps(export_data, indent=2, ensure_ascii=False))
            
            print(f"📄 Session data exported to: {filename}")
            return filename
            
        except Exception as e:
            print(f"❌ Error exporting data: {e}")
            return None
    
    @staticmethod
    def load_and_diagnose(filename):
        """Load session data from file and provide detailed diagnosis"""
        import json
        import numpy as np
        
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Extract JSON data from file
            # Look for the JSON block starting with patient_info
            json_start = content.find('{\n  "patient_info"')
            if json_start == -1:
                # Try alternative format
                json_start = content.find('{"patient_info"')
                if json_start == -1:
                    raise ValueError("No valid JSON data found in file")
                
            json_data = content[json_start:]
            
            # Clean up any trailing content after the JSON
            try:
                data = json.loads(json_data)
            except json.JSONDecodeError:
                # Try to find the end of the JSON block
                brace_count = 0
                json_end = 0
                for i, char in enumerate(json_data):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            json_end = i + 1
                            break
                
                if json_end > 0:
                    json_data = json_data[:json_end]
                    data = json.loads(json_data)
                else:
                    raise ValueError("Could not parse JSON data")
            
            # Generate comprehensive diagnosis
            diagnosis = GaitAnalyzer._generate_comprehensive_diagnosis(data)
            return diagnosis
            
        except Exception as e:
            print(f"❌ Error loading file: {e}")
            return None
    
    @staticmethod
    def _generate_comprehensive_diagnosis(data):
        """Generate comprehensive medical-style diagnosis from data"""
        import numpy as np
        import sys
        import os
        
        # Add src/data to path for importing normative data
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(os.path.dirname(current_dir), 'data')
        if data_dir not in sys.path:
            sys.path.append(data_dir)
        
        try:
            from normative_gait_data import NormativeGaitData
            normative_db = NormativeGaitData()
        except ImportError:
            print("⚠️ Không thể tải dữ liệu chuẩn, sử dụng phương pháp cũ")
            normative_db = None
        
        # Extract patient info
        patient_info = data.get('patient_info', {})
        
        diagnosis = {
            "patient_name": patient_info.get('name', 'Không rõ'),
            "patient_age": patient_info.get('age', 0),
            "patient_gender": patient_info.get('gender', 'Không rõ'),
            "patient_notes": patient_info.get('notes', ''),
            "session_date": data.get('session_info', {}).get('timestamp', 'Không rõ'),
            "assessment_summary": "",
            "detailed_findings": {},
            "recommendations": [],
            "severity_score": 0,
            "follow_up_needed": False
        }
        
        # Extract data
        bilateral = data.get('bilateral_comparison', {})
        raw_angles = data.get('raw_joint_angles', {})
        final_metrics = data.get('final_metrics', {})
        
        # If normative data is available, use evidence-based assessment
        if normative_db:
            # Prepare patient data for assessment
            patient_data = {
                'age': patient_info.get('age', 30),
                'gender': patient_info.get('gender', 'Nam')
            }
            
            # Prepare measured data (convert to expected format)
            measured_data = {
                'knee_flexion': bilateral.get('knee_angle_left_avg', 0),
                'hip_flexion': bilateral.get('hip_angle_left_avg', 0), 
                'ankle_dorsiflexion': bilateral.get('ankle_angle_left_avg', 0),
                'cadence': final_metrics.get('cadence', 0),
                'stride_length': final_metrics.get('step_length', 0),
                'walking_speed': final_metrics.get('walking_speed', 0) / 100 if final_metrics.get('walking_speed', 0) > 10 else final_metrics.get('walking_speed', 0),  # Convert if needed
                'stance_phase_percentage': final_metrics.get('stance_phase', 0),
                'left_knee': bilateral.get('knee_angle_left_avg', 0),
                'right_knee': bilateral.get('knee_angle_right_avg', 0),
                'left_hip': bilateral.get('hip_angle_left_avg', 0),
                'right_hip': bilateral.get('hip_angle_right_avg', 0),
                'left_ankle': bilateral.get('ankle_angle_left_avg', 0),
                'right_ankle': bilateral.get('ankle_angle_right_avg', 0)
            }
            
            # Get comprehensive assessment
            normative_assessment = normative_db.get_comprehensive_assessment(patient_data, measured_data)
            
            # Use normative-based assessment
            diagnosis.update({
                "assessment_summary": f"Đánh giá dựa trên dữ liệu chuẩn quốc tế (Điểm tổng: {normative_assessment['overall_score']}/3.0)",
                "detailed_findings": {},
                "recommendations": normative_assessment['recommendations'],
                "severity_score": int(normative_assessment['overall_score']),
                "follow_up_needed": normative_assessment['overall_score'] >= 1.5,
                "normative_data_used": True
            })
            
            # Convert individual assessments to detailed findings
            for param, assessment in normative_assessment['individual_assessments'].items():
                param_names = {
                    'knee_flexion': 'Khớp Gối',
                    'hip_flexion': 'Khớp Hông',
                    'ankle_dorsiflexion': 'Khớp Cổ Chân',
                    'cadence': 'Nhịp Độ Bước',
                    'stride_length': 'Chiều Dài Bước',
                    'walking_speed': 'Tốc Độ Đi',
                    'stance_phase_percentage': 'Pha Chống Đỡ'
                }
                
                param_vietnamese = param_names.get(param, param)
                status_map = {
                    'normal': 'BÌNH THƯỜNG',
                    'mild': 'NHẸ', 
                    'moderate': 'TRUNG BÌNH',
                    'severe': 'NẶNG'
                }
                
                diagnosis['detailed_findings'][param_vietnamese] = {
                    'status': status_map.get(assessment['status'], 'KHÔNG RÕ'),
                    'deviation_index': assessment['deviation_index'],
                    'position_percentage': assessment['position_percentage'],
                    'recommendation': assessment['interpretation'],
                    'normative_mean': assessment.get('normative_mean', 0),
                    'normative_std': assessment.get('normative_std', 0)
                }
            
            # Add asymmetry assessments
            for joint, assessment in normative_assessment['asymmetry_assessments'].items():
                joint_names = {'knee': 'Bất Cân Xứng Gối', 'hip': 'Bất Cân Xứng Hông', 'ankle': 'Bất Cân Xứng Cổ Chân'}
                joint_vietnamese = joint_names.get(joint, f'Bất Cân Xứng {joint}')
                
                status_map = {
                    'normal': 'BÌNH THƯỜNG',
                    'mild': 'NHẸ',
                    'moderate': 'TRUNG BÌNH', 
                    'severe': 'NẶNG'
                }
                
                diagnosis['detailed_findings'][joint_vietnamese] = {
                    'status': status_map.get(assessment['status'], 'KHÔNG RÕ'),
                    'asymmetry_percent': assessment['asymmetry_percent'],
                    'recommendation': assessment['interpretation']
                }
            
            return diagnosis
        
        # Fallback to original method if normative data not available
        print("📊 Sử dụng phương pháp đánh giá truyền thống")
        
        # Calculate severity scores
        severity_issues = []
        detailed_findings = {}
        
        # 1. Knee Analysis
        knee_diff = bilateral.get('knee_angle_difference', 0)
        knee_asymmetry = bilateral.get('knee_asymmetry_percent', 0)
        
        if knee_diff > 15:
            severity_issues.append("severe_knee_asymmetry")
            detailed_findings['Khớp Gối'] = {
                'status': 'NẶNG',
                'difference': knee_diff,
                'asymmetry': knee_asymmetry,
                'recommendation': 'Khuyến nghị tham khảo ngay chuyên khoa chỉnh hình'
            }
        elif knee_diff > 8:
            severity_issues.append("moderate_knee_asymmetry")
            detailed_findings['Khớp Gối'] = {
                'status': 'TRUNG BÌNH',
                'difference': knee_diff,
                'asymmetry': knee_asymmetry,
                'recommendation': 'Đề xuất đánh giá vật lý trị liệu'
            }
        elif knee_diff > 4:
            detailed_findings['Khớp Gối'] = {
                'status': 'NHẸ',
                'difference': knee_diff,
                'asymmetry': knee_asymmetry,
                'recommendation': 'Theo dõi và phân tích định kỳ'
            }
        else:
            detailed_findings['Khớp Gối'] = {
                'status': 'BÌNH THƯỜNG',
                'difference': knee_diff,
                'asymmetry': knee_asymmetry,
                'recommendation': 'Không cần can thiệp'
            }
        
        # 2. Hip Analysis
        hip_diff = bilateral.get('hip_angle_difference', 0)
        hip_asymmetry = bilateral.get('hip_asymmetry_percent', 0)
        
        if hip_diff > 12:
            severity_issues.append("severe_hip_asymmetry")
            detailed_findings['Khớp Hông'] = {
                'status': 'NẶNG',
                'difference': hip_diff,
                'asymmetry': hip_asymmetry,
                'recommendation': 'Cần đánh giá khớp hông'
            }
        elif hip_diff > 6:
            severity_issues.append("moderate_hip_asymmetry")
            detailed_findings['Khớp Hông'] = {
                'status': 'TRUNG BÌNH',
                'difference': hip_diff,
                'asymmetry': hip_asymmetry,
                'recommendation': 'Khuyến nghị luyện tập dáng đi'
            }
        elif hip_diff > 3:
            detailed_findings['Khớp Hông'] = {
                'status': 'NHẸ',
                'difference': hip_diff,
                'asymmetry': hip_asymmetry,
                'recommendation': 'Bài tập tăng cường cơ có thể hữu ích'
            }
        else:
            detailed_findings['Khớp Hông'] = {
                'status': 'BÌNH THƯỜNG',
                'difference': hip_diff,
                'asymmetry': hip_asymmetry,
                'recommendation': 'Không cần can thiệp'
            }
        
        # 3. Ankle Analysis
        ankle_diff = bilateral.get('ankle_angle_difference', 0)
        ankle_asymmetry = bilateral.get('ankle_asymmetry_percent', 0)
        
        if ankle_diff > 10:
            severity_issues.append("severe_ankle_asymmetry")
            detailed_findings['Khớp Cổ Chân'] = {
                'status': 'NẶNG',
                'difference': ankle_diff,
                'asymmetry': ankle_asymmetry,
                'recommendation': 'Tham khảo chuyên khoa cổ chân/bàn chân'
            }
        elif ankle_diff > 5:
            severity_issues.append("moderate_ankle_asymmetry")
            detailed_findings['Khớp Cổ Chân'] = {
                'status': 'TRUNG BÌNH',
                'difference': ankle_diff,
                'asymmetry': ankle_asymmetry,
                'recommendation': 'Khuyến nghị luyện tập thăng bằng'
            }
        elif ankle_diff > 2:
            detailed_findings['Khớp Cổ Chân'] = {
                'status': 'NHẸ',
                'difference': ankle_diff,
                'asymmetry': ankle_asymmetry,
                'recommendation': 'Bài tập vận động khớp cổ chân'
            }
        else:
            detailed_findings['Khớp Cổ Chân'] = {
                'status': 'BÌNH THƯỜNG',
                'difference': ankle_diff,
                'asymmetry': ankle_asymmetry,
                'recommendation': 'Không cần can thiệp'
            }
        
        # 4. Overall Assessment
        total_severity = len([x for x in severity_issues if 'severe' in x])
        moderate_issues = len([x for x in severity_issues if 'moderate' in x])
        
        if total_severity >= 2:
            diagnosis['severity_score'] = 3  # High
            diagnosis['assessment_summary'] = "Phát hiện nhiều bất cân xứng nghiêm trọng giữa hai bên. Khuyến nghị mạnh đánh giá y tế toàn diện."
            diagnosis['follow_up_needed'] = True
        elif total_severity >= 1 or moderate_issues >= 2:
            diagnosis['severity_score'] = 2  # Medium
            diagnosis['assessment_summary'] = "Phát hiện bất cân xứng dáng đi đáng kể. Khuyến nghị tham khảo ý kiến bác sĩ."
            diagnosis['follow_up_needed'] = True
        elif moderate_issues >= 1:
            diagnosis['severity_score'] = 1  # Low
            diagnosis['assessment_summary'] = "Phát hiện bất cân xứng nhẹ đến trung bình. Theo dõi và can thiệp có thể có lợi."
            diagnosis['follow_up_needed'] = False
        else:
            diagnosis['severity_score'] = 0  # Normal
            diagnosis['assessment_summary'] = "Dáng đi trong phạm vi cân xứng bình thường giữa hai bên."
            diagnosis['follow_up_needed'] = False
        
        # 5. Generate Recommendations
        recommendations = []
        if total_severity > 0:
            recommendations.append("Lên lịch đánh giá chỉnh hình toàn diện trong vòng 2 tuần")
            recommendations.append("Ghi nhận bất kỳ đau đớn, cứng khớp hoặc chấn thương trước đây")
            recommendations.append("Cân nhắc các nghiên cứu hình ảnh (X-quang, MRI) nếu có chỉ định lâm sàng")
        
        if moderate_issues > 0:
            recommendations.append("Bắt đầu đánh giá vật lý trị liệu")
            recommendations.append("Đánh giá mất cân bằng về sức mạnh và tính linh hoạt của cơ")
            recommendations.append("Cân nhắc chương trình luyện tập dáng đi")
        
        # General recommendations
        if any([total_severity, moderate_issues]):
            recommendations.append("Lặp lại phân tích dáng đi sau 4-6 tuần để theo dõi tiến triển")
            recommendations.append("Duy trì tập luyện thường xuyên trong giới hạn thoải mái")
            recommendations.append("Sử dụng giày hỗ trợ")
        
        diagnosis['detailed_findings'] = detailed_findings
        diagnosis['recommendations'] = recommendations
        
        return diagnosis
    
    def check_leg_detection(self, keypoints):
        """Check if essential leg keypoints are detected"""
        if keypoints is None:
            return False
            
        # Essential leg keypoints: at least 4 out of 7 lower body keypoints
        essential_points = [8, 9, 10, 11, 12, 13, 14]  # MidHip, RHip, RKnee, RAnkle, LHip, LKnee, LAnkle
        
        detected_count = 0
        for i in essential_points:
            if i < len(keypoints) and keypoints[i][2] > 0.1:  # Valid confidence
                detected_count += 1
        
        # Need at least 4 out of 7 key leg points
        return detected_count >= 4

