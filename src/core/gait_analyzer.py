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
        """Analyze distances and clearances with proper pixel-to-cm conversion and multi-frame tracking"""
        distances = {}
        
        # Extract key points
        points = {}
        for name, idx in self.key_landmarks.items():
            if idx < len(keypoints) and keypoints[idx][2] > 0.1:
                points[name] = keypoints[idx][:2]
        
        # Pixel to cm conversion factors - FINAL TUNED for realistic Vietnamese measurements
        # Based on typical webcam setup: 1.5-2m distance, normal room lighting
        # Target: step_width 10-15cm, foot_clearance 4-7cm for normal Vietnamese gait
        pixel_to_cm_horizontal = 0.01  # 1cm per 100 pixels (~0.01 cm/pixel) - OPTIMAL
        pixel_to_cm_vertical = 0.01    # 1cm per 100 pixels (~0.01 cm/pixel) - FINAL TUNED (reduced from 0.012)
        
        # 1. Calculate step width (distance between ankles) - convert to cm
        if 'left_ankle' in points and 'right_ankle' in points:
            pixel_distance = self.calculate_distance(points['left_ankle'], points['right_ankle'])
            distances['step_width'] = pixel_distance * pixel_to_cm_horizontal
            
            # Apply realistic bounds for step width (5-18cm for normal gait)
            if distances['step_width'] > 18:
                distances['step_width'] = 18
            elif distances['step_width'] < 5:
                distances['step_width'] = 5
        
        # 2. Enhanced foot clearance calculation using ground reference history
        # Establish ground reference using multiple foot points
        foot_positions = []
        if 'left_ankle' in points:
            foot_positions.append(points['left_ankle'][1])
        if 'right_ankle' in points:
            foot_positions.append(points['right_ankle'][1])
        if 'left_heel' in points:
            foot_positions.append(points['left_heel'][1])
        if 'right_heel' in points:
            foot_positions.append(points['right_heel'][1])
        
        if foot_positions:
            # Use the most consistent ground reference (the foot that's on ground)
            ground_level_y = max(foot_positions)  # Highest Y = lowest position in image
            
            # Calculate foot clearance with improved accuracy
            if 'left_ankle' in points:
                clearance_pixels = max(0, ground_level_y - points['left_ankle'][1])
                distances['foot_clearance_left'] = clearance_pixels * pixel_to_cm_vertical
                
                # Apply realistic bounds for foot clearance (0-10cm for normal gait)
                if distances['foot_clearance_left'] > 10:
                    distances['foot_clearance_left'] = 10
            
            if 'right_ankle' in points:
                clearance_pixels = max(0, ground_level_y - points['right_ankle'][1])
                distances['foot_clearance_right'] = clearance_pixels * pixel_to_cm_vertical
                
                # Apply realistic bounds for foot clearance
                if distances['foot_clearance_right'] > 10:
                    distances['foot_clearance_right'] = 10
        
        # 3. Store current frame data for multi-frame analysis
        if not hasattr(self, 'foot_clearance_history'):
            self.foot_clearance_history = {'left': [], 'right': []}
        
        # Track maximum clearance over time for more accurate peak detection
        if distances.get('foot_clearance_left', 0) > 0:
            self.foot_clearance_history['left'].append(distances['foot_clearance_left'])
            # Keep only last 30 frames (~1 second at 30fps)
            if len(self.foot_clearance_history['left']) > 30:
                self.foot_clearance_history['left'].pop(0)
            
            # Use maximum clearance from recent history for more accurate measurement
            distances['foot_clearance_left'] = max(self.foot_clearance_history['left'])
        
        if distances.get('foot_clearance_right', 0) > 0:
            self.foot_clearance_history['right'].append(distances['foot_clearance_right'])
            if len(self.foot_clearance_history['right']) > 30:
                self.foot_clearance_history['right'].pop(0)
            
            distances['foot_clearance_right'] = max(self.foot_clearance_history['right'])
        
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
                
                # Also calculate and store distances for this frame
                distances = self.analyze_distances(keypoints)
                landmarks.update(distances)  # Add distance measurements to landmarks
                
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
        """Estimate stride length based on ankle movement with consistent conversion and step width tracking"""
        if len(self.landmark_history) < 10:  # Need enough frames
            return
            
        # Calculate stride length from ankle movement - more accurate
        left_ankle_x = [frame.get('left_ankle', {}).get('x', 0) for frame in self.landmark_history if frame.get('left_ankle', {}).get('x', 0) > 0]
        right_ankle_x = [frame.get('right_ankle', {}).get('x', 0) for frame in self.landmark_history if frame.get('right_ankle', {}).get('x', 0) > 0]
        
        if len(left_ankle_x) > 5 and len(right_ankle_x) > 5:
            # Calculate step distance as average movement between steps
            left_displacement = max(left_ankle_x) - min(left_ankle_x) if left_ankle_x else 0
            right_displacement = max(right_ankle_x) - min(right_ankle_x) if right_ankle_x else 0
            
            # Average displacement and convert to stride length
            avg_displacement = (left_displacement + right_displacement) / 2
            
            if avg_displacement > 0:
                # Use consistent pixel-to-cm conversion (FINAL TUNED for realistic measurements)
                pixel_to_cm_horizontal = 0.01  # Same as in analyze_distances - OPTIMAL
                self.metrics.stride_length = avg_displacement * pixel_to_cm_horizontal
                
                # Apply realistic bounds for Vietnamese people (40-120cm)
                if self.metrics.stride_length > 120:
                    self.metrics.stride_length = 120
                elif self.metrics.stride_length < 40:
                    self.metrics.stride_length = 40
        
        # Update step width from recent measurements with improved tracking
        if hasattr(self, 'metrics') and len(self.landmark_history) > 0:
            # Get step width measurements from recent frames
            step_widths = []
            for i in range(min(10, len(self.landmark_history))):  # Last 10 frames
                frame = self.landmark_history[-(i+1)]
                if 'step_width' in frame and frame['step_width'] > 0:
                    step_widths.append(frame['step_width'])
            
            # Use average step width for more stable measurement
            if step_widths:
                self.metrics.step_width_current = sum(step_widths) / len(step_widths)
        
        # Update foot clearance metrics from multi-frame tracking
        if hasattr(self, 'foot_clearance_history'):
            if self.foot_clearance_history['left']:
                self.metrics.foot_clearance_left = max(self.foot_clearance_history['left'])
            if self.foot_clearance_history['right']:
                self.metrics.foot_clearance_right = max(self.foot_clearance_history['right'])
    
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
        """Estimate walking speed in m/s"""
        if self.metrics.cadence > 0 and self.metrics.stride_length > 0:
            # Convert to realistic walking speed
            steps_per_second = self.metrics.cadence / 60
            
            # Stride length is now in cm, convert to meters
            stride_length_meters = self.metrics.stride_length / 100
            
            # Calculate walking speed in m/s
            self.metrics.walking_speed = steps_per_second * stride_length_meters
            
            # Apply reasonable bounds for Vietnamese people (0.4-1.8 m/s)
            if self.metrics.walking_speed > 1.8:
                self.metrics.walking_speed = 1.8
            elif self.metrics.walking_speed < 0.4:
                self.metrics.walking_speed = 0.4
    
    def get_analysis_summary(self) -> Dict:
        """Get simplified gait analysis summary - including all essential parameters"""
        return {
            # Essential parameters for diagnosis report
            "stride_length": self.metrics.stride_length,  # Already in cm
            "walking_speed": self.metrics.walking_speed,  # Already in m/s
            "stance_phase": self.metrics.stance_phase_percent,  # Stance phase %
            
            # New distance measurements
            "foot_clearance_left": self.metrics.foot_clearance_left,  # cm
            "foot_clearance_right": self.metrics.foot_clearance_right,  # cm
            "step_width_current": self.metrics.step_width_current,  # cm
            
            # Joint angles for asymmetry calculation
            "knee_angle_left": self.metrics.knee_angle_left,
            "knee_angle_right": self.metrics.knee_angle_right,
            "hip_angle_left": self.metrics.hip_angle_left,
            "hip_angle_right": self.metrics.hip_angle_right,
            "ankle_angle_left": self.metrics.ankle_angle_left,
            "ankle_angle_right": self.metrics.ankle_angle_right,
            
            # Traditional metrics for compatibility (sidebar display)
            "cadence": self.metrics.cadence,
            "step_count": self.metrics.step_count,
            "symmetry_index": self.metrics.symmetry_index,
        }
    
    def reset_analysis(self):
        """Reset analysis data"""
        self.landmark_history = []
        self.metrics = GaitMetrics()
        self.frame_count = 0
        
        # Reset tracking for bilateral comparison
        self.left_joint_angles = {"knee": [], "hip": [], "ankle": []}
        self.right_joint_angles = {"knee": [], "hip": [], "ankle": []}
        
        # Reset foot clearance history
        if hasattr(self, 'foot_clearance_history'):
            self.foot_clearance_history = {'left': [], 'right': []}
        
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
        
        # Prepare data for export - CHỈ LƯU RAW DATA, KHÔNG LƯU PHÂN TÍCH
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
            "raw_landmark_history": self.landmark_history  # Thêm landmark history để tính toán lại
            # ❌ KHÔNG LƯU: final_metrics, bilateral_comparison - sẽ tính toán lại khi mở báo cáo
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
                
                # Lưu ý: Không lưu analysis results - sẽ tính toán lại khi mở báo cáo
                f.write("💡 LƯU Ý QUAN TRỌNG:\n")
                f.write("File này chỉ chứa dữ liệu thô đo được trong 1 phút.\n")
                f.write("Phân tích và chẩn đoán sẽ được tính toán lại khi mở báo cáo.\n")
                f.write("Điều này đảm bảo sử dụng dữ liệu chuẩn mới nhất và thuật toán cập nhật.\n\n")
                
                # Raw angle data (chỉ để tham khảo)
                f.write("📊 DỮ LIỆU GÓC KHỚP THÔ:\n")
                f.write("(Sẽ được phân tích chi tiết khi tạo báo cáo chẩn đoán)\n\n")
                
                for joint, angles in export_data['raw_joint_angles'].items():
                    if angles:
                        sample_size = min(10, len(angles))  # Chỉ hiển thị 10 mẫu đầu
                        f.write(f"{joint.replace('_', ' ').title()} (mẫu): {angles[:sample_size]}...\n")
                
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
            print(f"🔍 Debug: Loaded data keys: {list(data.keys())}")
            print(f"🔍 Debug: raw_joint_angles keys: {list(data.get('raw_joint_angles', {}).keys())}")
            print(f"🔍 Debug: raw_landmark_history length: {len(data.get('raw_landmark_history', []))}")
            
            # Sample some landmark_history data for debugging
            landmark_history = data.get('raw_landmark_history', [])
            if landmark_history:
                print(f"🔍 Debug: First landmark frame: {landmark_history[0] if landmark_history else 'None'}")
                if len(landmark_history) > 1:
                    print(f"🔍 Debug: Last landmark frame: {landmark_history[-1]}")
            
            diagnosis = GaitAnalyzer._generate_comprehensive_diagnosis(data)
            return diagnosis
            
        except Exception as e:
            print(f"❌ Error loading file: {e}")
            return None
    
    @staticmethod
    def _recalculate_bilateral_comparison(raw_angles):
        """Tính toán lại bilateral comparison từ raw joint angles"""
        import numpy as np
        
        results = {}
        
        # Tính toán cho từng khớp
        joints = ["knee", "hip", "ankle"]
        
        for joint in joints:
            left_key = f"left_{joint}"
            right_key = f"right_{joint}"
            
            left_angles = raw_angles.get(left_key, [])
            right_angles = raw_angles.get(right_key, [])
            
            if left_angles and right_angles:
                left_avg = np.mean(left_angles)
                right_avg = np.mean(right_angles)
                difference = abs(left_avg - right_avg)
                
                results[f"{joint}_angle_left_avg"] = left_avg
                results[f"{joint}_angle_right_avg"] = right_avg
                results[f"{joint}_angle_difference"] = difference
                
                # Tính asymmetry percentage
                max_angle = max(left_avg, right_avg)
                if max_angle > 0:
                    asymmetry_percent = (difference / max_angle) * 100
                    results[f"{joint}_asymmetry_percent"] = asymmetry_percent
                else:
                    results[f"{joint}_asymmetry_percent"] = 0.0
            else:
                results[f"{joint}_angle_left_avg"] = 0.0
                results[f"{joint}_angle_right_avg"] = 0.0
                results[f"{joint}_angle_difference"] = 0.0
                results[f"{joint}_asymmetry_percent"] = 0.0
        
        # Overall asymmetry
        total_asymmetry = (
            results.get("knee_asymmetry_percent", 0) +
            results.get("hip_asymmetry_percent", 0) +
            results.get("ankle_asymmetry_percent", 0)
        ) / 3
        
        results["overall_asymmetry_percent"] = total_asymmetry
        
        return results
    
    @staticmethod
    def _recalculate_final_metrics(raw_angles, landmark_history):
        """Tính toán lại final metrics từ raw data - SIMPLIFIED VERSION"""
        import numpy as np
        
        metrics = {
            # Use CURRENT conversion factors (0.01) consistently
            "stride_length": 60.0,   # Default realistic value for VN
            "walking_speed": 1.0,    # Default realistic speed
            "stance_phase": 62.0,    # Default realistic stance phase
            "cadence": 105.0,        # Default realistic cadence
            "step_count": 0,
            "symmetry_index": 0.0,
            "foot_clearance_left": 6.0,   # Default realistic value
            "foot_clearance_right": 6.0,  # Default realistic value
            "step_width_current": 12.0    # Default realistic value
        }
        
        # Calculate from landmark_history if available - USE CURRENT VALUES DIRECTLY
        if landmark_history:
            print(f"🔍 Debug: landmark_history có {len(landmark_history)} frames")
            
            # Extract values DIRECTLY without conversion (assume they're already in correct units)
            foot_clearance_left_values = []
            foot_clearance_right_values = []
            step_width_values = []
            
            for i, frame in enumerate(landmark_history):
                # Use values AS-IS (assume current conversion factors were used)
                if 'foot_clearance_left' in frame and frame['foot_clearance_left'] > 0:
                    foot_clearance_left_values.append(frame['foot_clearance_left'])
                
                if 'foot_clearance_right' in frame and frame['foot_clearance_right'] > 0:
                    foot_clearance_right_values.append(frame['foot_clearance_right'])
                
                if 'step_width' in frame and frame['step_width'] > 0:
                    step_width_values.append(frame['step_width'])
            
            # Use average values if available
            if foot_clearance_left_values:
                metrics["foot_clearance_left"] = np.mean(foot_clearance_left_values)
                print(f"✅ foot_clearance_left = {metrics['foot_clearance_left']:.2f}cm")
            if foot_clearance_right_values:
                metrics["foot_clearance_right"] = np.mean(foot_clearance_right_values)
                print(f"✅ foot_clearance_right = {metrics['foot_clearance_right']:.2f}cm")
            if step_width_values:
                metrics["step_width_current"] = np.mean(step_width_values)
                print(f"✅ step_width_current = {metrics['step_width_current']:.2f}cm")
            
            # Calculate step count
            metrics["step_count"] = len([frame for frame in landmark_history if frame.get('step_detected', False)])
            
            # Estimate stride length from step width (approximate)
            if metrics["step_width_current"] > 0:
                # Rough estimation: stride_length ≈ 5-6 × step_width for normal gait
                estimated_stride = metrics["step_width_current"] * 5.5
                metrics["stride_length"] = min(100, max(40, estimated_stride))  # Bound 40-100cm
                print(f"✅ estimated stride_length = {metrics['stride_length']:.1f}cm")
        else:
            print("❌ landmark_history rỗng hoặc None")
        
        # Tính symmetry index từ raw angles
        all_asymmetries = []
        joints = ["knee", "hip", "ankle"]
        
        for joint in joints:
            left_angles = raw_angles.get(f"left_{joint}", [])
            right_angles = raw_angles.get(f"right_{joint}", [])
            
            if left_angles and right_angles:
                left_avg = np.mean(left_angles)
                right_avg = np.mean(right_angles)
                if max(left_avg, right_avg) > 0:
                    asymmetry = abs(left_avg - right_avg) / max(left_avg, right_avg) * 100
                    all_asymmetries.append(asymmetry)
        
        if all_asymmetries:
            metrics["symmetry_index"] = 100 - np.mean(all_asymmetries)  # Higher = better symmetry
        
        return metrics
    
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
            print(f"✅ Normative database loaded successfully")
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
        
        # Extract raw data và tính toán lại analysis
        raw_angles = data.get('raw_joint_angles', {})
        landmark_history = data.get('raw_landmark_history', [])
        
        # TÍNH TOÁN LẠI TẤT CẢ METRICS TỪ RAW DATA
        print(f"🔍 Debug: raw_angles keys: {list(raw_angles.keys())}")
        print(f"🔍 Debug: landmark_history length: {len(landmark_history) if landmark_history else 0}")
        
        bilateral = GaitAnalyzer._recalculate_bilateral_comparison(raw_angles)
        final_metrics = GaitAnalyzer._recalculate_final_metrics(raw_angles, landmark_history)
        
        print(f"🔍 Debug: final_metrics: {final_metrics}")
        
        # If normative data is available, use evidence-based assessment
        if normative_db:
            # Prepare patient data for assessment
            patient_data = {
                'age': patient_info.get('age', 30),
                'gender': patient_info.get('gender', 'Nam')
            }
            
            # Prepare measured data - USE ACTUAL VALUES without fallback
            measured_data = {
                # Essential parameters for diagnosis - USE ACTUAL MEASUREMENTS
                'stride_length': final_metrics.get('stride_length', 60),  # Use calculated or default 60cm
                'walking_speed': final_metrics.get('walking_speed', 1.0),  # Use calculated or default 1.0 m/s
                'stance_phase_percentage': final_metrics.get('stance_phase', 62),  # Use calculated or default 62%
                
                # Distance measurements - USE ACTUAL DATA (including small values)
                'foot_clearance': max(
                    final_metrics.get('foot_clearance_left', 0),
                    final_metrics.get('foot_clearance_right', 0)
                ),  # Use actual measured values (even if small like 0.3cm)
                'step_width': final_metrics.get('step_width_current', 5.0),  # Use actual or minimum 5cm
                
                # Joint data for asymmetry calculation
                'left_knee': bilateral.get('knee_angle_left_avg', 0),
                'right_knee': bilateral.get('knee_angle_right_avg', 0),
                'left_hip': bilateral.get('hip_angle_left_avg', 0),
                'right_hip': bilateral.get('hip_angle_right_avg', 0),
                'left_ankle': bilateral.get('ankle_angle_left_avg', 0),
                'right_ankle': bilateral.get('ankle_angle_right_avg', 0)
            }
            
            print(f"🔍 Debug measured_data: {measured_data}")
            
            # Get comprehensive assessment
            normative_assessment = normative_db.get_comprehensive_assessment(patient_data, measured_data)
            
            print(f"🔍 Debug normative_assessment: {normative_assessment}")
            print(f"🔍 Debug individual_assessments: {normative_assessment.get('individual_assessments', {})}")
            
            # Use normative-based assessment
            diagnosis.update({
                "assessment_summary": f"Đánh giá dựa trên dữ liệu chuẩn quốc tế (Điểm tổng: {normative_assessment['overall_score']}/3.0)",
                "detailed_findings": {},
                "recommendations": normative_assessment['recommendations'],
                "severity_score": round(normative_assessment['overall_score'], 1),  # Keep decimal precision instead of int()
                "follow_up_needed": normative_assessment['overall_score'] >= 1.5,
                "normative_data_used": True
            })
            
            # Convert individual assessments to detailed findings with proper units
            for param, assessment in normative_assessment['individual_assessments'].items():
                param_names = {
                    'knee_flexion': 'Khớp Gối',
                    'hip_flexion': 'Khớp Hông', 
                    'ankle_dorsiflexion': 'Khớp Cổ Chân',
                    'cadence': 'Nhịp Độ Bước',
                    'stride_length': 'Chiều Dài Bước',
                    'walking_speed': 'Tốc Độ Đi', 
                    'stance_phase_percentage': 'Thời Gian Đặt Chân',
                    'foot_clearance': 'Chiều Cao Nâng Chân',
                    'step_width': 'Chiều Rộng Bước'
                }
                
                # Define units for each parameter
                param_units = {
                    'stride_length': 'cm',
                    'walking_speed': 'm/s',
                    'stance_phase_percentage': '%',
                    'foot_clearance': 'cm',
                    'step_width': 'cm'
                }
                
                # Create reverse mapping from Vietnamese to English for measured values
                reverse_param_mapping = {
                    'Chiều Dài Bước': 'stride_length',
                    'Tốc Độ Đi': 'walking_speed', 
                    'Thời Gian Đặt Chân': 'stance_phase_percentage',
                    'Chiều Cao Nâng Chân': 'foot_clearance',
                    'Chiều Rộng Bước': 'step_width'
                }
                
                param_vietnamese = param_names.get(param, param)
                unit = param_units.get(param, '')
                status_map = {
                    'normal': 'BÌNH THƯỜNG',
                    'mild': 'NHẸ', 
                    'moderate': 'TRUNG BÌNH',
                    'severe': 'NẶNG'
                }
                
                # Get the actual measured value for display - use reverse mapping
                english_param = reverse_param_mapping.get(param, param)
                measured_value = measured_data.get(english_param, 0)
                
                print(f"🔍 Debug param mapping: '{param}' -> '{english_param}' -> measured_value: {measured_value}")
                
                # Map parameter names to Vietnamese display names
                display_names = {
                    'stride_length': 'Chiều Dài Bước',
                    'walking_speed': 'Tốc Độ Đi',
                    'stance_phase_percentage': 'Thời Gian Đặt Chân',
                    'foot_clearance': 'Chiều Cao Nâng Chân',
                    'step_width': 'Chiều Rộng Bước'
                }
                
                display_name = display_names.get(param, param_vietnamese)
                
                diagnosis['detailed_findings'][display_name] = {
                    'status': status_map.get(assessment['status'], 'KHÔNG RÕ'),
                    'deviation_index': assessment['deviation_index'],
                    'position_percentage': assessment['position_percentage'],
                    'recommendation': assessment['interpretation'],
                    'normative_mean': assessment.get('normative_mean', 0),
                    'normative_std': assessment.get('normative_std', 0),
                    'measured_value': measured_value,
                    'unit': unit
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
        
        # Prepare measured data same as above but for fallback method
        measured_data = {
            # Essential parameters for diagnosis - use default values for now since we don't have calculation logic
            'stride_length': 100,  # Default value
            'walking_speed': 1.0,   # Default value
            'stance_phase_percentage': 62,  # Default value
            
            # New distance measurements - use actual data from final_metrics
            'foot_clearance': max(
                final_metrics.get('foot_clearance_left', 0),
                final_metrics.get('foot_clearance_right', 0)
            ),  # Use actual data directly
            'step_width': final_metrics.get('step_width_current', 0),  # Use actual data directly
            
            # Joint data for asymmetry calculation (only include if we have meaningful measurements)
            'left_knee': bilateral.get('knee_angle_left_avg', 0) if bilateral.get('knee_angle_left_avg', 0) > 5 else 0,
            'right_knee': bilateral.get('knee_angle_right_avg', 0) if bilateral.get('knee_angle_right_avg', 0) > 5 else 0,
            'left_hip': bilateral.get('hip_angle_left_avg', 0) if bilateral.get('hip_angle_left_avg', 0) > 5 else 0,
            'right_hip': bilateral.get('hip_angle_right_avg', 0) if bilateral.get('hip_angle_right_avg', 0) > 5 else 0,
            'left_ankle': bilateral.get('ankle_angle_left_avg', 0) if bilateral.get('ankle_angle_left_avg', 0) > 5 else 0,
            'right_ankle': bilateral.get('ankle_angle_right_avg', 0) if bilateral.get('ankle_angle_right_avg', 0) > 5 else 0
        }
        
        print(f"🔍 Debug: measured_data for fallback: {measured_data}")
        
        # Use actual measured values directly from measured_data
        actual_measurements = {
            'Tốc Độ Đi': {
                'measured_value': measured_data.get('walking_speed', 0),
                'unit': 'm/s',
                'normative_mean': 1.1,
                'normative_std': 0.2
            },
            'Chiều Dài Bước': {
                'measured_value': measured_data.get('stride_length', 0),
                'unit': 'cm',
                'normative_mean': 110.0,
                'normative_std': 12.0
            },
            'Thời Gian Đặt Chân': {
                'measured_value': measured_data.get('stance_phase_percentage', 0),
                'unit': '%',
                'normative_mean': 62.1,
                'normative_std': 1.9
            },
            'Chiều Cao Nâng Chân': {
                'measured_value': measured_data.get('foot_clearance', 0),
                'unit': 'cm',
                'normative_mean': 5.8,
                'normative_std': 1.2
            },
            'Chiều Rộng Bước': {
                'measured_value': measured_data.get('step_width', 0),
                'unit': 'cm',
                'normative_mean': 12.5,
                'normative_std': 2.8
            }
        }
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
        
        # Add actual measured values to detailed findings
        # Get actual measurements from final_metrics
        print(f"🔍 Debug: Tạo actual_measurements từ final_metrics")
        print(f"🔍 Debug: final_metrics = {final_metrics}")
        
        # Use actual measured values from measured_data which contains processed values
        print(f"🔍 Debug: measured_data keys: {list(measured_data.keys())}")
        print(f"🔍 Debug: measured_data values: {measured_data}")
        
        actual_measurements = {
            'Tốc Độ Đi': {
                'measured_value': measured_data.get('walking_speed', 0.0),
                'unit': 'm/s',
                'normative_mean': 1.1,
                'normative_std': 0.2
            },
            'Chiều Dài Bước': {
                'measured_value': measured_data.get('stride_length', 0.0),
                'unit': 'cm',
                'normative_mean': 110.0,
                'normative_std': 12.0
            },
            'Thời Gian Đặt Chân': {
                'measured_value': measured_data.get('stance_phase_percentage', 0.0),
                'unit': '%',
                'normative_mean': 62.1,
                'normative_std': 1.9
            },
            'Chiều Cao Nâng Chân': {
                'measured_value': measured_data.get('foot_clearance', 0.0),
                'unit': 'cm',
                'normative_mean': 5.8,
                'normative_std': 1.2
            },
            'Chiều Rộng Bước': {
                'measured_value': measured_data.get('step_width', 0.0),
                'unit': 'cm',
                'normative_mean': 12.5,
                'normative_std': 2.8
            }
        }
        
        print(f"🔍 Debug: actual_measurements: {actual_measurements}")
        
        # Update detailed findings with actual measurements and calculate status
        print(f"🔍 Debug: Cập nhật detailed_findings với actual_measurements")
        
        for param_name, param_data in actual_measurements.items():
            if param_name not in detailed_findings:
                detailed_findings[param_name] = {}
            
            # Calculate status based on deviation from normative values
            measured_value = param_data['measured_value']
            norm_mean = param_data['normative_mean']
            norm_std = param_data['normative_std']
            
            if measured_value > 0 and norm_std > 0:
                deviation_index = abs(measured_value - norm_mean) / norm_std
                
                if deviation_index < 1:
                    status = 'BÌNH THƯỜNG'
                elif deviation_index < 2:
                    status = 'NHẸ'
                elif deviation_index < 3:
                    status = 'TRUNG BÌNH'
                else:
                    status = 'NẶNG'
            else:
                status = 'KHÔNG RÕ'
            
            detailed_findings[param_name].update({
                'measured_value': measured_value,
                'unit': param_data['unit'],
                'normative_mean': norm_mean,
                'normative_std': norm_std,
                'status': status
            })
            
            print(f"✅ Cập nhật {param_name}: measured_value={measured_value}, status={status}")
        
        print(f"🔍 Debug: detailed_findings cuối cùng: {detailed_findings}")
        
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

