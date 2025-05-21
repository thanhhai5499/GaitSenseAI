"""
MediaPipe Pose Model Module
--------------------------
This module provides a class for using the MediaPipe pose detection model.
"""

import cv2
import numpy as np
import mediapipe as mp
import time
from typing import Optional, Dict, List, Tuple, Any

class MediaPipePoseModel:
    """Class for using the MediaPipe pose detection model."""
    
    # MediaPipe pose landmarks
    POSE_LANDMARKS = {
        # Face
        "NOSE": 0,
        "LEFT_EYE_INNER": 1,
        "LEFT_EYE": 2,
        "LEFT_EYE_OUTER": 3,
        "RIGHT_EYE_INNER": 4,
        "RIGHT_EYE": 5,
        "RIGHT_EYE_OUTER": 6,
        "LEFT_EAR": 7,
        "RIGHT_EAR": 8,
        "MOUTH_LEFT": 9,
        "MOUTH_RIGHT": 10,
        # Upper body
        "LEFT_SHOULDER": 11,
        "RIGHT_SHOULDER": 12,
        "LEFT_ELBOW": 13,
        "RIGHT_ELBOW": 14,
        "LEFT_WRIST": 15,
        "RIGHT_WRIST": 16,
        # Torso
        "LEFT_PINKY": 17,
        "RIGHT_PINKY": 18,
        "LEFT_INDEX": 19,
        "RIGHT_INDEX": 20,
        "LEFT_THUMB": 21,
        "RIGHT_THUMB": 22,
        "LEFT_HIP": 23,
        "RIGHT_HIP": 24,
        # Lower body
        "LEFT_KNEE": 25,
        "RIGHT_KNEE": 26,
        "LEFT_ANKLE": 27,
        "RIGHT_ANKLE": 28,
        "LEFT_HEEL": 29,
        "RIGHT_HEEL": 30,
        "LEFT_FOOT_INDEX": 31,
        "RIGHT_FOOT_INDEX": 32
    }
    
    # Pairs of joints for drawing the leg skeleton
    LEG_CONNECTIONS = [
        ("LEFT_HIP", "LEFT_KNEE"),
        ("LEFT_KNEE", "LEFT_ANKLE"),
        ("LEFT_ANKLE", "LEFT_HEEL"),
        ("LEFT_HEEL", "LEFT_FOOT_INDEX"),
        ("RIGHT_HIP", "RIGHT_KNEE"),
        ("RIGHT_KNEE", "RIGHT_ANKLE"),
        ("RIGHT_ANKLE", "RIGHT_HEEL"),
        ("RIGHT_HEEL", "RIGHT_FOOT_INDEX"),
        ("LEFT_HIP", "RIGHT_HIP")
    ]
    
    # Full body connections for visualization
    FULL_BODY_CONNECTIONS = [
        # Face
        ("NOSE", "LEFT_EYE_INNER"),
        ("LEFT_EYE_INNER", "LEFT_EYE"),
        ("LEFT_EYE", "LEFT_EYE_OUTER"),
        ("LEFT_EYE_OUTER", "LEFT_EAR"),
        ("NOSE", "RIGHT_EYE_INNER"),
        ("RIGHT_EYE_INNER", "RIGHT_EYE"),
        ("RIGHT_EYE", "RIGHT_EYE_OUTER"),
        ("RIGHT_EYE_OUTER", "RIGHT_EAR"),
        # Upper body
        ("LEFT_SHOULDER", "RIGHT_SHOULDER"),
        ("LEFT_SHOULDER", "LEFT_ELBOW"),
        ("LEFT_ELBOW", "LEFT_WRIST"),
        ("RIGHT_SHOULDER", "RIGHT_ELBOW"),
        ("RIGHT_ELBOW", "RIGHT_WRIST"),
        # Hands
        ("LEFT_WRIST", "LEFT_PINKY"),
        ("LEFT_WRIST", "LEFT_INDEX"),
        ("LEFT_WRIST", "LEFT_THUMB"),
        ("RIGHT_WRIST", "RIGHT_PINKY"),
        ("RIGHT_WRIST", "RIGHT_INDEX"),
        ("RIGHT_WRIST", "RIGHT_THUMB"),
        # Torso
        ("LEFT_SHOULDER", "LEFT_HIP"),
        ("RIGHT_SHOULDER", "RIGHT_HIP"),
        # Legs (already defined in LEG_CONNECTIONS)
    ]
    
    def __init__(self, model_complexity: int = 1, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        """
        Initialize the MediaPipe pose model.
        
        Args:
            model_complexity: Model complexity (0=Lite, 1=Full, 2=Heavy)
            min_detection_confidence: Minimum confidence for detection
            min_tracking_confidence: Minimum confidence for tracking
        """
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_pose = mp.solutions.pose
        
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        
        self.pose = None
        self.initialized = False
        self.last_process_time = 0
        self.processing_fps = 0
        
        # Initialize the model
        self.initialize()
    
    def initialize(self) -> bool:
        """
        Initialize the MediaPipe pose model.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self.pose = self.mp_pose.Pose(
                model_complexity=self.model_complexity,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
                static_image_mode=False
            )
            
            self.initialized = True
            return True
        except Exception as e:
            print(f"Error initializing MediaPipe pose model: {e}")
            return False
    
    def detect_pose(self, frame: np.ndarray, draw_full_body: bool = False) -> Tuple[Dict[str, Tuple[float, float, float]], np.ndarray]:
        """
        Detect pose in a frame.
        
        Args:
            frame: Input frame
            draw_full_body: Whether to draw the full body skeleton or just legs
            
        Returns:
            Tuple[Dict[str, Tuple[float, float, float]], np.ndarray]: 
                Detected landmarks and output frame with visualization
        """
        if not self.initialized or self.pose is None:
            return {}, frame
        
        start_time = time.time()
        
        try:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame
            results = self.pose.process(frame_rgb)
            
            # Create output frame
            output_frame = frame.copy()
            
            # Initialize landmarks dictionary
            landmarks_dict = {}
            
            if results.pose_landmarks:
                # Extract landmarks
                for landmark_name, landmark_idx in self.POSE_LANDMARKS.items():
                    landmark = results.pose_landmarks.landmark[landmark_idx]
                    landmarks_dict[landmark_name] = (landmark.x, landmark.y, landmark.visibility)
                
                # Draw skeleton
                if draw_full_body:
                    # Draw full body using MediaPipe's built-in function
                    self.mp_drawing.draw_landmarks(
                        output_frame,
                        results.pose_landmarks,
                        self.mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                    )
                else:
                    # Draw only leg skeleton
                    self._draw_leg_skeleton(output_frame, landmarks_dict)
            
            # Calculate processing FPS
            process_time = time.time() - start_time
            self.last_process_time = process_time
            self.processing_fps = 1.0 / process_time if process_time > 0 else 0
            
            return landmarks_dict, output_frame
        except Exception as e:
            print(f"Error detecting pose: {e}")
            return {}, frame
    
    def _draw_leg_skeleton(self, frame: np.ndarray, landmarks_dict: Dict[str, Tuple[float, float, float]]) -> None:
        """
        Draw the leg skeleton on the frame.
        
        Args:
            frame: Frame to draw on
            landmarks_dict: Dictionary of landmarks
        """
        h, w, _ = frame.shape
        
        for connection in self.LEG_CONNECTIONS:
            start_point_name, end_point_name = connection
            
            if start_point_name in landmarks_dict and end_point_name in landmarks_dict:
                start_point = landmarks_dict[start_point_name]
                end_point = landmarks_dict[end_point_name]
                
                # Check visibility
                if start_point[2] > 0.5 and end_point[2] > 0.5:
                    # Convert normalized coordinates to pixel coordinates
                    start_x, start_y = int(start_point[0] * w), int(start_point[1] * h)
                    end_x, end_y = int(end_point[0] * w), int(end_point[1] * h)
                    
                    # Draw line
                    cv2.line(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 3)
                    
                    # Draw joints
                    cv2.circle(frame, (start_x, start_y), 5, (0, 0, 255), -1)
                    cv2.circle(frame, (end_x, end_y), 5, (0, 0, 255), -1)
    
    def get_processing_info(self) -> Dict[str, float]:
        """
        Get processing information.
        
        Returns:
            Dict[str, float]: Processing information
        """
        return {
            "processing_time": self.last_process_time,
            "processing_fps": self.processing_fps
        }
    
    def release(self) -> None:
        """Release resources."""
        if self.pose:
            self.pose.close()
            self.pose = None
            self.initialized = False
