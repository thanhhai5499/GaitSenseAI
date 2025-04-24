"""
OpenPose Model Module
-------------------
This module provides a class for using the OpenPose model for pose estimation.
"""

import cv2
import numpy as np
import os
from typing import Optional, Dict, List, Tuple, Any

# Check if OpenCV DNN module is available
OPENCV_DNN_AVAILABLE = hasattr(cv2, 'dnn')


class OpenPoseModel:
    """Class for using the OpenPose model for pose estimation."""
    
    # COCO body parts
    BODY_PARTS = {
        "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
        "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
        "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
        "LEye": 15, "REar": 16, "LEar": 17, "Background": 18
    }
    
    # Pairs of joints for drawing
    POSE_PAIRS = [
        ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
        ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
        ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
        ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
        ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]
    ]
    
    def __init__(self, model_path: str = "models"):
        """
        Initialize the OpenPose model.
        
        Args:
            model_path: Path to the directory containing the model files
        """
        self.model_path = model_path
        self.net = None
        self.initialized = False
        
        # Initialize the model
        self.initialize()
    
    def initialize(self) -> bool:
        """
        Initialize the OpenPose model.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        if not OPENCV_DNN_AVAILABLE:
            print("OpenCV DNN module is not available. Please install OpenCV with DNN support.")
            return False
        
        try:
            # Check if model files exist
            proto_file = os.path.join(self.model_path, "pose_deploy_linevec.prototxt")
            weights_file = os.path.join(self.model_path, "pose_iter_440000.caffemodel")
            
            if not os.path.isfile(proto_file) or not os.path.isfile(weights_file):
                print(f"Model files not found at {self.model_path}")
                print("Please download the model files from:")
                print("https://github.com/CMU-Perceptual-Computing-Lab/openpose/tree/master/models")
                return False
            
            # Load the model
            self.net = cv2.dnn.readNetFromCaffe(proto_file, weights_file)
            
            # Check if CUDA is available
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            
            self.initialized = True
            return True
        except Exception as e:
            print(f"Error initializing OpenPose model: {e}")
            return False
    
    def detect_pose(self, frame: np.ndarray, threshold: float = 0.1) -> Tuple[List[List[float]], np.ndarray]:
        """
        Detect pose in a frame.
        
        Args:
            frame: Input frame
            threshold: Confidence threshold for keypoints
            
        Returns:
            Tuple[List[List[float]], np.ndarray]: Detected keypoints and output frame
        """
        if not self.initialized or self.net is None:
            return [], frame
        
        try:
            # Prepare input
            input_width = 368
            input_height = 368
            inp_blob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (input_width, input_height),
                                            (0, 0, 0), swapRB=False, crop=False)
            
            # Set input
            self.net.setInput(inp_blob)
            
            # Forward pass
            output = self.net.forward()
            
            # Get frame dimensions
            frame_height, frame_width = frame.shape[:2]
            
            # Initialize keypoints list
            keypoints = []
            
            # Create output frame
            output_frame = frame.copy()
            
            # Process keypoints
            for i in range(len(self.BODY_PARTS) - 1):  # Exclude background
                # Confidence map for the body part
                prob_map = output[0, i, :, :]
                
                # Find global maxima of the probability map
                _, prob, _, point = cv2.minMaxLoc(prob_map)
                
                # Scale the point to the original frame
                x = int((frame_width * point[0]) / output.shape[3])
                y = int((frame_height * point[1]) / output.shape[2])
                
                # Add keypoint if confidence is above threshold
                if prob > threshold:
                    cv2.circle(output_frame, (x, y), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
                    cv2.putText(output_frame, f"{i}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)
                    keypoints.append([x, y])
                else:
                    keypoints.append(None)
            
            # Draw skeleton
            for pair in self.POSE_PAIRS:
                part_a = self.BODY_PARTS[pair[0]]
                part_b = self.BODY_PARTS[pair[1]]
                
                if keypoints[part_a] and keypoints[part_b]:
                    cv2.line(output_frame, tuple(keypoints[part_a]), tuple(keypoints[part_b]), (0, 255, 0), 3)
            
            return keypoints, output_frame
        except Exception as e:
            print(f"Error detecting pose: {e}")
            return [], frame
    
    def calculate_joint_angles(self, keypoints: List[List[float]]) -> Dict[str, float]:
        """
        Calculate joint angles from keypoints.
        
        Args:
            keypoints: List of keypoints
            
        Returns:
            Dict[str, float]: Dictionary of joint angles
        """
        angles = {}
        
        # Check if keypoints are valid
        if not keypoints or len(keypoints) < 14:  # Need at least 14 keypoints for basic angles
            return angles
        
        # Calculate hip angle (between torso and thigh)
        if keypoints[self.BODY_PARTS["Neck"]] and keypoints[self.BODY_PARTS["RHip"]] and keypoints[self.BODY_PARTS["RKnee"]]:
            neck = np.array(keypoints[self.BODY_PARTS["Neck"]])
            hip = np.array(keypoints[self.BODY_PARTS["RHip"]])
            knee = np.array(keypoints[self.BODY_PARTS["RKnee"]])
            
            angles["hip"] = self._calculate_angle(neck, hip, knee)
        
        # Calculate knee angle
        if keypoints[self.BODY_PARTS["RHip"]] and keypoints[self.BODY_PARTS["RKnee"]] and keypoints[self.BODY_PARTS["RAnkle"]]:
            hip = np.array(keypoints[self.BODY_PARTS["RHip"]])
            knee = np.array(keypoints[self.BODY_PARTS["RKnee"]])
            ankle = np.array(keypoints[self.BODY_PARTS["RAnkle"]])
            
            angles["knee"] = self._calculate_angle(hip, knee, ankle)
        
        # Calculate ankle angle
        if keypoints[self.BODY_PARTS["RKnee"]] and keypoints[self.BODY_PARTS["RAnkle"]]:
            knee = np.array(keypoints[self.BODY_PARTS["RKnee"]])
            ankle = np.array(keypoints[self.BODY_PARTS["RAnkle"]])
            
            # Assuming the foot is horizontal
            foot = np.array([ankle[0] + 100, ankle[1]])
            
            angles["ankle"] = self._calculate_angle(knee, ankle, foot)
        
        return angles
    
    def _calculate_angle(self, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        """
        Calculate the angle between three points.
        
        Args:
            a: First point
            b: Second point (vertex)
            c: Third point
            
        Returns:
            float: Angle in degrees
        """
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        
        return np.degrees(angle)
