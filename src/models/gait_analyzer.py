"""
Gait Analyzer Module
------------------
This module provides a class for analyzing gait from pose data.
"""

import numpy as np
from typing import Optional, Dict, List, Tuple, Any
import time


class GaitAnalyzer:
    """Class for analyzing gait from pose data."""
    
    def __init__(self):
        """Initialize the gait analyzer."""
        # Initialize data structures
        self.joint_angles_history = {
            'hip': [],
            'knee': [],
            'ankle': []
        }
        
        self.gait_parameters = {
            'stride_length': [],
            'step_length': [],
            'cadence': [],
            'walking_speed': [],
            'stance_time': [],
            'swing_time': []
        }
        
        # Initialize state variables
        self.last_heel_strike_time = None
        self.last_toe_off_time = None
        self.is_stance_phase = True
        self.step_count = 0
        self.step_times = []
        self.last_position = None
        self.current_position = None
    
    def update_joint_angles(self, angles: Dict[str, float]) -> None:
        """
        Update joint angle history.
        
        Args:
            angles: Dictionary of joint angles
        """
        for joint, angle in angles.items():
            if joint in self.joint_angles_history:
                self.joint_angles_history[joint].append(angle)
                
                # Keep only the last 100 values
                if len(self.joint_angles_history[joint]) > 100:
                    self.joint_angles_history[joint] = self.joint_angles_history[joint][-100:]
    
    def update_position(self, keypoints: List[List[float]]) -> None:
        """
        Update position based on keypoints.
        
        Args:
            keypoints: List of keypoints
        """
        # Check if keypoints are valid
        if not keypoints or len(keypoints) < 14:  # Need at least 14 keypoints for basic analysis
            return
        
        # Update position
        self.last_position = self.current_position
        
        # Use the midpoint between ankles as the current position
        left_ankle_idx = 13  # LAnkle
        right_ankle_idx = 10  # RAnkle
        
        if keypoints[left_ankle_idx] and keypoints[right_ankle_idx]:
            left_ankle = np.array(keypoints[left_ankle_idx])
            right_ankle = np.array(keypoints[right_ankle_idx])
            self.current_position = (left_ankle + right_ankle) / 2
        elif keypoints[left_ankle_idx]:
            self.current_position = np.array(keypoints[left_ankle_idx])
        elif keypoints[right_ankle_idx]:
            self.current_position = np.array(keypoints[right_ankle_idx])
        else:
            return
        
        # Detect gait events
        self._detect_gait_events(keypoints)
        
        # Calculate gait parameters
        if self.last_position is not None:
            self._calculate_gait_parameters()
    
    def _detect_gait_events(self, keypoints: List[List[float]]) -> None:
        """
        Detect gait events (heel strike, toe off).
        
        Args:
            keypoints: List of keypoints
        """
        # This is a simplified detection based on ankle velocity
        # In a real application, you would use more sophisticated methods
        
        left_ankle_idx = 13  # LAnkle
        right_ankle_idx = 10  # RAnkle
        left_knee_idx = 12  # LKnee
        right_knee_idx = 9  # RKnee
        
        current_time = time.time()
        
        # Detect heel strike (when the ankle is in front and moving downward)
        if (keypoints[left_ankle_idx] and keypoints[left_knee_idx]) or (keypoints[right_ankle_idx] and keypoints[right_knee_idx]):
            # Simplified detection: when the knee is extended
            if self.is_stance_phase is False:
                self.is_stance_phase = True
                self.last_heel_strike_time = current_time
                self.step_count += 1
                
                # Calculate step time
                if self.last_toe_off_time is not None:
                    step_time = current_time - self.last_toe_off_time
                    self.step_times.append(step_time)
                    
                    # Keep only the last 10 step times
                    if len(self.step_times) > 10:
                        self.step_times = self.step_times[-10:]
        
        # Detect toe off (when the ankle is behind and moving upward)
        if (keypoints[left_ankle_idx] and keypoints[left_knee_idx]) or (keypoints[right_ankle_idx] and keypoints[right_knee_idx]):
            # Simplified detection: when the knee is flexed
            if self.is_stance_phase is True:
                self.is_stance_phase = False
                self.last_toe_off_time = current_time
                
                # Calculate stance time
                if self.last_heel_strike_time is not None:
                    stance_time = current_time - self.last_heel_strike_time
                    self.gait_parameters['stance_time'].append(stance_time)
                    
                    # Keep only the last 10 values
                    if len(self.gait_parameters['stance_time']) > 10:
                        self.gait_parameters['stance_time'] = self.gait_parameters['stance_time'][-10:]
    
    def _calculate_gait_parameters(self) -> None:
        """Calculate gait parameters."""
        # Calculate stride length (distance between consecutive heel strikes of the same foot)
        # This is a simplified calculation
        if self.last_position is not None and self.current_position is not None:
            distance = np.linalg.norm(self.current_position - self.last_position)
            
            # Convert pixels to meters (assuming 100 pixels = 1 meter)
            # This would need to be calibrated in a real application
            distance_meters = distance / 100.0
            
            # Add to stride length history
            self.gait_parameters['stride_length'].append(distance_meters)
            
            # Keep only the last 10 values
            if len(self.gait_parameters['stride_length']) > 10:
                self.gait_parameters['stride_length'] = self.gait_parameters['stride_length'][-10:]
            
            # Calculate step length (half of stride length for simplicity)
            step_length = distance_meters / 2.0
            self.gait_parameters['step_length'].append(step_length)
            
            # Keep only the last 10 values
            if len(self.gait_parameters['step_length']) > 10:
                self.gait_parameters['step_length'] = self.gait_parameters['step_length'][-10:]
        
        # Calculate cadence (steps per minute)
        if len(self.step_times) > 0:
            avg_step_time = np.mean(self.step_times)
            cadence = 60.0 / avg_step_time  # steps per minute
            self.gait_parameters['cadence'].append(cadence)
            
            # Keep only the last 10 values
            if len(self.gait_parameters['cadence']) > 10:
                self.gait_parameters['cadence'] = self.gait_parameters['cadence'][-10:]
        
        # Calculate walking speed
        if len(self.gait_parameters['stride_length']) > 0 and len(self.step_times) > 0:
            avg_stride_length = np.mean(self.gait_parameters['stride_length'])
            avg_step_time = np.mean(self.step_times)
            walking_speed = avg_stride_length / (2 * avg_step_time)  # meters per second
            self.gait_parameters['walking_speed'].append(walking_speed)
            
            # Keep only the last 10 values
            if len(self.gait_parameters['walking_speed']) > 10:
                self.gait_parameters['walking_speed'] = self.gait_parameters['walking_speed'][-10:]
        
        # Calculate swing time (time between toe off and heel strike)
        if self.last_heel_strike_time is not None and self.last_toe_off_time is not None:
            if self.last_heel_strike_time > self.last_toe_off_time:
                swing_time = self.last_heel_strike_time - self.last_toe_off_time
                self.gait_parameters['swing_time'].append(swing_time)
                
                # Keep only the last 10 values
                if len(self.gait_parameters['swing_time']) > 10:
                    self.gait_parameters['swing_time'] = self.gait_parameters['swing_time'][-10:]
    
    def get_joint_angles(self) -> Dict[str, List[float]]:
        """
        Get joint angle history.
        
        Returns:
            Dict[str, List[float]]: Dictionary of joint angle history
        """
        return self.joint_angles_history
    
    def get_gait_parameters(self) -> Dict[str, List[float]]:
        """
        Get gait parameters.
        
        Returns:
            Dict[str, List[float]]: Dictionary of gait parameters
        """
        return self.gait_parameters
    
    def clear_data(self) -> None:
        """Clear all data."""
        self.joint_angles_history = {
            'hip': [],
            'knee': [],
            'ankle': []
        }
        
        self.gait_parameters = {
            'stride_length': [],
            'step_length': [],
            'cadence': [],
            'walking_speed': [],
            'stance_time': [],
            'swing_time': []
        }
        
        self.last_heel_strike_time = None
        self.last_toe_off_time = None
        self.is_stance_phase = True
        self.step_count = 0
        self.step_times = []
        self.last_position = None
        self.current_position = None
