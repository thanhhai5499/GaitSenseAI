"""
Sensor Interface Module
--------------------
This module provides interfaces for accelerometer and gyroscope sensors.
"""

import numpy as np
import time
from typing import Optional, Dict, List, Tuple, Any, Callable
from abc import ABC, abstractmethod


class SensorInterface(ABC):
    """Abstract base class for sensor interfaces."""
    
    def __init__(self, name: str):
        """
        Initialize the sensor interface.
        
        Args:
            name: Sensor name
        """
        self.name = name
        self.is_connected = False
        self.data_buffer = []
        self.max_buffer_size = 1000
    
    @abstractmethod
    def connect(self) -> bool:
        """
        Connect to the sensor.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the sensor."""
        pass
    
    @abstractmethod
    def read_data(self) -> Optional[np.ndarray]:
        """
        Read data from the sensor.
        
        Returns:
            np.ndarray: Sensor data or None if no data is available
        """
        pass
    
    def add_to_buffer(self, data: np.ndarray) -> None:
        """
        Add data to the buffer.
        
        Args:
            data: Sensor data
        """
        self.data_buffer.append(data)
        
        # Keep buffer size limited
        if len(self.data_buffer) > self.max_buffer_size:
            self.data_buffer = self.data_buffer[-self.max_buffer_size:]
    
    def get_buffer(self) -> List[np.ndarray]:
        """
        Get the data buffer.
        
        Returns:
            List[np.ndarray]: Data buffer
        """
        return self.data_buffer
    
    def clear_buffer(self) -> None:
        """Clear the data buffer."""
        self.data_buffer = []


class AccelerometerInterface(SensorInterface):
    """Interface for accelerometer sensors."""
    
    def __init__(self, name: str, port: str = None):
        """
        Initialize the accelerometer interface.
        
        Args:
            name: Sensor name
            port: Serial port for the sensor
        """
        super().__init__(name)
        self.port = port
        self.serial_connection = None
    
    def connect(self) -> bool:
        """
        Connect to the accelerometer.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            # This is a placeholder for actual serial connection code
            # In a real application, you would use a library like pyserial
            # to connect to the sensor
            print(f"Connecting to accelerometer {self.name} on port {self.port}")
            
            # Simulate connection
            time.sleep(0.5)
            
            self.is_connected = True
            return True
        except Exception as e:
            print(f"Error connecting to accelerometer {self.name}: {e}")
            return False
    
    def disconnect(self) -> None:
        """Disconnect from the accelerometer."""
        if self.is_connected:
            # This is a placeholder for actual serial disconnection code
            print(f"Disconnecting from accelerometer {self.name}")
            
            self.is_connected = False
    
    def read_data(self) -> Optional[np.ndarray]:
        """
        Read data from the accelerometer.
        
        Returns:
            np.ndarray: Accelerometer data (x, y, z) or None if no data is available
        """
        if not self.is_connected:
            return None
        
        try:
            # This is a placeholder for actual data reading code
            # In a real application, you would read data from the serial connection
            
            # Simulate data
            data = np.random.normal(0, 1, 3)  # x, y, z acceleration
            
            # Add data to buffer
            self.add_to_buffer(data)
            
            return data
        except Exception as e:
            print(f"Error reading data from accelerometer {self.name}: {e}")
            return None


class GyroscopeInterface(SensorInterface):
    """Interface for gyroscope sensors."""
    
    def __init__(self, name: str, port: str = None):
        """
        Initialize the gyroscope interface.
        
        Args:
            name: Sensor name
            port: Serial port for the sensor
        """
        super().__init__(name)
        self.port = port
        self.serial_connection = None
    
    def connect(self) -> bool:
        """
        Connect to the gyroscope.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            # This is a placeholder for actual serial connection code
            # In a real application, you would use a library like pyserial
            # to connect to the sensor
            print(f"Connecting to gyroscope {self.name} on port {self.port}")
            
            # Simulate connection
            time.sleep(0.5)
            
            self.is_connected = True
            return True
        except Exception as e:
            print(f"Error connecting to gyroscope {self.name}: {e}")
            return False
    
    def disconnect(self) -> None:
        """Disconnect from the gyroscope."""
        if self.is_connected:
            # This is a placeholder for actual serial disconnection code
            print(f"Disconnecting from gyroscope {self.name}")
            
            self.is_connected = False
    
    def read_data(self) -> Optional[np.ndarray]:
        """
        Read data from the gyroscope.
        
        Returns:
            np.ndarray: Gyroscope data (roll, pitch, yaw) or None if no data is available
        """
        if not self.is_connected:
            return None
        
        try:
            # This is a placeholder for actual data reading code
            # In a real application, you would read data from the serial connection
            
            # Simulate data
            data = np.random.normal(0, 1, 3)  # roll, pitch, yaw
            
            # Add data to buffer
            self.add_to_buffer(data)
            
            return data
        except Exception as e:
            print(f"Error reading data from gyroscope {self.name}: {e}")
            return None


class SensorManager:
    """Class for managing multiple sensors."""
    
    def __init__(self):
        """Initialize the sensor manager."""
        self.sensors = {}
    
    def add_sensor(self, sensor_id: str, sensor: SensorInterface) -> None:
        """
        Add a sensor to the manager.
        
        Args:
            sensor_id: Sensor ID
            sensor: Sensor interface
        """
        self.sensors[sensor_id] = sensor
    
    def remove_sensor(self, sensor_id: str) -> None:
        """
        Remove a sensor from the manager.
        
        Args:
            sensor_id: Sensor ID
        """
        if sensor_id in self.sensors:
            self.sensors[sensor_id].disconnect()
            del self.sensors[sensor_id]
    
    def connect_sensor(self, sensor_id: str) -> bool:
        """
        Connect to a sensor.
        
        Args:
            sensor_id: Sensor ID
            
        Returns:
            bool: True if connection successful, False otherwise
        """
        if sensor_id in self.sensors:
            return self.sensors[sensor_id].connect()
        return False
    
    def disconnect_sensor(self, sensor_id: str) -> None:
        """
        Disconnect from a sensor.
        
        Args:
            sensor_id: Sensor ID
        """
        if sensor_id in self.sensors:
            self.sensors[sensor_id].disconnect()
    
    def disconnect_all_sensors(self) -> None:
        """Disconnect from all sensors."""
        for sensor in self.sensors.values():
            sensor.disconnect()
    
    def read_sensor_data(self, sensor_id: str) -> Optional[np.ndarray]:
        """
        Read data from a sensor.
        
        Args:
            sensor_id: Sensor ID
            
        Returns:
            np.ndarray: Sensor data or None if no data is available
        """
        if sensor_id in self.sensors:
            return self.sensors[sensor_id].read_data()
        return None
    
    def get_sensor_buffer(self, sensor_id: str) -> List[np.ndarray]:
        """
        Get the data buffer for a sensor.
        
        Args:
            sensor_id: Sensor ID
            
        Returns:
            List[np.ndarray]: Data buffer
        """
        if sensor_id in self.sensors:
            return self.sensors[sensor_id].get_buffer()
        return []
    
    def clear_sensor_buffer(self, sensor_id: str) -> None:
        """
        Clear the data buffer for a sensor.
        
        Args:
            sensor_id: Sensor ID
        """
        if sensor_id in self.sensors:
            self.sensors[sensor_id].clear_buffer()
    
    def clear_all_buffers(self) -> None:
        """Clear all data buffers."""
        for sensor in self.sensors.values():
            sensor.clear_buffer()
