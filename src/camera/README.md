# Camera System

This directory contains the camera management system for the Gait Analysis application.

## Overview

The camera system is designed to handle multiple cameras simultaneously using multithreading to prevent stream congestion and frame freezing. It automatically initializes and starts all configured cameras upon application startup.

## Camera Types

The system supports the following camera types:

1. **Intel RealSense D435i** (Front Camera)
   - Resolution: 1280x720
   - FPS: 30
   - Refresh Rate: 60Hz

2. **Logitech Brio 300** (Left Camera)
   - Resolution: 1920x1080
   - FPS: 30
   - Refresh Rate: 60Hz

3. **Logitech Brio 300** (Right Camera)
   - Resolution: 1920x1080
   - FPS: 30
   - Refresh Rate: 60Hz

4. **Placeholder** (Back Camera)
   - Currently not assigned to any physical camera

## Architecture

The camera system consists of the following components:

- **ThreadedCameraManager**: Main class that manages all cameras and provides a unified interface for the UI
- **CameraThread**: Thread class for handling individual camera streams
- **BaseCamera**: Abstract base class for all camera types
- **RealSenseCamera**: Implementation for Intel RealSense cameras
- **LogitechCamera**: Implementation for Logitech cameras
- **PlaceholderCamera**: Implementation for positions without a physical camera

## Configuration

Camera configurations are defined in `camera_config.py`. To modify camera settings or add new cameras, edit this file.

## Usage

The camera manager is automatically initialized when the application starts. It creates separate threads for each camera and starts capturing frames.

To access the camera manager from code:

```python
from src.camera import ThreadedCameraManager

# Get the singleton instance
camera_manager = ThreadedCameraManager.get_instance()

# Get a frame from a specific camera
frame = camera_manager.get_frame("front")

# Check if a camera is active
is_active = camera_manager.is_camera_active("left")

# Disconnect a camera
camera_manager.disconnect_camera("right")

# Disconnect all cameras
camera_manager.disconnect_all_cameras()
```

## Thread Safety

The camera system is designed to be thread-safe. Each camera runs in its own thread, and frame access is managed through thread-safe queues.

## Error Handling

The system includes comprehensive error handling to manage camera initialization and streaming issues. If a camera fails to initialize or encounters an error during streaming, it will log the error and continue operating with the available cameras.

## Performance Considerations

- The system uses a small delay between camera initializations to prevent USB bandwidth issues
- Frame queues have a maximum size to prevent memory issues
- Old frames are automatically discarded if not consumed in time
