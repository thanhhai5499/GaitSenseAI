import logging
import time
import sys
import os
import traceback

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Add src directory to path
sys.path.append(os.path.abspath('.'))

# Import camera detector
from src.camera.camera_detector import CameraDetector

def main():
    try:
        print("Testing camera detection...")
        
        # Create camera detector
        print("Creating camera detector...")
        detector = CameraDetector()
        
        # Start detection
        print("Starting camera detection...")
        start_time = time.time()
        detector.detect_cameras_once()
        
        # Wait for detection to complete
        print("Waiting for detection to complete...")
        detector.detection_complete.wait(timeout=30.0)
        end_time = time.time()
        
        # Get available cameras
        print("Getting available cameras...")
        cameras = detector.get_available_cameras()
        
        # Print results
        print(f"\nDetection completed in {end_time - start_time:.2f} seconds")
        print(f"Found {len(cameras)} cameras:")
        
        for i, camera in enumerate(cameras):
            print(f"\nCamera {i+1}:")
            print(f"  ID: {camera['id']}")
            print(f"  Name: {camera['name']}")
            print(f"  Type: {camera['type']}")
            print(f"  Resolution: {camera['width']}x{camera['height']}@{camera['fps']}fps")
            
            # Print all available keys
            print(f"  All keys: {list(camera.keys())}")
            
            # Print instance_id if available
            if 'instance_id' in camera:
                print(f"  Instance ID: {camera['instance_id']}")
        
        # Test creating a camera
        if cameras:
            print("\nTesting camera creation...")
            camera_id = cameras[0]['id']
            print(f"Creating camera with ID: {camera_id}")
            camera = detector.create_camera(camera_id)
            
            if camera:
                print(f"Successfully created camera: {camera}")
                
                # Test connecting to the camera
                print("Testing camera connection...")
                if camera.connect():
                    print("Successfully connected to camera")
                    
                    # Get a test frame
                    print("Getting test frame...")
                    frame = camera.get_frame()
                    if frame is not None:
                        print(f"Successfully got frame with shape: {frame.shape}")
                    else:
                        print("Failed to get frame")
                    
                    # Disconnect
                    print("Disconnecting from camera...")
                    camera.disconnect()
                    print("Disconnected from camera")
                else:
                    print("Failed to connect to camera")
            else:
                print(f"Failed to create camera with ID: {camera_id}")
    except Exception as e:
        print(f"Error in main function: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
