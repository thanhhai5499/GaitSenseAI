import sys
import logging
from PyQt6.QtWidgets import QApplication

from src.ui.main_window import MainWindow
from src.camera.threaded_camera_manager import ThreadedCameraManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def main():
    # Initialize the camera manager before creating the UI
    # This will automatically start all cameras in separate threads
    camera_manager = ThreadedCameraManager.get_instance()

    # Create and show the application
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()

    # Clean up cameras when the application exits
    app.aboutToQuit.connect(camera_manager.disconnect_all_cameras)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
