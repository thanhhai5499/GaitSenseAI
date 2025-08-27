#!/usr/bin/env python3
"""
Launch script for GaitSenseAI application
"""

import sys
import os
import subprocess

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    return True

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'PyQt6', 'opencv-python', 'numpy', 'matplotlib',
        'mediapipe', 'scipy', 'scikit-learn', 'Pillow'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_').lower())
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n💡 Install missing packages with:")
        print("   pip install -r requirements.txt")
        return False
    
    return True

def install_dependencies():
    """Install dependencies from requirements.txt"""
    if not os.path.exists('requirements.txt'):
        print("❌ requirements.txt not found")
        return False
    
    print("📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def check_camera():
    """Check if camera is available"""
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("✅ Camera detected")
            cap.release()
            return True
        else:
            print("⚠️  No camera detected - you can still run the app but camera features won't work")
            return True
    except Exception as e:
        print(f"⚠️  Camera check failed: {e}")
        return True  # Don't block app launch

def main():
    """Main launcher function"""
    print("🚀 GaitSenseAI Launcher")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return 1
    
    # Check dependencies
    if not check_dependencies():
        response = input("\n❓ Install missing dependencies? (y/n): ").lower().strip()
        if response == 'y':
            if not install_dependencies():
                return 1
            print("🔄 Please run the launcher again after installation")
            return 0
        else:
            print("❌ Cannot proceed without required dependencies")
            return 1
    
    # Check camera
    check_camera()
    
    # Launch application
    print("\n🎯 Launching GaitSenseAI...")
    try:
        # Import and run main application
        from main import main as app_main
        app_main()
        return 0
    except KeyboardInterrupt:
        print("\n👋 Application closed by user")
        return 0
    except Exception as e:
        print(f"\n❌ Application error: {e}")
        print("\n🐛 Debug information:")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())

