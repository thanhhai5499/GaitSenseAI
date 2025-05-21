# Hệ thống Phân tích Dáng đi (Gait Analysis System)

Hệ thống phân tích dáng đi sử dụng camera và các cảm biến gia tốc, góc để nhận diện và phân tích dáng đi của người dùng.

## Tính năng

- Hiển thị 4 camera (trước, sau, trái, phải) trong giao diện toàn màn hình
- Dropdown để chọn camera cho từng vùng hiển thị
- Hỗ trợ camera Intel RealSense D435i và Logitech Brio 300FHD
- Phân tích dáng đi sử dụng model OpenPose
- Hiển thị biểu đồ các thông số dáng đi
- Tích hợp với cảm biến gia tốc và cảm biến góc

## Yêu cầu hệ thống

- Python 3.8 trở lên
- OpenCV 4.5.0 trở lên
- PyQt6
- Matplotlib
- Intel RealSense SDK 2.0 (nếu sử dụng camera RealSense)

## Cài đặt

1. Clone repository:

```
git clone https://github.com/yourusername/gait-analysis-system.git
cd gait-analysis-system
```

2. Cài đặt các thư viện cần thiết:

```
pip install -r requirements.txt
```

3. Tải model OpenPose:

- Tạo thư mục `models`
- Tải file `pose_deploy_linevec.prototxt` và `pose_iter_440000.caffemodel` từ [OpenPose GitHub](https://github.com/CMU-Perceptual-Computing-Lab/openpose/tree/master/models)
- Đặt các file này vào thư mục `models`

## Sử dụng

Chạy chương trình:

```
python main.py
```

## Cấu trúc dự án

- `main.py`: File chính để chạy ứng dụng
- `src/camera/`: Chứa các class khởi tạo camera
  - `base_camera.py`: Class cơ sở cho tất cả các loại camera
  - `realsense_camera.py`: Class cho camera RealSense D435i
  - `logitech_camera.py`: Class cho camera Logitech Brio 300FHD
  - `camera_manager.py`: Quản lý tất cả các camera
- `src/ui/`: Chứa các thành phần giao diện
  - `main_window.py`: Cửa sổ chính của ứng dụng
  - `camera_view.py`: Widget hiển thị camera
  - `gait_analysis_view.py`: Widget hiển thị biểu đồ phân tích dáng đi
- `src/models/`: Chứa các model AI
  - `openpose_model.py`: Tích hợp model OpenPose
  - `gait_analyzer.py`: Phân tích dáng đi từ dữ liệu skeleton
- `src/utils/`: Chứa các tiện ích
  - `sensor_interface.py`: Giao diện cho các cảm biến gia tốc và góc

## Cấu hình camera

- RealSense D435i: 1280x720, 30fps, 60Hz
- Logitech Brio 300FHD: 1920x1080, 30fps, 60Hz
