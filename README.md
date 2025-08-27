# GaitSenseAI - Hệ thống phân tích dáng đi thời gian thực

GaitSenseAI là một ứng dụng phân tích dáng đi sử dụng computer vision và machine learning để đánh giá các thông số gait từ camera RGB thời gian thực.

## 🌟 Tính năng chính

- **Camera RGB HD**: Hỗ trợ camera 1920x1080@60fps
- **Phân tích dáng đi real-time**: Sử dụng MediaPipe để phát hiện pose
- **Giao diện người dùng hiện đại**: Thiết kế dark theme tương tự Intel RealSense Viewer
- **Thông số dáng đi chi tiết**:
  - Số bước
  - Tần số bước (cadence)
  - Độ dài bước
  - Thời gian bước
  - Tốc độ đi bộ
  - Góc chân trái/phải
- **Xuất dữ liệu**: Hỗ trợ xuất CSV, JSON, Excel

## 📁 Cấu trúc thư mục

```
GaitSenseAI/
├── main.py                 # File chính của ứng dụng
├── requirements.txt        # Danh sách thư viện cần thiết
├── README.md              # Hướng dẫn sử dụng
├── src/                   # Mã nguồn chính
│   ├── __init__.py
│   ├── core/              # Modules cốt lõi
│   │   ├── __init__.py
│   │   ├── camera_handler.py    # Xử lý camera RGB
│   │   └── gait_analyzer.py     # Phân tích dáng đi
│   ├── ui/                # Components giao diện
│   │   ├── __init__.py
│   │   ├── sidebar.py           # Sidebar hiển thị thông số
│   │   └── camera_panel.py      # Panel hiển thị camera
│   └── utils/             # Tiện ích
│       ├── __init__.py
│       └── helpers.py           # Functions hỗ trợ
├── assets/                # Tài nguyên
│   └── styles/
│       ├── __init__.py
│       └── dark_theme.py        # Theme tối
├── config/                # Cấu hình
│   ├── __init__.py
│   └── settings.py              # Cài đặt ứng dụng
└── models/                # Models AI
    └── pose/
        └── body_25/             # Model pose detection
```

## 🚀 Cài đặt và chạy

### Yêu cầu hệ thống

- Python 3.8 hoặc mới hơn
- Camera RGB (USB/built-in)
- Windows 10/11, macOS, hoặc Linux

### Bước 1: Clone repository

```bash
git clone <repository-url>
cd GaitSenseAI
```

### Bước 2: Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### Bước 3: Chạy ứng dụng

```bash
python main.py
```

## 🎮 Hướng dẫn sử dụng

### Kết nối Camera

1. Kết nối camera RGB với máy tính
2. Mở ứng dụng GaitSenseAI
3. Click menu **Camera > Connect Camera** hoặc click **"Bắt đầu phân tích"**

### Phân tích dáng đi

1. Đứng trong tầm nhìn camera ở khoảng cách 2-3 mét
2. Click **"Bắt đầu phân tích"** trong sidebar
3. Đi bộ tự nhiên trong khung hình
4. Quan sát các thông số được cập nhật real-time trong sidebar:
   - **Số bước**: Tổng số bước đã đi
   - **Tần số bước**: Số bước mỗi phút
   - **Độ dài bước**: Độ dài bước trung bình (pixel)
   - **Thời gian bước**: Thời gian trung bình cho một bước
   - **Tốc độ đi**: Tốc độ đi ước tính
   - **Góc chân**: Góc nghiêng của chân trái/phải

### Điều khiển

- **Bắt đầu/Dừng**: Toggle phân tích dáng đi
- **Đặt lại**: Reset tất cả thông số về 0
- **Menu Camera**: Kết nối/ngắt kết nối camera
- **Menu Analysis**: Điều khiển phân tích qua menu

### Phím tắt

- `Ctrl+S`: Bắt đầu phân tích
- `Ctrl+T`: Dừng phân tích  
- `Ctrl+R`: Đặt lại phân tích
- `Ctrl+Q`: Thoát ứng dụng

## 🔧 Cấu hình

### Camera Settings

Chỉnh sửa `config/settings.py` để thay đổi cài đặt camera:

```python
CAMERA_SETTINGS = {
    "width": 1920,      # Độ phân giải ngang
    "height": 1080,     # Độ phân giải dọc
    "fps": 60,          # Khung hình mỗi giây
    "device_id": 0      # ID camera (0 = camera mặc định)
}
```

### UI Settings

```python
UI_SETTINGS = {
    "window_width": 1400,   # Chiều rộng cửa sổ
    "window_height": 900,   # Chiều cao cửa sổ
    "sidebar_width": 350,   # Chiều rộng sidebar
    "dark_theme": True      # Sử dụng theme tối
}
```

### Gait Analysis Settings

```python
GAIT_ANALYSIS = {
    "sampling_rate": 60,                # Tần số lấy mẫu (Hz)
    "step_detection_threshold": 0.3,    # Ngưỡng phát hiện bước
    "stride_length_estimation": True,   # Ước tính độ dài bước
    "cadence_calculation": True         # Tính toán cadence
}
```

## 📊 Xuất dữ liệu

Dữ liệu phân tích được lưu tự động trong thư mục:
- **Windows**: `%APPDATA%/GaitSenseAI/`
- **macOS**: `~/Library/Application Support/GaitSenseAI/`
- **Linux**: `~/.local/share/GaitSenseAI/`

Các định dạng hỗ trợ:
- JSON (raw data)
- CSV (metrics timeline)
- Excel (formatted report)

## 🛠️ Phát triển

### Thêm metric mới

1. Mở `src/core/gait_analyzer.py`
2. Thêm field mới vào `GaitMetrics` dataclass
3. Implement logic tính toán trong `GaitAnalyzer`
4. Cập nhật `get_analysis_summary()` method
5. Thêm metric card trong `src/ui/sidebar.py`

### Custom theme

1. Mở `assets/styles/dark_theme.py`
2. Chỉnh sửa `DARK_THEME_STYLESHEET`
3. Cập nhật `COLORS` dictionary cho color scheme mới

### Thêm camera support

1. Mở `src/core/camera_handler.py`
2. Implement camera backend mới (RealSense, etc.)
3. Cập nhật `CameraHandler` class

## 🐛 Troubleshooting

### Camera không kết nối được

```bash
# Kiểm tra camera có sẵn
python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"

# List cameras (Windows)
python -c "import cv2; [print(f'Camera {i}: {cv2.VideoCapture(i).isOpened()}') for i in range(5)]"
```

### Lỗi MediaPipe

```bash
# Reinstall MediaPipe
pip uninstall mediapipe
pip install mediapipe==0.10.7
```

### Performance issues

1. Giảm resolution camera trong settings
2. Giảm FPS analysis rate
3. Disable pose landmark drawing

## 📝 Ghi chú kỹ thuật

### Pose Detection

- Sử dụng MediaPipe Pose solution
- 33 landmarks cho full body pose
- Confidence threshold có thể điều chỉnh

### Gait Analysis Algorithm

1. **Step Detection**: Phát hiện bước dựa trên chuyển động ankle
2. **Cadence Calculation**: Tần số bước từ peak detection
3. **Stride Length**: Ước tính từ hip displacement
4. **Foot Angle**: Tính từ heel-toe vector

### Performance Optimization

- Multi-threading cho camera capture
- Efficient frame processing
- Optimized UI updates

## 🤝 Đóng góp

1. Fork repository
2. Tạo feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Mở Pull Request

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

## 📞 Liên hệ

- Email: support@gaitsenseai.com
- GitHub: [GaitSenseAI](https://github.com/username/GaitSenseAI)

## 🙏 Acknowledgments

- [MediaPipe](https://mediapipe.dev/) for pose detection
- [OpenCV](https://opencv.org/) for computer vision
- [PyQt6](https://riverbankcomputing.com/software/pyqt/) for GUI framework
- [Intel RealSense](https://www.intelrealsense.com/) for UI inspiration

