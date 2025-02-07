# Computer Vision Projects Collection 🎥

A collection of advanced computer vision projects implementing real-time object detection, pose estimation, and face recognition using YOLOv8, MediaPipe, and DeepFace.

## 🚀 Projects Overview

### 1. Combined Detector (Multiple Implementations)
- **File**: `combined_detector.py` & `combined_multi.py`
- **Features**:
  - Real-time object detection using YOLOv8
  - Multi-person pose estimation
  - Simultaneous object and pose detection
  - FPS monitoring
  - Color-coded visualization for multiple people

### 2. Face Matching System
- **File**: `face_match.py`
- **Features**:
  - Real-time face matching using DeepFace
  - Threading implementation for smooth performance
  - Visual feedback for match/no-match scenarios
  - Configurable reference image comparison

### 3. Multiple Pose Detection
- **File**: `multiple_pose.py`
- **Features**:
  - Support for up to 10 simultaneous pose detections
  - Integration of YOLO and MediaPipe
  - Color-coded skeleton visualization
  - High-resolution support (1920x1080)
  - Real-time FPS counter

### 4. Object Detection System
- **File**: `object_detection.py`
- **Features**:
  - YOLOv8 implementation for object detection
  - Support for 80 COCO dataset classes
  - Confidence threshold customization
  - Colored bounding boxes and labels
  - Performance monitoring

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/Amityadav9/computer-vision-toolkit.git
cd computer-vision-toolkit
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

Required packages:
- opencv-python
- mediapipe
- ultralytics
- numpy
- deepface
- torch

## 📋 Usage

### Combined Detector
```bash
python combined_detector.py  # MediaPipe version
# or
python combined_multi.py     # YOLO-only version
```

### Face Matching
```bash
python face_match.py
# Note: Replace "Amit_0.jpg" with your reference image
```

### Multiple Pose Detection
```bash
python multiple_pose.py
```

### Object Detection
```bash
python object_detection.py
```

## ⚙️ Configuration

### Camera Settings
All projects support customizable camera resolution:
- Default resolution: 1280x720 (most projects)
- High-res option: 1920x1080 (multiple_pose.py)
- Adjustable FPS settings

### Detection Parameters
- Confidence thresholds can be adjusted (default: 0.5)
- Number of simultaneous pose detections (up to 10)
- Customizable visualization colors

## 🎯 Features in Detail

### Combined Detection
- Simultaneous object and pose detection
- Color-coded visualization for different people
- Real-time performance metrics
- Two implementation options:
  1. MediaPipe + YOLO
  2. Pure YOLO approach

### Face Matching
- Real-time face comparison
- Threading for improved performance
- Visual match indicators
- Customizable reference image

### Pose Detection
- Multi-person tracking
- Skeleton visualization
- Bounding box detection
- Person counting
- FPS monitoring

### Object Detection
- 80 COCO classes support
- Confidence scoring
- Custom color schemes
- Performance optimization

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔍 Future Improvements

- [ ] Add support for video file processing
- [ ] Implement pose tracking persistence
- [ ] Add more face recognition features
- [ ] Optimize performance for lower-end hardware
- [ ] Add support for custom object detection models
- [ ] Implement data logging and analytics

## ⚠️ Requirements

- Python 3.8+
- Webcam or video input device
- CUDA-capable GPU (recommended for optimal performance)
- Minimum 4GB RAM (8GB+ recommended)
