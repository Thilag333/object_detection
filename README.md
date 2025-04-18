# Real-Time Object Detection using YOLO

This project implements real-time object detection using the YOLO (You Only Look Once) algorithm. It can detect objects in both images and video streams with high accuracy and speed.

## Features
- Real-time object detection from webcam feed
- Support for image and video file processing
- Customizable confidence threshold
- Object class filtering
- FPS counter display
- Bounding box visualization with class labels and confidence scores

## Installation

1. Clone this repository
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### For Webcam Detection
```bash
python detect.py --source 0
```

### For Image Detection
```bash
python detect.py --source path/to/image.jpg
```

### For Video Detection
```bash
python detect.py --source path/to/video.mp4
```

### Additional Options
- `--conf`: Confidence threshold (default: 0.5)
- `--classes`: Filter by class names (e.g., "person car")
- `--save`: Save output to file

## Project Structure
```
├── detect.py           # Main detection script
├── utils/             # Utility functions
│   ├── __init__.py
│   └── helpers.py     # Helper functions
├── models/            # Model storage
├── requirements.txt   # Project dependencies
└── README.md         # Project documentation
``` 