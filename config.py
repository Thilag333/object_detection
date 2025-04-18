"""
Configuration settings for the YOLO object detection project.
"""

# Model settings
MODEL_CONFIG = {
    'model_size': 's',  # Options: 'n' (nano), 's' (small), 'm' (medium), 'l' (large), 'x' (xlarge)
    'device': 'cpu',    # Options: 'cpu', 'cuda' (if GPU available)
    'pretrained': True
}

# Detection settings
DETECTION_CONFIG = {
    'conf_threshold': 0.5,
    'iou_threshold': 0.45,
    'max_detections': 1000,
    'classes': None,    # List of class names to filter, None for all classes
}

# Visualization settings
VISUALIZATION_CONFIG = {
    'line_thickness': 2,
    'font_scale': 0.5,
    'font_thickness': 1,
    'show_fps': True,
    'show_stats': True,
    'random_colors': True,
    'show_heatmap': False,
    'heatmap_decay': 0.95,
    'show_speed': True,
    'pixels_per_meter': 100,  # Conversion factor for speed calculation
}

# Output settings
OUTPUT_CONFIG = {
    'save_video': False,
    'output_format': 'mp4v',
    'output_fps': 20.0,
    'output_path': 'output.mp4',
    'save_snapshots': False,
    'snapshot_dir': 'snapshots',
    'snapshot_interval': 5,  # Save snapshot every N seconds
}

# Camera settings
CAMERA_CONFIG = {
    'width': 640,
    'height': 480,
    'fps': 30
}

# Tracking settings
TRACKING_CONFIG = {
    'max_disappeared': 30,
    'max_distance': 50,
    'show_trails': True,
    'trail_length': 30,
    'trail_color': (0, 255, 0),
}

# Region of Interest settings
ROI_CONFIG = {
    'enabled': False,
    'regions': [
        {
            'name': 'ROI 1',
            'x1': 100,
            'y1': 100,
            'x2': 300,
            'y2': 300,
            'color': (0, 255, 0),
        }
    ]
}

# Database settings
DATABASE_CONFIG = {
    'db_path': 'detections.db',  # Path to SQLite database file
    'save_detections': True,      # Whether to save individual detections
    'save_events': True,          # Whether to save detected events
    'save_statistics': True,      # Whether to save frame statistics
    'report_interval': 3600,      # Interval in seconds to generate reports
    'report_dir': 'reports',      # Directory to save reports
} 