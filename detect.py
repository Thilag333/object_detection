import cv2
import torch
import argparse
import numpy as np
from pathlib import Path
from typing import List, Tuple, Union, Dict
import time
from collections import defaultdict
from dataclasses import dataclass

from utils.helpers import (
    draw_boxes, calculate_fps, display_fps, display_stats,
    preprocess_frame, load_model, get_class_colors
)
from utils.tracker import ObjectTracker
from utils.logger import DetectionLogger
from utils.visualization import VisualizationUtils, ROI
from config import (
    MODEL_CONFIG, DETECTION_CONFIG, VISUALIZATION_CONFIG,
    OUTPUT_CONFIG, CAMERA_CONFIG, TRACKING_CONFIG, ROI_CONFIG
)

def parse_args():
    parser = argparse.ArgumentParser(description='YOLO Object Detection')
    parser.add_argument('--source', type=str, default='0', 
                       help='Source for detection (0 for webcam, or path to image/video)')
    parser.add_argument('--model-size', type=str, default=MODEL_CONFIG['model_size'],
                       choices=['n', 's', 'm', 'l', 'x'],
                       help='YOLO model size')
    parser.add_argument('--device', type=str, default=MODEL_CONFIG['device'],
                       choices=['cpu', 'cuda'],
                       help='Device to run model on')
    parser.add_argument('--conf', type=float, default=DETECTION_CONFIG['conf_threshold'],
                       help='Confidence threshold')
    parser.add_argument('--classes', nargs='+', type=str, 
                       default=DETECTION_CONFIG['classes'],
                       help='Filter by class names')
    parser.add_argument('--save', action='store_true', 
                       default=OUTPUT_CONFIG['save_video'],
                       help='Save output to file')
    parser.add_argument('--track', action='store_true',
                       help='Enable object tracking')
    parser.add_argument('--log', action='store_true',
                       help='Enable detection logging')
    parser.add_argument('--roi', action='store_true',
                       default=ROI_CONFIG['enabled'],
                       help='Enable region of interest')
    parser.add_argument('--heatmap', action='store_true',
                       default=VISUALIZATION_CONFIG['show_heatmap'],
                       help='Show detection heatmap')
    parser.add_argument('--speed', action='store_true',
                       default=VISUALIZATION_CONFIG['show_speed'],
                       help='Show object speed')
    parser.add_argument('--snapshots', action='store_true',
                       default=OUTPUT_CONFIG['save_snapshots'],
                       help='Save detection snapshots')
    return parser.parse_args()

def process_detections(results, conf_threshold: float = 0.5, 
                     class_filter: List[str] = None) -> Tuple[List, List, List, List]:
    """
    Process detection results and filter based on confidence and classes.
    
    Args:
        results: YOLO detection results
        conf_threshold: Confidence threshold
        class_filter: List of classes to filter
    
    Returns:
        Tuple of (boxes, class_names, confidences, colors)
    """
    boxes = []
    class_names = []
    confidences = []
    
    for *box, conf, cls in results.xyxy[0]:
        if conf < conf_threshold:
            continue
            
        class_name = results.names[int(cls)]
        if class_filter and class_name not in class_filter:
            continue
            
        boxes.append(box)
        class_names.append(class_name)
        confidences.append(float(conf))
    
    # Get colors for each class
    colors = get_class_colors(class_names, VISUALIZATION_CONFIG['random_colors'])
    
    return boxes, class_names, confidences, colors

def get_detection_stats(class_names: List[str]) -> Dict[str, int]:
    """
    Calculate detection statistics.
    
    Args:
        class_names: List of detected class names
    
    Returns:
        Dictionary of detection statistics
    """
    stats = defaultdict(int)
    for name in class_names:
        stats[name] += 1
    return dict(stats)

def main():
    args = parse_args()
    
    # Initialize video capture
    source = 0 if args.source == '0' else args.source
    cap = cv2.VideoCapture(source)
    
    # Set camera properties if using webcam
    if source == 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_CONFIG['width'])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_CONFIG['height'])
        cap.set(cv2.CAP_PROP_FPS, CAMERA_CONFIG['fps'])
    
    if not cap.isOpened():
        print(f"Error: Could not open source {source}")
        return
    
    # Load YOLO model
    print("Loading YOLO model...")
    try:
        model = load_model(args.model_size, args.device, MODEL_CONFIG['pretrained'])
        print(f"Model loaded successfully! (Size: {args.model_size}, Device: {args.device})")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
    
    # Initialize tracker if enabled
    tracker = ObjectTracker(
        max_disappeared=TRACKING_CONFIG['max_disappeared'],
        max_distance=TRACKING_CONFIG['max_distance']
    ) if args.track else None
    
    # Initialize logger if enabled
    logger = DetectionLogger() if args.log else None
    
    # Initialize ROIs if enabled
    rois = [ROI(**region) for region in ROI_CONFIG['regions']] if args.roi else []
    
    # Initialize variables
    frame_count = 0
    start_time = time.time()
    last_snapshot_time = time.time()
    
    # Create output writer if save is enabled
    if args.save:
        output_path = Path(OUTPUT_CONFIG['output_path'])
        fourcc = cv2.VideoWriter_fourcc(*OUTPUT_CONFIG['output_format'])
        out = cv2.VideoWriter(str(output_path), fourcc, OUTPUT_CONFIG['output_fps'],
                            (int(cap.get(3)), int(cap.get(4))))
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            results = model(frame)
            
            # Process detections
            boxes, class_names, confidences, colors = process_detections(
                results, args.conf, args.classes
            )
            
            # Update tracker if enabled
            if tracker:
                tracked_objects = tracker.update(boxes)
                # Draw motion trails
                if TRACKING_CONFIG['show_trails']:
                    frame = tracker.draw_trails(frame, TRACKING_CONFIG['trail_color'])
            
            # Draw ROIs if enabled
            if args.roi:
                for roi in rois:
                    frame = VisualizationUtils.draw_roi(frame, roi)
            
            # Create heatmap if enabled
            if args.heatmap:
                frame = VisualizationUtils.create_heatmap(
                    frame, boxes, VISUALIZATION_CONFIG['heatmap_decay']
                )
            
            # Draw boxes and labels
            frame = draw_boxes(
                frame, boxes, class_names, confidences, colors,
                VISUALIZATION_CONFIG['line_thickness'],
                VISUALIZATION_CONFIG['font_scale']
            )
            
            # Add object IDs and speed if tracking is enabled
            if tracker and args.speed:
                for object_id, centroid in tracked_objects.items():
                    # Draw object ID
                    cv2.putText(frame, f"ID: {object_id}", 
                              (int(centroid[0]), int(centroid[1])),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Calculate and draw speed if we have previous position
                    if object_id in tracker.trails and len(tracker.trails[object_id]) > 1:
                        prev_centroid = tracker.trails[object_id][-2]
                        speed = VisualizationUtils.calculate_speed(
                            prev_centroid, centroid,
                            CAMERA_CONFIG['fps'],
                            VISUALIZATION_CONFIG['pixels_per_meter']
                        )
                        frame = VisualizationUtils.draw_speed(frame, centroid, speed)
            
            # Calculate and display FPS
            frame_count += 1
            fps, _ = calculate_fps(start_time, frame_count)
            
            if VISUALIZATION_CONFIG['show_fps']:
                frame = display_fps(frame, fps)
            
            if VISUALIZATION_CONFIG['show_stats']:
                stats = get_detection_stats(class_names)
                frame = display_stats(frame, stats)
            
            # Log detections if enabled
            if logger:
                # Log individual detections
                for box, class_name, conf in zip(boxes, class_names, confidences):
                    logger.log_detection(frame_count, -1, class_name, conf, box)
                
                # Log frame statistics
                logger.log_frame_stats(frame_count, stats)
            
            # Save snapshot if enabled and interval has passed
            if args.snapshots and (time.time() - last_snapshot_time) >= OUTPUT_CONFIG['snapshot_interval']:
                snapshot_path = VisualizationUtils.save_snapshot(
                    frame, boxes, class_names, confidences,
                    OUTPUT_CONFIG['snapshot_dir']
                )
                if snapshot_path:
                    print(f"Saved snapshot: {snapshot_path}")
                last_snapshot_time = time.time()
            
            # Display frame
            cv2.imshow('YOLO Object Detection', frame)
            
            # Save frame if enabled
            if args.save:
                out.write(frame)
            
            # Break loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nDetection stopped by user")
    finally:
        cap.release()
        if args.save:
            out.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main() 