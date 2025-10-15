import sys
print(sys.executable)  # Prints the Python interpreter path being used
import asyncio
import logging
import math
import json
import cv2  # Import OpenCV for camera capture
import numpy as np  # Import numpy for array manipulation
from aiortc import RTCIceCandidate, RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from aiortc.mediastreams import VideoStreamTrack  # Import VideoStreamTrack
from av import VideoFrame  # Import VideoFrame for aiortc video processing
import websockets  # Import the websockets library
import re  # Import regex for parsing candidate string
from ultralytics import YOLO
import time
import board
import busio
import adafruit_mlx90640
import os
from datetime import datetime
from typing import List, Dict, Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DualCameraCalibrator:
    def __init__(self, rgb_resolution=(320, 240), thermal_resolution=(32, 24)):
        self.rgb_resolution = rgb_resolution
        self.thermal_resolution = thermal_resolution
        self.homography_matrix = None
        self.calibration_file = "camera_calibration.json"
        
        # Load existing calibration if available
        self.load_calibration()
        
    def load_calibration(self):
        """Load existing calibration from file"""
        try:
            if os.path.exists(self.calibration_file):
                with open(self.calibration_file, 'r') as f:
                    data = json.load(f)
                
                self.homography_matrix = np.array(data['homography_matrix'])
                logger.info(f"Loaded calibration from {self.calibration_file}")
                logger.info(f"Calibration date: {data.get('calibration_date', 'Unknown')}")
                logger.info(f"Reprojection error: {data.get('reprojection_error', 'Unknown'):.2f} pixels")
                return True
        except Exception as e:
            logger.error(f"Failed to load calibration: {e}")
        
        return False
    
    def save_calibration(self, rgb_points, thermal_points):
        """Calculate and save homography matrix"""
        try:
            src_points = np.float32(rgb_points)
            dst_points = np.float32(thermal_points)
            
            self.homography_matrix, mask = cv2.findHomography(
                src_points, dst_points, 
                cv2.RANSAC, 
                ransacReprojThreshold=2.0
            )
            
            if self.homography_matrix is not None:
                error = self._calculate_reprojection_error(rgb_points, thermal_points)
                
                calibration_data = {
                    'homography_matrix': self.homography_matrix.tolist(),
                    'rgb_points': rgb_points,
                    'thermal_points': thermal_points,
                    'reprojection_error': error,
                    'rgb_resolution': self.rgb_resolution,
                    'thermal_resolution': self.thermal_resolution,
                    'calibration_date': datetime.now().isoformat(),
                    'inlier_mask': mask.tolist() if mask is not None else None
                }
                
                with open(self.calibration_file, 'w') as f:
                    json.dump(calibration_data, f, indent=2)
                
                logger.info(f"Calibration saved! Reprojection error: {error:.2f} pixels")
                return True
            
        except Exception as e:
            logger.error(f"Error saving calibration: {e}")
        
        return False
    
    def _calculate_reprojection_error(self, rgb_points, thermal_points):
        """Calculate average reprojection error"""
        if self.homography_matrix is None:
            return float('inf')
        
        rgb_pts = np.float32(rgb_points).reshape(-1, 1, 2)
        projected_pts = cv2.perspectiveTransform(rgb_pts, self.homography_matrix)
        projected_pts = projected_pts.reshape(-1, 2)
        
        errors = []
        for (tx, ty), (px, py) in zip(thermal_points, projected_pts):
            error = np.sqrt((tx - px)**2 + (ty - py)**2)
            errors.append(error)
        
        return np.mean(errors)
    
    def map_rgb_bbox_to_thermal(self, rgb_bbox):
        """Map RGB bounding box to thermal coordinates using homography"""
        if self.homography_matrix is None:
            # Fallback to simple scaling
            x1, y1, x2, y2 = rgb_bbox
            thermal_x1 = int(x1 * 32 / 320)
            thermal_y1 = int(y1 * 24 / 240)
            thermal_x2 = int(x2 * 32 / 320)
            thermal_y2 = int(y2 * 24 / 240)
            
            thermal_x1 = max(0, min(31, thermal_x1))
            thermal_y1 = max(0, min(23, thermal_y1))
            thermal_x2 = max(0, min(31, thermal_x2))
            thermal_y2 = max(0, min(23, thermal_y2))
            
            if thermal_x2 > thermal_x1 and thermal_y2 > thermal_y1:
                return (thermal_x1, thermal_y1, thermal_x2, thermal_y2)
            return None
        
        x1, y1, x2, y2 = rgb_bbox
        
        # Map all four corners using homography
        corners = np.float32([
            [x1, y1],  # Top-left
            [x2, y1],  # Top-right
            [x2, y2],  # Bottom-right
            [x1, y2]   # Bottom-left
        ]).reshape(-1, 1, 2)
        
        mapped_corners = cv2.perspectiveTransform(corners, self.homography_matrix)
        mapped_corners = mapped_corners.reshape(-1, 2)
        
        # Find bounding box of mapped corners
        min_x = max(0, min(mapped_corners[:, 0]))
        min_y = max(0, min(mapped_corners[:, 1]))
        max_x = min(31, max(mapped_corners[:, 0]))
        max_y = min(23, max(mapped_corners[:, 1]))
        
        if max_x > min_x and max_y > min_y:
            return (int(min_x), int(min_y), int(max_x), int(max_y))
        
        return None

class CalibratedHumanDetector:
    def __init__(self, calibrator):
        self.calibrator = calibrator
        
        # Temperature thresholds (Celsius)
        self.TEMP_THRESHOLDS = {
            'human_min': 33.0,        # Minimum human skin temperature
            'human_normal': 36.0,     # Normal human skin temperature  
            'human_fever': 38.0,      # Fever threshold
            'ambient_diff_min': 4.0,  # Minimum difference from ambient
            'dead_threshold': 30.0,   # Below this = likely dead
        }
        
    def analyze_human_regions(self, yolo_results, thermal_data, ambient_temp=None):
        """Main function to analyze human detections using thermal data"""
        if ambient_temp is None:
            ambient_temp = self._estimate_ambient_temperature(thermal_data)
        
        human_analyses = []
        
        for box in yolo_results.boxes.data:
            x1, y1, x2, y2, conf, cls = box
            class_id = int(cls)
            
            if class_id == 0:  # Person class
                rgb_bbox = (int(x1), int(y1), int(x2), int(y2))
                
                # Map RGB detection to thermal region
                analysis = self._analyze_single_human(rgb_bbox, thermal_data, ambient_temp)
                if analysis:
                    analysis['rgb_bbox'] = rgb_bbox
                    analysis['yolo_confidence'] = float(conf)
                    human_analyses.append(analysis)
        
        return human_analyses
    
    def _analyze_single_human(self, rgb_bbox, thermal_data, ambient_temp):
        """Analyze single human detection"""
        # Map RGB bbox to thermal coordinates
        thermal_bbox = self.calibrator.map_rgb_bbox_to_thermal(rgb_bbox)
        
        if thermal_bbox is None:
            return None
        
        tx1, ty1, tx2, ty2 = thermal_bbox
        
        # Extract thermal region
        thermal_region = thermal_data[ty1:ty2, tx1:tx2]
        
        if thermal_region.size == 0:
            return None
        
        # Thermal analysis
        max_temp = np.max(thermal_region)
        mean_temp = np.mean(thermal_region)
        std_temp = np.std(thermal_region)
        
        # Hot pixel analysis
        hot_pixels = np.sum(thermal_region >= self.TEMP_THRESHOLDS['human_min'])
        total_pixels = thermal_region.size
        hot_pixel_ratio = hot_pixels / total_pixels if total_pixels > 0 else 0
        
        temp_above_ambient = max_temp - ambient_temp
        
        # Determine status and confidence
        status, confidence = self._determine_human_status(
            max_temp, temp_above_ambient, hot_pixel_ratio
        )
        
        return {
            'status': status,
            'confidence': confidence,
            'max_temp': round(max_temp, 1),
            'mean_temp': round(mean_temp, 1),
            'temp_above_ambient': round(temp_above_ambient, 1),
            'hot_pixel_ratio': round(hot_pixel_ratio, 2),
            'thermal_bbox': thermal_bbox,
            'ambient_temp': round(ambient_temp, 1)
        }
    
    def _determine_human_status(self, max_temp, temp_above_ambient, hot_pixel_ratio):
        """Determine human status based on thermal analysis"""
        confidence = 0.0
        
        # Temperature-based confidence
        if max_temp >= self.TEMP_THRESHOLDS['human_normal']:
            temp_conf = 1.0
        elif max_temp >= self.TEMP_THRESHOLDS['human_min']:
            temp_conf = 0.7
        elif max_temp >= 30.0:
            temp_conf = 0.3
        else:
            temp_conf = 0.0
        
        # Ambient difference confidence
        if temp_above_ambient >= self.TEMP_THRESHOLDS['ambient_diff_min']:
            ambient_conf = 1.0
        elif temp_above_ambient >= 2.0:
            ambient_conf = 0.5
        else:
            ambient_conf = 0.0
        
        # Hot pixel confidence
        pixel_conf = min(1.0, hot_pixel_ratio * 5)  # Scale up to make it meaningful
        
        # Overall confidence (weighted average)
        confidence = (temp_conf * 0.5 + ambient_conf * 0.3 + pixel_conf * 0.2)
        
        # Determine status
        if confidence >= 0.8:
            if max_temp >= self.TEMP_THRESHOLDS['human_fever']:
                return "ALIVE - Fever", confidence
            elif max_temp >= self.TEMP_THRESHOLDS['human_normal']:
                return "ALIVE - Normal", confidence
            else:
                return "ALIVE - Cool", confidence
        
        elif confidence >= 0.6:
            return "LIKELY ALIVE", confidence
        
        elif confidence >= 0.4:
            return "UNCERTAIN", confidence
        
        else:
            if max_temp <= self.TEMP_THRESHOLDS['dead_threshold']:
                return "LIKELY DEAD", confidence
            else:
                return "UNKNOWN", confidence
    
    def _estimate_ambient_temperature(self, thermal_data):
        """Estimate ambient temperature from thermal image"""
        return np.percentile(thermal_data, 25)

class ThermalVideoStreamTrack(VideoStreamTrack):
    def __init__(self):
        super().__init__()
        self.width = 320
        self.height = 240
        self._initialize_thermal_camera()
        
        # Frame rate control
        self.last_frame_time = 0
        self.frame_interval = 1.0 / 8  # 8 FPS for thermal camera
        
        logger.info("ThermalVideoStreamTrack initialized.")

    def _initialize_thermal_camera(self):
        """Initialize MLX90640 thermal camera"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Initialize I2C bus
                self.i2c = busio.I2C(board.SCL, board.SDA, frequency=400000)
                
                # Initialize MLX90640 camera
                self.mlx = adafruit_mlx90640.MLX90640(self.i2c)
                logger.info(f"MLX90640 detected, serial number: {[hex(i) for i in self.mlx.serial_number]}")
                
                # Set refresh rate
                self.mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_8_HZ
                
                # Create a frame buffer to store thermal data
                self.thermal_frame = np.zeros((24 * 32,), dtype=float)
                
                logger.info(f"Thermal camera initialized successfully on attempt {attempt + 1}")
                return
            except Exception as e:
                logger.error(f"Thermal camera initialization error on attempt {attempt + 1}: {e}")
                time.sleep(0.5)
        
        raise IOError("Cannot initialize thermal camera after multiple attempts. Please ensure MLX90640 is connected.")

    def thermal_to_colormap(self, thermal_data):
        """Convert thermal data to displayable color image with fixed temperature range"""
        # Fixed temperature range optimized for indoor + human detection
        vmin = 20.0   # Room temperature baseline
        vmax = 50.0   # Human body temp + some margin (37°C + buffer)
        
        # Scale to 0-1 range using fixed range
        scaled = np.clip((thermal_data - vmin) / (vmax - vmin), 0, 1)
        
        # Convert to 8-bit
        thermal_8bit = (scaled * 255).astype(np.uint8)
        
        # Resize to desired output size
        thermal_resized = cv2.resize(thermal_8bit, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        
        # Apply colormap
        thermal_color = cv2.applyColorMap(thermal_resized, cv2.COLORMAP_INFERNO)
        
        return thermal_color

    def get_thermal_data(self):
        """Get raw thermal data for analysis"""
        try:
            self.mlx.getFrame(self.thermal_frame)
            thermal_data = np.array(self.thermal_frame).reshape((24, 32))
            return thermal_data
        except Exception as e:
            logger.error(f"Error getting thermal data: {e}")
            return None

    async def recv(self):
        """Receive and process thermal video frames"""
        try:
            # Frame rate control
            current_time = time.time()
            if current_time - self.last_frame_time < self.frame_interval:
                await asyncio.sleep(self.frame_interval - (current_time - self.last_frame_time))
            
            self.last_frame_time = time.time()
            
            pts, time_base = await self.next_timestamp()

            # Read thermal frame
            try:
                thermal_data = self.get_thermal_data()
                if thermal_data is not None:
                    frame = self.thermal_to_colormap(thermal_data)
                    
                    # Add temperature info overlay
                    avg_temp = np.mean(thermal_data)
                    min_temp = np.min(thermal_data)
                    max_temp = np.max(thermal_data)
                    
                    # Add "THERMAL" identifier in top-left corner
                    cv2.putText(frame, "THERMAL", (10, 25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    cv2.putText(frame, f"Avg: {avg_temp:.1f}C", (10, 55), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(frame, f"Min: {min_temp:.1f}C", (10, 80), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    cv2.putText(frame, f"Max: {max_temp:.1f}C", (10, 105), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    
                    # Add thermal-specific visual marker (orange border)
                    cv2.rectangle(frame, (0, 0), (frame.shape[1]-1, frame.shape[0]-1), (0, 165, 255), 3)
                else:
                    raise Exception("No thermal data")
                
            except Exception as e:
                logger.error(f"Error reading thermal frame: {e}")
                # Return a black frame with error indicator
                frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                cv2.rectangle(frame, (50, 50), (100, 100), (0, 0, 255), -1)
                cv2.putText(frame, "Thermal Error", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Convert BGR to RGB for WebRTC
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_frame = VideoFrame.from_ndarray(frame, format="rgb24")
            video_frame.pts = pts
            video_frame.time_base = time_base

            return video_frame
            
        except Exception as e:
            logger.error(f"Error in thermal recv(): {e}")
            # Return a black frame with marker in case of error
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            cv2.rectangle(frame, (50, 50), (100, 100), (0, 0, 255), -1)
            video_frame = VideoFrame.from_ndarray(frame, format="rgb24")
            video_frame.pts = pts
            video_frame.time_base = time_base
            return video_frame

class CameraVideoStreamTrack(VideoStreamTrack):
    def __init__(self, thermal_track=None):
        super().__init__()
        self.cap = None
        self.width = 320
        self.height = 240
        self.thermal_track = thermal_track  # Reference to thermal camera
        self._initialize_camera()
        
        # Initialize calibrator and human detector
        self.calibrator = DualCameraCalibrator()
        self.human_detector = CalibratedHumanDetector(self.calibrator)
        
        # Load YOLOv8 model
        try:
            self.model = YOLO("yolov8n.pt")
            logger.info("YOLO model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise
            
        self.target_class_ids = {0: "person", 56: "chair", 60: "dining table"}
        
        # Frame rate control
        self.last_frame_time = 0
        self.frame_interval = 1.0 / 15  # 15 FPS
        
        # Detection optimization
        self.frame_counter = 0
        self.last_results = None
        self.last_human_analyses = []
        
        logger.info("CameraVideoStreamTrack with calibrated thermal mapping initialized.")

    def _initialize_camera(self):
        """Initialize camera with optimized GStreamer pipeline for Raspberry Pi Camera v2"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if self.cap:
                    self.cap.release()
                    time.sleep(0.5)
                    logger.info("Previous camera instance released")
                    
                # Optimized GStreamer pipeline for low latency
                pipeline = (
                    f"libcamerasrc ! "
                    f"video/x-raw, width={self.width}, height={self.height}, format=YUY2, framerate=15/1 ! "
                    f"queue max-size-buffers=1 leaky=downstream ! "
                    f"videoconvert ! video/x-raw, format=BGR ! "
                    f"appsink sync=false drop=true max-buffers=1"
                )
                self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
                if self.cap.isOpened():
                    logger.info(f"Camera initialized successfully on attempt {attempt + 1}")
                    return
                else:
                    logger.warning(f"Failed to open camera pipeline on attempt {attempt + 1}")
                    time.sleep(0.5)
            except Exception as e:
                logger.error(f"Camera initialization error on attempt {attempt + 1}: {e}")
                time.sleep(0.5)
        
        raise IOError("Cannot open camera pipeline after multiple attempts. Please ensure camera is connected and libcamera is configured.")

    def cleanup(self):
        """Clean up camera resources"""
        if self.cap and self.cap.isOpened():
            self.cap.release()
            time.sleep(0.5)
            logger.info("Camera released.")
        self.cap = None

    def __del__(self):
        self.cleanup()
        
    async def recv(self):
        """Receive and process video frames"""
        try:
            # Frame rate control
            current_time = time.time()
            delta = current_time - self.last_frame_time
            if delta < self.frame_interval:
                await asyncio.sleep(self.frame_interval - delta)
            
            self.last_frame_time = current_time
            
            pts, time_base = await self.next_timestamp()
            timestamp = time.time()

            # Check if camera is still open
            if not self.cap or not self.cap.isOpened():
                logger.warning("Camera not available, reinitializing...")
                self._initialize_camera()

            ret, frame = self.cap.read()
            if not ret:
                logger.error("Failed to read frame from camera, returning black frame with marker.")
                frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                cv2.rectangle(frame, (50, 50), (100, 100), (0, 0, 255), -1)
            else:
                frame = self._process_frame(frame, timestamp)

            # Convert BGR to RGB for WebRTC
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_frame = VideoFrame.from_ndarray(frame, format="rgb24")
            video_frame.pts = pts
            video_frame.time_base = time_base

            return video_frame
            
        except Exception as e:
            logger.error(f"Error in recv(): {e}")
            # Return a black frame with marker in case of error
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            cv2.rectangle(frame, (50, 50), (100, 100), (0, 0, 255), -1)
            video_frame = VideoFrame.from_ndarray(frame, format="rgb24")
            video_frame.pts = pts
            video_frame.time_base = time_base
            return video_frame

    def _process_frame(self, frame, timestamp):
        """Process frame with calibrated thermal-RGB human detection"""
        # Draw processing indicator
        cv2.rectangle(frame, (50, 50), (100, 100), (0, 255, 0), 2)

        self.frame_counter += 1
        human_analyses = self.last_human_analyses

        # Run YOLO detection every 3 frames for performance
        if self.frame_counter % 3 == 0:
            results = self.model(frame, verbose=False, imgsz=320, conf=0.5)[0]
            self.last_results = results
            
            # Get thermal data for analysis
            thermal_data = None
            if self.thermal_track:
                thermal_data = self.thermal_track.get_thermal_data()
            
            if thermal_data is not None:
                # Perform calibrated thermal-RGB analysis
                human_analyses = self.human_detector.analyze_human_regions(
                    results, thermal_data
                )
                self.last_human_analyses = human_analyses
            else:
                # Fallback to basic YOLO detection without thermal
                human_analyses = self._basic_yolo_analysis(results)
                self.last_human_analyses = human_analyses
        
        # Draw results on frame
        frame = self._draw_results(frame, human_analyses)
        
        return frame
    
    def _basic_yolo_analysis(self, results):
        """Fallback analysis when thermal data is not available"""
        analyses = []
        for box in results.boxes.data:
            x1, y1, x2, y2, conf, cls = box
            class_id = int(cls)
            
            if class_id == 0:  # Person
                analyses.append({
                    'rgb_bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'status': 'DETECTED - No Thermal',
                    'confidence': float(conf),
                    'max_temp': 0.0,
                    'yolo_confidence': float(conf)
                })
        
        return analyses
    
    def _draw_results(self, frame, human_analyses):
        """Draw detection and thermal analysis results on frame"""
        alive_count = 0
        dead_count = 0
        uncertain_count = 0
        
        for analysis in human_analyses:
            x1, y1, x2, y2 = analysis['rgb_bbox']
            status = analysis['status']
            confidence = analysis['confidence']
            max_temp = analysis.get('max_temp', 0.0)
            
            # Color coding based on status
            if "ALIVE" in status:
                color = (0, 255, 0)  # Green
                alive_count += 1
            elif "DEAD" in status:
                color = (0, 0, 255)  # Red
                dead_count += 1
            elif "UNCERTAIN" in status:
                color = (0, 255, 255)  # Yellow
                uncertain_count += 1
            else:
                color = (255, 255, 255)  # White
                uncertain_count += 1
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Status and temperature text
            status_text = status
            temp_text = f"{max_temp:.1f}°C" if max_temp > 0 else "No Temp"
            conf_text = f"Conf: {confidence:.2f}"
            
            # Draw text with background for readability
            cv2.putText(frame, status_text, (x1, y1 - 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.putText(frame, temp_text, (x1, y1 - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.putText(frame, conf_text, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Draw other objects (chairs, tables)
        if self.last_results:
            for box in self.last_results.boxes.data:
                x1, y1, x2, y2, conf, cls = box
                class_id = int(cls)
                
                if class_id in self.target_class_ids and class_id != 0:
                    label = self.target_class_ids[class_id]
                    x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                    color = (0, 128, 255)  # Orange for objects
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Display summary statistics
        cv2.putText(frame, "RGB + THERMAL", (20, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Alive: {alive_count}", (20, 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Dead: {dead_count}", (20, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Uncertain: {uncertain_count}", (20, 105), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Calibration status
        calib_status = "Calibrated" if self.calibrator.homography_matrix is not None else "Not Calibrated"
        cv2.putText(frame, f"Status: {calib_status}", (20, 130), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add RGB-specific visual marker (green border)
        cv2.rectangle(frame, (0, 0), (frame.shape[1]-1, frame.shape[0]-1), (0, 255, 0), 3)
        
        return frame

def run_calibration_process(camera_track, thermal_track):
    """
    Run camera calibration process
    Call this function to calibrate your cameras before streaming
    """
    print("=== CAMERA CALIBRATION SETUP ===")
    print("This will help you calibrate RGB and thermal cameras for accurate mapping.")
    print("You'll need some hot objects (like a warm mug, heated metal, etc.)")
    
    calibrator = DualCameraCalibrator()
    
    # Check if calibration already exists
    if calibrator.homography_matrix is not None:
        print("✅ Existing calibration found!")
        print("Current calibration is loaded and ready to use.")
        response = input("Do you want to recalibrate? (y/n): ").lower()
        if response != 'y':
            return calibrator
    else:
        print("❌ No calibration found. Running calibration process...")
    
    # Manual calibration points (you can modify these)
    print("\nFor quick setup, you can enter calibration points manually:")
    print("Or type 'interactive' for guided calibration")
    
    mode = input("Choose mode (manual/interactive): ").lower()
    
    if mode == 'manual':
        success = manual_calibration_setup(calibrator)
    else:
        success = interactive_calibration_setup(calibrator, camera_track, thermal_track)
    
    if success:
        print("✅ Calibration completed successfully!")
        print("The system is now ready for accurate thermal-RGB mapping.")
    else:
        print("❌ Calibration failed. Using fallback scaling method.")
    
    return calibrator

def manual_calibration_setup(calibrator):
    """Manual calibration setup with predefined points"""
    print("\n=== MANUAL CALIBRATION ===")
    print("Enter corresponding points in RGB and thermal images.")
    print("Format: x,y (e.g., 160,120)")
    print("You need at least 4 point pairs.")
    
    rgb_points = []
    thermal_points = []
    
    try:
        while len(rgb_points) < 4:
            print(f"\nPoint {len(rgb_points) + 1}:")
            rgb_input = input(f"RGB point (x,y): ").strip()
            thermal_input = input(f"Thermal point (x,y): ").strip()
            
            # Parse inputs
            rgb_x, rgb_y = map(int, rgb_input.split(','))
            thermal_x, thermal_y = map(int, thermal_input.split(','))
            
            # Validate ranges
            if 0 <= rgb_x < 320 and 0 <= rgb_y < 240 and 0 <= thermal_x < 32 and 0 <= thermal_y < 24:
                rgb_points.append([rgb_x, rgb_y])
                thermal_points.append([thermal_x, thermal_y])
                print(f"✅ Point {len(rgb_points)} added")
            else:
                print("❌ Invalid coordinates. RGB: 0-319,0-239, Thermal: 0-31,0-23")
        
        # Save calibration
        return calibrator.save_calibration(rgb_points, thermal_points)
        
    except Exception as e:
        print(f"❌ Manual calibration failed: {e}")
        return False

def interactive_calibration_setup(calibrator, camera_track, thermal_track):
    """Interactive calibration with live camera feeds"""
    print("\n=== INTERACTIVE CALIBRATION ===")
    print("Starting interactive calibration...")
    print("Make sure both cameras are working before proceeding.")
    
    # Test camera access
    rgb_frame = get_test_frame(camera_track)
    thermal_frame = get_test_thermal_frame(thermal_track)
    
    if rgb_frame is None:
        print("❌ Cannot access RGB camera")
        return False
    
    if thermal_frame is None:
        print("❌ Cannot access thermal camera")
        return False
    
    print("✅ Both cameras accessible")
    
    # Start interactive process
    return start_interactive_calibration_process(calibrator, camera_track, thermal_track)

def get_test_frame(camera_track):
    """Get a test frame from RGB camera"""
    try:
        if hasattr(camera_track, 'cap') and camera_track.cap and camera_track.cap.isOpened():
            ret, frame = camera_track.cap.read()
            return frame if ret else None
    except:
        pass
    return None

def get_test_thermal_frame(thermal_track):
    """Get a test frame from thermal camera"""
    try:
        if hasattr(thermal_track, 'get_thermal_data'):
            thermal_data = thermal_track.get_thermal_data()
            if thermal_data is not None:
                return thermal_track.thermal_to_colormap(thermal_data)
    except:
        pass
    return None

def start_interactive_calibration_process(calibrator, camera_track, thermal_track):
    """Start the interactive calibration process"""
    print("\n📋 CALIBRATION INSTRUCTIONS:")
    print("1. Place a HOT object (warm mug, heated metal, etc.) at different positions")
    print("2. Click the SAME hot object in both RGB and thermal windows")
    print("3. Press 'c' to capture the point pair")
    print("4. Move object to new position and repeat")
    print("5. Press 's' to save when you have 4+ points")
    print("6. Press 'q' to quit")
    
    input("Press Enter to start calibration...")
    
    # Calibration variables
    rgb_points = []
    thermal_points = []
    rgb_click = None
    thermal_click = None
    
    # Mouse callbacks
    def rgb_mouse(event, x, y, flags, param):
        nonlocal rgb_click
        if event == cv2.EVENT_LBUTTONDOWN:
            rgb_click = (x, y)
            print(f"RGB point: ({x}, {y})")
    
    def thermal_mouse(event, x, y, flags, param):
        nonlocal thermal_click
        if event == cv2.EVENT_LBUTTONDOWN:
            thermal_click = (x, y)
            print(f"Thermal point: ({x}, {y})")
    
    # Create windows
    cv2.namedWindow('RGB Calibration', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Thermal Calibration', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('RGB Calibration', rgb_mouse)
    cv2.setMouseCallback('Thermal Calibration', thermal_mouse)
    
    try:
        while True:
            # Get frames
            rgb_frame = get_test_frame(camera_track)
            thermal_frame = get_test_thermal_frame(thermal_track)
            
            if rgb_frame is None or thermal_frame is None:
                print("❌ Lost camera connection")
                break
            
            # Create display copies
            rgb_display = rgb_frame.copy()
            thermal_display = thermal_frame.copy()
            
            # Draw existing points
            for i, (rpt, tpt) in enumerate(zip(rgb_points, thermal_points)):
                cv2.circle(rgb_display, rpt, 5, (0, 255, 0), -1)
                cv2.putText(rgb_display, str(i+1), (rpt[0]+10, rpt[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.circle(thermal_display, tpt, 3, (0, 255, 0), -1)
                cv2.putText(thermal_display, str(i+1), (tpt[0]+5, tpt[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Draw current selections
            if rgb_click:
                cv2.circle(rgb_display, rgb_click, 5, (0, 0, 255), -1)
            if thermal_click:
                cv2.circle(thermal_display, thermal_click, 3, (0, 0, 255), -1)
            
            # Add status text
            cv2.putText(rgb_display, f"Points: {len(rgb_points)}/4 min", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(thermal_display, f"Points: {len(thermal_points)}/4 min", (5, 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Show frames
            cv2.imshow('RGB Calibration', rgb_display)
            cv2.imshow('Thermal Calibration', thermal_display)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('c'):  # Capture point pair
                if rgb_click and thermal_click:
                    rgb_points.append(rgb_click)
                    thermal_points.append(thermal_click)
                    print(f"✅ Point pair {len(rgb_points)} captured")
                    rgb_click = None
                    thermal_click = None
                else:
                    print("❌ Please click points in both images first")
            
            elif key == ord('r'):  # Reset
                rgb_points.clear()
                thermal_points.clear()
                rgb_click = None
                thermal_click = None
                print("🔄 Reset calibration")
            
            elif key == ord('s'):  # Save
                if len(rgb_points) >= 4:
                    success = calibrator.save_calibration(rgb_points, thermal_points)
                    if success:
                        print("✅ Calibration saved successfully!")
                        cv2.destroyAllWindows()
                        return True
                    else:
                        print("❌ Failed to save calibration")
                else:
                    print("❌ Need at least 4 point pairs")
            
            elif key == ord('q'):  # Quit
                break
        
    except Exception as e:
        print(f"❌ Calibration error: {e}")
    
    cv2.destroyAllWindows()
    return False

async def run(offer, pc, camera_track, thermal_track):
    @pc.on("track")
    def on_track(track):
        logger.info(f"Received track: {track.kind}")

    @pc.on("iceconnectionstatechange")
    async def on_ice_connection_state_change():
        logger.info(f"ICE connection state: {pc.iceConnectionState}")
        if pc.iceConnectionState in ["failed", "disconnected"]:
            logger.warning(f"ICE connection issue: {pc.iceConnectionState}")

    @pc.on("connectionstatechange")
    async def on_connection_state_change():
        logger.info(f"Connection state: {pc.connectionState}")
        if pc.connectionState == "failed":
            logger.error("WebRTC connection failed")

    try:
        # Add RGB camera track first (will be track 0)
        if camera_track:
            pc.addTrack(camera_track)
            logger.info("RGB camera track added to peer connection (Track 0)")
            
        # Add thermal camera track second (will be track 1)
        if thermal_track:
            pc.addTrack(thermal_track)
            logger.info("Thermal camera track added to peer connection (Track 1)")

        if not offer:
            logger.error("Invalid offer: Empty SDP")
            return None

        logger.info(f"Processing offer SDP: {offer}")
        if "m=video" not in offer:
            logger.error("Invalid offer SDP: Missing video media line")
            return None

        await pc.setRemoteDescription(RTCSessionDescription(sdp=offer, type="offer"))
        logger.info("Set remote description (offer)")
        
        answer = await pc.createAnswer()
        if not answer:
            logger.error("Failed to create answer")
            return None
        
        await pc.setLocalDescription(answer)
        logger.info(f"Set local description (answer)")
        logger.info(f"Answer SDP contains {answer.sdp.count('m=video')} video tracks")
        return pc.localDescription
    except Exception as e:
        logger.error(f"Error in run: {str(e)}")
        return None

def parse_ice_candidate_string(candidate_str):
    """
    Parses a raw ICE candidate string into its components required by RTCIceCandidate constructor.
    """
    if not candidate_str.startswith("candidate:"):
        raise ValueError(f"Invalid ICE candidate string format: does not start with 'candidate:': {candidate_str}")
    
    # Split the string after "candidate:"
    parts = candidate_str[len("candidate:"):].split()

    # Ensure enough core parts are present
    if len(parts) < 8:
        raise ValueError(f"Invalid ICE candidate string format: not enough core parts: {candidate_str}")

    foundation = parts[0]
    component = int(parts[1])
    protocol = parts[2]
    priority = int(parts[3])
    ip = parts[4]
    port = int(parts[5])
    
    # 'typ' is at index 6, the actual candidate type is at index 7
    candidate_type = parts[7]

    related_address = None
    related_port = None

    # Parse optional attributes like raddr and rport
    i = 8
    while i < len(parts):
        if parts[i] == "raddr" and i + 1 < len(parts):
            related_address = parts[i+1]
            i += 2
        elif parts[i] == "rport" and i + 1 < len(parts):
            related_port = int(parts[i+1])
            i += 2
        else:
            i += 1

    return {
        "foundation": foundation,
        "component": component,
        "protocol": protocol,
        "priority": priority,
        "ip": ip,
        "port": port,
        "type": candidate_type,
        "relatedAddress": related_address,
        "relatedPort": related_port
    }

async def main():
    """
    Main asynchronous function to set up and run the WebRTC streamer.
    """
    uri = "wss://websockettest-eggy.onrender.com"
    peer_id = "python-peer"
    
    # Ask user if they want to run calibration first
    print("=== THERMAL-RGB MAPPING SYSTEM ===")
    print("This system provides accurate human detection using both RGB and thermal cameras.")
    print("")
    
    calibration_choice = input("Do you want to run camera calibration? (y/n/check): ").lower()
    
    while True:  # Main reconnection loop
        websocket = None
        pc = None
        camera_track = None
        thermal_track = None
        
        try:
            # Create new peer connection for each attempt
            pc = RTCPeerConnection()
            
            # Initialize thermal camera track first (needed for calibration)
            try:
                thermal_track = ThermalVideoStreamTrack()
                logger.info("Thermal camera track initialized")
            except Exception as e:
                logger.error(f"Failed to initialize thermal camera track: {e}")
                continue
                
            # Initialize RGB camera track with thermal reference
            try:
                camera_track = CameraVideoStreamTrack(thermal_track)
                logger.info("RGB camera track initialized")
            except Exception as e:
                logger.error(f"Failed to initialize RGB camera track: {e}")
                continue

            # Handle calibration
            if calibration_choice == 'y':
                print("\n🔧 Starting calibration process...")
                run_calibration_process(camera_track, thermal_track)
            elif calibration_choice == 'check':
                calibrator = DualCameraCalibrator()
                if calibrator.homography_matrix is not None:
                    print("✅ Calibration loaded successfully")
                else:
                    print("❌ No calibration found - using fallback scaling")
                    choice = input("Run calibration now? (y/n): ").lower()
                    if choice == 'y':
                        run_calibration_process(camera_track, thermal_track)
            
            print("\n🚀 Starting WebRTC streaming...")

            @pc.on("datachannel")
            def on_datachannel(channel):
                logger.info(f"Data channel '{channel.label}' received from remote peer.")

                @channel.on("message")
                def on_message(message):
                    logger.info(f"Data channel message received: {message}")

                @channel.on("open")
                def on_open():
                    logger.info(f"Data channel '{channel.label}' opened.")

                @channel.on("close")
                def on_close():
                    logger.info(f"Data channel '{channel.label}' closed.")

            @pc.on("icecandidate")
            async def on_ice_candidate(candidate):
                if candidate and websocket and not websocket.closed:
                    logger.info(f"Generated ICE candidate: {candidate.candidate}")
                    candidate_message_data = {
                        "SdpMid": candidate.sdpMid,
                        "SdpMLineIndex": candidate.sdpMLineIndex,
                        "Candidate": candidate.candidate
                    }
                    full_candidate_message = "CANDIDATE!" + json.dumps(candidate_message_data)
                    try:
                        await websocket.send(full_candidate_message)
                        logger.info("Sent ICE candidate to signaling server")
                    except Exception as e:
                        logger.error(f"Failed to send ICE candidate: {e}")

            async with websockets.connect(uri) as ws_conn:
                websocket = ws_conn
                logger.info(f"Connected to signaling server at {uri}")
                
                register_message = json.dumps({"type": "register", "peer_id": peer_id})
                await websocket.send(register_message)
                logger.info(f"Sent registration: {register_message}")
                
                # Send a notification to web clients that python-peer is connected
                notification_message = json.dumps({"type": "peer_connected", "peer_id": peer_id})
                await websocket.send(notification_message)
                logger.info("Sent peer connection notification to signaling server")

                while True:
                    try:
                        message = await websocket.recv()
                        message = message.decode('utf-8')
                        logger.info(f"Raw message received: {message}")
                        
                        if message.startswith("OFFER!"):
                            json_str = message[len("OFFER!"):]
                            try:
                                data = json.loads(json_str)
                                logger.info("Received offer from client")
                                offer_sdp = data["Sdp"]
                                
                                answer = await run(offer_sdp, pc, camera_track, thermal_track)
                                answer_message_data = {
                                    "SessionType": answer.type.capitalize(),
                                    "Sdp": answer.sdp
                                }
                                full_answer_message = "ANSWER!" + json.dumps(answer_message_data)
                                await websocket.send(full_answer_message)
                                logger.info("Sent answer to client")
                                
                            except json.JSONDecodeError as e:
                                logger.error(f"Malformed OFFER JSON: {e}")
                            except KeyError as e:
                                logger.error(f"Missing key in OFFER: {e}")
                            except Exception as e:
                                logger.error(f"Error processing offer: {e}")
                                
                        elif message.startswith("CANDIDATE!"):
                            json_str = message[len("CANDIDATE!"):]
                            try:
                                data = json.loads(json_str)
                                logger.info("Received ICE candidate from client")
                                parsed_candidate_data = parse_ice_candidate_string(data["Candidate"])
                                
                                ice_candidate = RTCIceCandidate(
                                    foundation=parsed_candidate_data["foundation"],
                                    component=parsed_candidate_data["component"],
                                    protocol=parsed_candidate_data["protocol"],
                                    priority=parsed_candidate_data["priority"],
                                    ip=parsed_candidate_data["ip"],
                                    port=parsed_candidate_data["port"],
                                    type=parsed_candidate_data["type"],
                                    sdpMid=data["SdpMid"],
                                    sdpMLineIndex=data["SdpMLineIndex"],
                                    relatedAddress=parsed_candidate_data["relatedAddress"],
                                    relatedPort=parsed_candidate_data["relatedPort"]
                                )
                                await pc.addIceCandidate(ice_candidate)
                                logger.info("Added ICE candidate from client")
                                
                            except json.JSONDecodeError as e:
                                logger.error(f"Malformed CANDIDATE JSON: {e}")
                            except KeyError as e:
                                logger.error(f"Missing key in CANDIDATE: {e}")
                            except ValueError as e:
                                logger.error(f"Error parsing ICE candidate: {e}")
                            except Exception as e:
                                logger.error(f"Error processing candidate: {e}")
                                
                        elif message == "bye":
                            logger.info("Received 'bye', exiting")
                            return
                        else:
                            logger.info(f"Unhandled message: {message}")
                            
                    except websockets.ConnectionClosedOK:
                        logger.info("WebSocket connection closed gracefully.")
                        break
                    except websockets.ConnectionClosedError as e:
                        logger.error(f"WebSocket connection closed unexpectedly: {e}")
                        break
                    except Exception as e:
                        logger.exception("Error during WebSocket communication")
                        break

        except websockets.exceptions.ConnectionClosedError as e:
            logger.error(f"Failed to connect to the signaling server: {e}")
        except Exception as e:
            logger.exception("An error occurred in main function")
        finally:
            # Clean up resources
            if camera_track:
                camera_track.cleanup()
            if pc and pc.connectionState != "closed":
                await pc.close()
            if websocket and websocket.state != websockets.protocol.State.CLOSED:
                await websocket.close()
            logger.info("Connections closed, attempting reconnection in 5 seconds...")
            await asyncio.sleep(5)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nScript terminated by user.")
    except Exception as e:
        logger.exception("An unhandled error occurred outside main:")