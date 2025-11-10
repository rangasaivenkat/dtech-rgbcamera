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
from aiortc.mediastreams import MediaStreamError  
from av import VideoFrame  # Import VideoFrame for aiortc processing
import websockets  # Import the websockets library
import re  # Import regex for parsing candidate string
from ultralytics import YOLO
import time
import board
import busio
import adafruit_mlx90640
from queue import Queue, Empty
import threading

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Shared thermal data manager for inter-track communication
class ThermalDataManager:
    """Singleton class to share thermal data between tracks"""
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance.thermal_data = None
                    cls._instance.timestamp = 0
                    cls._instance.data_lock = threading.Lock()
        return cls._instance
    
    def update_thermal_data(self, thermal_array, timestamp):
        """Update thermal data (called by thermal camera track)"""
        with self.data_lock:
            self.thermal_data = thermal_array.copy()
            self.timestamp = timestamp
    
    def get_thermal_data(self):
        """Get latest thermal data (called by RGB camera track)"""
        with self.data_lock:
            if self.thermal_data is not None:
                return self.thermal_data.copy(), self.timestamp
            return None, 0

class ThermalVideoStreamTrack(VideoStreamTrack):
    def __init__(self):
        super().__init__()
        self.width = 320
        self.height = 240
        self._initialize_thermal_camera()
        
        # Frame rate control
        self.last_frame_time = 0
        self.frame_interval = 1.0 / 2  # 2 FPS for thermal camera (matches REFRESH_2_HZ)
        
        # Get thermal data manager instance
        self.thermal_manager = ThermalDataManager()
        
        logger.info("ThermalVideoStreamTrack initialized.")

    def _initialize_thermal_camera(self):
        """Initialize MLX90640 thermal camera"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Initialize I2C bus (frequency set in /boot/config.txt)
                self.i2c = busio.I2C(board.SCL, board.SDA)
                
                # Initialize MLX90640 camera
                self.mlx = adafruit_mlx90640.MLX90640(self.i2c)
                logger.info(f"MLX90640 detected, serial number: {[hex(i) for i in self.mlx.serial_number]}")
                
                # Set refresh rate to 2 Hz
                self.mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_2_HZ
                
                # Create a frame buffer to store thermal data
                self.thermal_frame = np.zeros((24 * 32,), dtype=float)
                
                # Allow sensor to stabilize
                time.sleep(1.0)
                
                logger.info(f"Thermal camera initialized successfully on attempt {attempt + 1}")
                return
            except Exception as e:
                logger.error(f"Thermal camera initialization error on attempt {attempt + 1}: {e}")
                time.sleep(0.5)
        
        raise IOError("Cannot initialize thermal camera after multiple attempts. Please ensure MLX90640 is connected.")

    def cleanup(self):
        """Clean up thermal camera resources"""
        if hasattr(self, 'i2c') and self.i2c:
            try:
                self.i2c.deinit()
                logger.info("I2C bus deinitialized")
            except Exception as e:
                logger.error(f"Failed to deinitialize I2C: {e}")
        logger.info("Thermal camera resources cleaned up")

    def __del__(self):
        self.cleanup()

    def thermal_to_colormap(self, thermal_data):
        """Convert thermal data to displayable color image"""
        # Calculate dynamic range
        vmin = np.percentile(thermal_data, 5)
        vmax = np.percentile(thermal_data, 95)
        
        # Scale to 0-1 range
        scaled = np.clip((thermal_data - vmin) / (vmax - vmin + 1e-8), 0, 1)
        
        # Convert to 8-bit
        thermal_8bit = (scaled * 255).astype(np.uint8)
        
        # Resize to desired output size
        thermal_resized = cv2.resize(thermal_8bit, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        
        # Apply colormap
        thermal_color = cv2.applyColorMap(thermal_resized, cv2.COLORMAP_INFERNO)
        
        return thermal_color

    async def recv(self):
        """Receive and process thermal video frames"""
        pts = None
        time_base = None
        
        try:
            # Frame rate control
            current_time = time.time()
            if current_time - self.last_frame_time < self.frame_interval:
                await asyncio.sleep(self.frame_interval - (current_time - self.last_frame_time))
            
            self.last_frame_time = time.time()
            timestamp = self.last_frame_time
            
            pts, time_base = await self.next_timestamp()

            # Read thermal frame with retries
            max_attempts = 3
            for attempt in range(max_attempts):
                try:
                    self.mlx.getFrame(self.thermal_frame)
                    thermal_data = np.array(self.thermal_frame).reshape((24, 32))
                    
                    # Share thermal data with RGB track
                    self.thermal_manager.update_thermal_data(thermal_data, timestamp)
                    
                    # Convert to color image
                    frame = self.thermal_to_colormap(thermal_data)
                    
                    # Add temperature info overlay
                    avg_temp = np.mean(thermal_data)
                    min_temp = np.min(thermal_data)
                    max_temp = np.max(thermal_data)
                    
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
                    
                    # Convert BGR to RGB for WebRTC
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    video_frame = VideoFrame.from_ndarray(frame, format="rgb24")
                    video_frame.pts = pts
                    video_frame.time_base = time_base
                    return video_frame
                
                except Exception as e:
                    logger.error(f"Attempt {attempt + 1} failed: {e}")
                    if attempt < max_attempts - 1:
                        logger.info("Reinitializing thermal camera...")
                        self._initialize_thermal_camera()
                        await asyncio.sleep(0.5)
                    else:
                        logger.error("Max retries reached, returning error frame")
                        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                        cv2.rectangle(frame, (50, 50), (100, 100), (0, 0, 255), -1)
                        cv2.putText(frame, "Thermal Error", (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        video_frame = VideoFrame.from_ndarray(frame, format="rgb24")
                        video_frame.pts = pts
                        video_frame.time_base = time_base
                        return video_frame
                        
        except MediaStreamError:
            # Track has been stopped - this is normal when stopping the stream
            logger.info("Thermal track stopped (MediaStreamError)")
            raise  # Re-raise to signal track has stopped
        except Exception as e:
            logger.error(f"Error in thermal recv(): {e}")
            if pts is not None and time_base is not None:
                frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                cv2.rectangle(frame, (50, 50), (100, 100), (0, 0, 255), -1)
                video_frame = VideoFrame.from_ndarray(frame, format="rgb24")
                video_frame.pts = pts
                video_frame.time_base = time_base
                return video_frame
            else:
                # If we don't have pts/time_base, we can't return a proper frame
                raise

class OptimizedCameraVideoStreamTrack(VideoStreamTrack):
    def __init__(self):
        super().__init__()
        self.cap = None
        # Optimized resolution - balance between quality and performance
        self.width = 640
        self.height = 480
        self._initialize_camera()
        
        # Load YOLOv8 model with optimizations
        try:
            self.model = YOLO("yolov8n.pt")
            # Warm up the model with a dummy frame
            dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            self.model(dummy_frame, verbose=False, imgsz=640, conf=0.5)
            logger.info("YOLO model loaded and warmed up successfully")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise
            
        self.target_class_ids = {0: "person", 56: "chair", 60: "dining table"}
        
        # Motion tracking parameters (optimized)
        self.motion_threshold = 12  # Lowered for better sensitivity
        self.dead_time_seconds = 5
        self.human_tracker = {}  # Dictionary to track {person_id: {last_move: timestamp, last_temp: float}}
        self.prev_gray = None  # Will be initialized on first frame
        
        # Thermal detection parameters
        self.thermal_manager = ThermalDataManager()
        self.min_human_temp = 30.0  # Minimum body temperature (°C)
        self.max_human_temp = 42.0  # Maximum body temperature (°C)
        self.temp_drop_threshold = 3.0  # Temperature drop indicating death (°C)
        self.thermal_weight = 0.6  # Weight for thermal vs motion (60% thermal, 40% motion)
        
        # Frame rate and processing optimization
        self.target_fps = 30
        self.frame_interval = 1.0 / self.target_fps
        self.last_frame_time = 0
        
        # Smart detection optimization
        self.frame_counter = 0
        self.detection_interval = 2  # Run YOLO every 2 frames
        self.last_results = None
        self.last_detection_frame = 0
        
        # Frame preprocessing optimization
        self.gaussian_kernel = (5, 5)
        
        # Threading for YOLO processing
        self.yolo_queue = Queue(maxsize=2)
        self.result_queue = Queue(maxsize=2)
        self.yolo_thread = threading.Thread(target=self._yolo_worker, daemon=True)
        self.yolo_thread.start()
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        logger.info("Optimized CameraVideoStreamTrack initialized with thermal integration.")

    def _yolo_worker(self):
        """Background thread for YOLO processing"""
        while True:
            try:
                frame_data = self.yolo_queue.get(timeout=1.0)
                if frame_data is None:  # Shutdown signal
                    break
                    
                frame, timestamp = frame_data
                results = self.model(frame, verbose=False, imgsz=640, conf=0.4, iou=0.5)
                
                try:
                    self.result_queue.put((results[0], timestamp), block=False)
                except:
                    pass  # Queue full, skip this result
                    
            except Empty:
                continue
            except Exception as e:
                logger.error(f"YOLO worker error: {e}")

    def _initialize_camera(self):
        """Initialize camera with optimized pipeline for better quality and lower latency"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if self.cap:
                    self.cap.release()
                    time.sleep(0.3)
                    logger.info("Previous camera instance released")
                    
                # Highly optimized GStreamer pipeline
                pipeline = (
                    f"libcamerasrc ! "
                    f"video/x-raw, width={self.width}, height={self.height}, format=YUY2, framerate=30/1 ! "
                    f"queue max-size-buffers=2 leaky=downstream ! "
                    f"videoconvert n-threads=4 ! "
                    f"video/x-raw, format=BGR ! "
                    f"appsink sync=false drop=true max-buffers=2 emit-signals=true"
                )
                
                self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
                
                if self.cap.isOpened():
                    # Set additional optimizations
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer
                    self.cap.set(cv2.CAP_PROP_FPS, 30)
                    
                    # Test capture
                    ret, test_frame = self.cap.read()
                    if ret and test_frame is not None:
                        logger.info(f"Camera initialized successfully on attempt {attempt + 1}")
                        logger.info(f"Actual resolution: {test_frame.shape[1]}x{test_frame.shape[0]}")
                        return
                    else:
                        logger.warning(f"Camera opened but failed to capture test frame on attempt {attempt + 1}")
                else:
                    logger.warning(f"Failed to open camera pipeline on attempt {attempt + 1}")
                    
                time.sleep(0.3)
            except Exception as e:
                logger.error(f"Camera initialization error on attempt {attempt + 1}: {e}")
                time.sleep(0.3)
        
        raise IOError("Cannot open camera pipeline after multiple attempts. Please ensure camera is connected and libcamera is configured.")

    def cleanup(self):
        """Clean up camera resources"""
        # Shutdown YOLO worker
        try:
            self.yolo_queue.put(None, timeout=1.0)  # Shutdown signal
            self.yolo_thread.join(timeout=2.0)
        except:
            pass
            
        if self.cap and self.cap.isOpened():
            self.cap.release()
            time.sleep(0.2)
            logger.info("Camera released.")
        self.cap = None

    def __del__(self):
        self.cleanup()
        
    async def recv(self):
        """Optimized frame receiving and processing"""
        pts = None
        time_base = None
        
        try:
            # Precise frame timing
            current_time = time.time()
            delta = current_time - self.last_frame_time
            if delta < self.frame_interval:
                await asyncio.sleep(self.frame_interval - delta)
            
            self.last_frame_time = current_time
            timestamp = current_time
            
            pts, time_base = await self.next_timestamp()

            # Check camera availability
            if not self.cap or not self.cap.isOpened():
                logger.warning("Camera not available, reinitializing...")
                self._initialize_camera()

            # Capture frame
            ret, frame = self.cap.read()
            if not ret or frame is None:
                logger.error("Failed to read frame from camera")
                frame = self._create_error_frame()
            else:
                # Process the frame efficiently
                frame = await self._process_frame_optimized(frame, timestamp)

            # Update FPS counter
            self._update_fps_counter()

            # Convert BGR to RGB for WebRTC
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_frame = VideoFrame.from_ndarray(frame, format="rgb24")
            video_frame.pts = pts
            video_frame.time_base = time_base

            return video_frame
            
        except MediaStreamError:
            # Track has been stopped - this is normal when stopping the stream
            logger.info("Camera track stopped (MediaStreamError)")
            raise  # Re-raise to signal track has stopped
        except Exception as e:
            logger.error(f"Error in recv(): {e}")
            if pts is not None and time_base is not None:
                frame = self._create_error_frame()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_frame = VideoFrame.from_ndarray(frame, format="rgb24")
                video_frame.pts = pts
                video_frame.time_base = time_base
                return video_frame
            else:
                # If we don't have pts/time_base, we can't return a proper frame
                raise

    def _create_error_frame(self):
        """Create error frame"""
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        cv2.rectangle(frame, (50, 50), (150, 150), (0, 0, 255), -1)
        cv2.putText(frame, "Camera Error", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return frame

    def _update_fps_counter(self):
        """Update FPS counter"""
        self.fps_counter += 1
        if self.fps_counter % 30 == 0:  # Update every 30 frames
            current_time = time.time()
            elapsed = current_time - self.fps_start_time
            if elapsed > 0:
                self.current_fps = 30 / elapsed
                self.fps_start_time = current_time

    def _get_thermal_temperature(self, x1, y1, x2, y2, thermal_data):
        """Extract average temperature from thermal data for a bounding box region"""
        try:
            # Thermal camera is 32x24, RGB is 640x480
            # Scale bounding box coordinates to thermal resolution
            thermal_height, thermal_width = 24, 32
            
            # Calculate scaled coordinates
            scale_x = thermal_width / self.width
            scale_y = thermal_height / self.height
            
            thermal_x1 = int(x1 * scale_x)
            thermal_y1 = int(y1 * scale_y)
            thermal_x2 = int(x2 * scale_x)
            thermal_y2 = int(y2 * scale_y)
            
            # Ensure coordinates are within thermal image bounds
            thermal_x1 = max(0, min(thermal_x1, thermal_width - 1))
            thermal_y1 = max(0, min(thermal_y1, thermal_height - 1))
            thermal_x2 = max(0, min(thermal_x2, thermal_width - 1))
            thermal_y2 = max(0, min(thermal_y2, thermal_height - 1))
            
            # Ensure we have a valid region
            if thermal_x2 <= thermal_x1 or thermal_y2 <= thermal_y1:
                return None
            
            # Extract region from thermal data
            roi = thermal_data[thermal_y1:thermal_y2, thermal_x1:thermal_x2]
            
            if roi.size == 0:
                return None
            
            # Return average temperature in the region
            avg_temp = np.mean(roi)
            max_temp = np.max(roi)
            
            # Return the max temperature (more reliable for detecting body heat)
            return max_temp
            
        except Exception as e:
            logger.debug(f"Error extracting thermal temperature: {e}")
            return None

    async def _process_frame_optimized(self, frame, timestamp):
        """Highly optimized frame processing with smart YOLO usage and thermal integration"""
        # Add processing indicator
        cv2.rectangle(frame, (50, 50), (100, 100), (0, 255, 0), 2)

        # Convert to grayscale for motion tracking (optimized)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Initialize prev_gray if not set
        if self.prev_gray is None:
            self.prev_gray = gray.copy()

        # Get thermal data
        thermal_data, thermal_timestamp = self.thermal_manager.get_thermal_data()
        thermal_available = thermal_data is not None and (timestamp - thermal_timestamp) < 2.0
        
        if not thermal_available:
            # Show warning if thermal data is stale or unavailable
            cv2.putText(frame, "Thermal: NOT AVAILABLE", (self.width - 250, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Initialize tracking variables
        humans_detected = 0
        possibly_dead = 0

        self.frame_counter += 1
        
        # Smart YOLO processing - only when needed
        should_run_yolo = (
            self.frame_counter % self.detection_interval == 0 or  # Regular interval
            self.last_results is None or  # No previous results
            (timestamp - self.last_detection_frame) > 1.0  # Force detection every second
        )
        
        if should_run_yolo:
            # Submit frame for background YOLO processing
            try:
                # Resize frame for YOLO to balance speed and accuracy
                yolo_frame = cv2.resize(frame, (640, 480)) if frame.shape[:2] != (480, 640) else frame.copy()
                self.yolo_queue.put((yolo_frame, timestamp), block=False)
            except:
                pass  # Queue full, skip this frame
        
        # Check for YOLO results
        yolo_updated = False
        try:
            results, result_timestamp = self.result_queue.get_nowait()
            self.last_results = results
            self.last_detection_frame = result_timestamp
            yolo_updated = True
        except Empty:
            pass

        # Process detections (either new or cached)
        if self.last_results is not None:
            humans_detected, possibly_dead = self._process_detections_with_motion_and_thermal(
                self.last_results, frame, gray, timestamp, yolo_updated, thermal_data, thermal_available
            )

        # Update prev_gray for next frame
        self.prev_gray = gray.copy()

        # Add optimized overlays
        self._add_frame_overlays(frame, humans_detected, possibly_dead, thermal_available)
        
        return frame

    def _process_detections_with_motion_and_thermal(self, results, frame, gray, timestamp, yolo_updated, thermal_data, thermal_available):
        """Process YOLO detections with motion tracking AND thermal analysis"""
        humans_detected = 0
        possibly_dead = 0
        current_detections = set()

        for box in results.boxes.data:
            x1, y1, x2, y2, conf, cls = box.cpu().numpy()
            class_id = int(cls)
            if class_id not in self.target_class_ids:
                continue

            label_name = self.target_class_ids[class_id]
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            
            # Ensure coordinates are within frame bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            if class_id == 0:  # Person detection
                humans_detected += 1
                
                # Create a more stable person_id based on center position
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                person_id = f"person_{center_x//50}_{center_y//50}"  # Grid-based ID for stability
                
                current_detections.add(person_id)
                
                # 1. CHECK MOTION (RGB-based)
                moved = False
                if self.prev_gray is not None and y2 > y1 and x2 > x1:
                    try:
                        roi_now = gray[y1:y2, x1:x2]
                        roi_prev = self.prev_gray[y1:y2, x1:x2]
                        
                        if roi_now.shape == roi_prev.shape and roi_now.size > 0:
                            moved = self._detect_motion_optimized(roi_now, roi_prev)
                    except Exception as e:
                        logger.debug(f"Motion detection error: {e}")
                        moved = True  # Assume movement on error

                # 2. CHECK THERMAL (Temperature-based)
                current_temp = None
                thermal_alive = None  # None = unknown, True = alive, False = dead
                
                if thermal_available and thermal_data is not None:
                    current_temp = self._get_thermal_temperature(x1, y1, x2, y2, thermal_data)
                    
                    if current_temp is not None:
                        # Check if temperature is in human range
                        if self.min_human_temp <= current_temp <= self.max_human_temp:
                            thermal_alive = True
                            
                            # Check for temperature drop (indicating death)
                            if person_id in self.human_tracker and 'last_temp' in self.human_tracker[person_id]:
                                last_temp = self.human_tracker[person_id]['last_temp']
                                if last_temp > 0:  # Valid previous temperature
                                    temp_drop = last_temp - current_temp
                                    
                                    if temp_drop > self.temp_drop_threshold:
                                        thermal_alive = False
                                        logger.warning(f"{person_id}: Temperature dropped {temp_drop:.1f}°C (was {last_temp:.1f}°C, now {current_temp:.1f}°C)")
                        else:
                            # Temperature outside human range
                            if current_temp < self.min_human_temp:
                                thermal_alive = False  # Too cold
                            else:
                                thermal_alive = None  # Too hot (might be error)

                # 3. UPDATE TRACKING
                if person_id not in self.human_tracker:
                    # First time seeing this person
                    self.human_tracker[person_id] = {
                        'last_move': timestamp,
                        'last_temp': current_temp if current_temp else 0.0
                    }
                    logger.info(f"New person detected: {person_id} (Temp: {current_temp:.1f}°C)" if current_temp else f"New person detected: {person_id}")
                
                # Update last movement time if motion detected
                if moved:
                    self.human_tracker[person_id]['last_move'] = timestamp
                
                # Update temperature
                if current_temp is not None:
                    self.human_tracker[person_id]['last_temp'] = current_temp

                # 4. DETERMINE ALIVE/DEAD STATUS (Combined logic)
                last_move_time = self.human_tracker[person_id]['last_move']
                time_inactive = timestamp - last_move_time
                
                # Decision logic: Combine motion and thermal
                is_dead = False
                status_reason = ""
                
                if thermal_available and thermal_alive is not None:
                    # We have thermal data - use weighted decision
                    motion_score = 1.0 if moved else (0.0 if time_inactive > self.dead_time_seconds else 0.5)
                    thermal_score = 1.0 if thermal_alive else 0.0
                    
                    # Weighted combination
                    alive_score = (motion_score * (1 - self.thermal_weight)) + (thermal_score * self.thermal_weight)
                    
                    is_dead = alive_score < 0.4  # Threshold for death
                    
                    if is_dead:
                        if not thermal_alive and time_inactive > self.dead_time_seconds:
                            status_reason = "Cold + No Motion"
                        elif not thermal_alive:
                            status_reason = f"Cold ({current_temp:.1f}°C)"
                        else:
                            status_reason = f"No Motion ({time_inactive:.1f}s)"
                    else:
                        if moved and thermal_alive:
                            status_reason = "Moving + Warm"
                        elif moved:
                            status_reason = "Moving"
                        elif thermal_alive:
                            status_reason = f"Warm ({current_temp:.1f}°C)"
                        else:
                            status_reason = f"Monitoring ({time_inactive:.1f}s)"
                else:
                    # No thermal data - fall back to motion only
                    is_dead = time_inactive > self.dead_time_seconds
                    status_reason = f"No Motion ({time_inactive:.1f}s)" if is_dead else ("Moving" if moved else f"Still ({time_inactive:.1f}s)")

                # Count possibly dead
                if is_dead:
                    possibly_dead += 1

                # Visual feedback
                if is_dead:
                    label = f"DEAD: {status_reason}"
                    color = (0, 0, 255)  # Red
                elif moved or (thermal_alive and time_inactive < self.dead_time_seconds):
                    label = f"ALIVE: {status_reason}"
                    color = (0, 255, 0)  # Green
                else:
                    label = f"UNCERTAIN: {status_reason}"
                    color = (0, 255, 255)  # Yellow
                
                # Draw enhanced detection box
                thickness = 3 if is_dead else 2
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                
                # Enhanced label with background
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(frame, (x1, y1-25), (x1+label_size[0]+10, y1-2), color, -1)
                cv2.putText(frame, label, (x1+5, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # Draw temperature indicator if available
                if current_temp is not None:
                    temp_text = f"{current_temp:.1f}C"
                    cv2.putText(frame, temp_text, (x1+5, y2-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # Draw motion indicator (left circle)
                if moved:
                    cv2.circle(frame, (x1+15, y1+15), 8, (0, 255, 0), -1)
                else:
                    cv2.circle(frame, (x1+15, y1+15), 8, (128, 128, 128), -1)
                
                # Draw thermal indicator (right circle)
                if thermal_alive is not None:
                    thermal_color = (0, 255, 0) if thermal_alive else (0, 0, 255)
                    cv2.circle(frame, (x2-15, y1+15), 8, thermal_color, -1)
                elif thermal_available:
                    cv2.circle(frame, (x2-15, y1+15), 8, (128, 128, 128), -1)
                
            else:
                # Non-person objects
                color = (0, 128, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label_name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Clean up tracker - remove old detections not seen in current frame
        if yolo_updated:
            # Only clean up when we have fresh YOLO results
            to_remove = []
            for person_id in self.human_tracker.keys():
                if person_id not in current_detections:
                    # Person not detected in this frame
                    time_since_last_seen = timestamp - self.human_tracker[person_id]['last_move']
                    if time_since_last_seen > self.dead_time_seconds * 2:  # Remove after 2x dead time
                        to_remove.append(person_id)
            
            for person_id in to_remove:
                del self.human_tracker[person_id]
                logger.info(f"Removed old tracking for: {person_id}")

        return humans_detected, possibly_dead

    def _detect_motion_optimized(self, current_roi, previous_roi):
        """Optimized motion detection with better sensitivity"""
        # Use absdiff for faster computation
        diff = cv2.absdiff(current_roi, previous_roi)
        
        # Apply Gaussian blur to reduce noise
        diff = cv2.GaussianBlur(diff, (5, 5), 0)
        
        # Optimized threshold
        _, motion_mask = cv2.threshold(diff, self.motion_threshold, 255, cv2.THRESH_BINARY)
        
        # Count non-zero pixels efficiently
        motion_pixels = cv2.countNonZero(motion_mask)
        motion_ratio = motion_pixels / (current_roi.shape[0] * current_roi.shape[1])
        
        # Lower threshold for better sensitivity (2% instead of 5%)
        return motion_ratio > 0.02

    def _add_frame_overlays(self, frame, humans_detected, possibly_dead, thermal_available):
        """Add optimized frame overlays with better visual quality"""
        # Title with background
        title_size = cv2.getTextSize("RGB + YOLO + THERMAL", cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        cv2.rectangle(frame, (15, 5), (15 + title_size[0] + 10, 35), (0, 0, 0), -1)
        cv2.putText(frame, "RGB + YOLO + THERMAL", (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Stats with backgrounds
        stats = [
            (f"Humans: {humans_detected}", (0, 255, 0)),
            (f"Dead: {possibly_dead}", (0, 0, 255)),
            (f"FPS: {self.current_fps:.1f}", (0, 255, 255)),
            (f"Thermal: {'OK' if thermal_available else 'UNAVAIL'}", (0, 255, 0) if thermal_available else (255, 0, 0))
        ]
        
        y_pos = 55
        for text, color in stats:
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (15, y_pos - 20), (15 + text_size[0] + 10, y_pos + 5), (0, 0, 0), -1)
            cv2.putText(frame, text, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_pos += 28
        
        # RGB-specific visual marker (enhanced green border)
        cv2.rectangle(frame, (0, 0), (frame.shape[1]-1, frame.shape[0]-1), (0, 255, 0), 4)
        
        # Legend (bottom left)
        legend_y = frame.shape[0] - 90
        cv2.rectangle(frame, (10, legend_y - 5), (220, frame.shape[0] - 10), (0, 0, 0), -1)
        cv2.putText(frame, "Legend:", (15, legend_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.circle(frame, (25, legend_y + 35), 6, (0, 255, 0), -1)
        cv2.putText(frame, "Motion Detected", (35, legend_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.circle(frame, (25, legend_y + 55), 6, (0, 255, 0), -1)
        cv2.putText(frame, "Warm (Thermal)", (35, legend_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.circle(frame, (25, legend_y + 75), 6, (0, 0, 255), -1)
        cv2.putText(frame, "Cold (Dead)", (35, legend_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Performance indicator (top right)
        mode_text = "Mode: THERMAL+MOTION"
        mode_size = cv2.getTextSize(mode_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(frame, (frame.shape[1] - mode_size[0] - 20, 5), 
                     (frame.shape[1] - 5, 30), (0, 100, 0), -1)
        cv2.putText(frame, mode_text, (frame.shape[1] - mode_size[0] - 15, 22), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

# Replace the original CameraVideoStreamTrack with the optimized version
CameraVideoStreamTrack = OptimizedCameraVideoStreamTrack

async def run(offer, pc, camera_track, thermal_track):
    @pc.on("track")
    def on_track(track):
        logger.info(f"Received track: {track.kind}")

    @pc.on("iceconnectionstatechange")
    async def on_ice_connection_state_change():
        logger.info(f"ICE connection state: {pc.iceConnectionState}")
        if pc.iceConnectionState in ["failed", "disconnected"]:
            logger.warning(f"ICE connection issue: {pc.iceConnectionState}")
            if pc and pc.connectionState != "closed":
                await pc.close()
            logger.info("Peer connection closed due to ICE failure")

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
    
    while True:  # Main reconnection loop
        websocket = None
        pc = None
        camera_track = None
        thermal_track = None
        offer_received = False  # Track if we've received an offer
        
        try:
            # Initialize camera tracks with optimized settings
            try:
                camera_track = OptimizedCameraVideoStreamTrack()
                logger.info("Optimized RGB camera track initialized")
            except Exception as e:
                logger.error(f"Failed to initialize RGB camera track: {e}")
                await asyncio.sleep(5)
                continue
                
            try:
                thermal_track = ThermalVideoStreamTrack()
                logger.info("Thermal camera track initialized")
            except Exception as e:
                logger.error(f"Failed to initialize thermal camera track: {e}")
                await asyncio.sleep(5)
                continue

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
                                
                                # Create NEW peer connection for each offer
                                if pc:
                                    logger.info("Closing existing peer connection before creating new one")
                                    await pc.close()
                                    await asyncio.sleep(0.5)  # Give it time to close
                                
                                pc = RTCPeerConnection()
                                logger.info("Created new RTCPeerConnection")
                                
                                # Setup data channel handler
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

                                # Setup ICE candidate handler
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
                                
                                # Setup connection state handlers
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
                                
                                # Process the offer and create answer
                                answer = await run(offer_sdp, pc, camera_track, thermal_track)
                                
                                if answer:
                                    answer_message_data = {
                                        "SessionType": answer.type.capitalize(),
                                        "Sdp": answer.sdp
                                    }
                                    full_answer_message = "ANSWER!" + json.dumps(answer_message_data)
                                    await websocket.send(full_answer_message)
                                    logger.info("Sent answer to client")
                                    offer_received = True
                                else:
                                    logger.error("Failed to create answer")
                                
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
                                
                                if not pc:
                                    logger.warning("Received ICE candidate but no peer connection exists")
                                    continue
                                
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
            logger.info("Cleaning up resources...")
            
            if camera_track:
                try:
                    camera_track.cleanup()
                    logger.info("Camera track cleaned up")
                except Exception as e:
                    logger.error(f"Error cleaning up camera track: {e}")
                    
            if thermal_track:
                try:
                    thermal_track.cleanup()
                    logger.info("Thermal track cleaned up")
                except Exception as e:
                    logger.error(f"Error cleaning up thermal track: {e}")
                    
            if pc and pc.connectionState != "closed":
                try:
                    await pc.close()
                    logger.info("Peer connection closed")
                except Exception as e:
                    logger.error(f"Error closing peer connection: {e}")
                    
            if websocket and websocket.state != websockets.protocol.State.CLOSED:
                try:
                    await websocket.close()
                    logger.info("WebSocket closed")
                except Exception as e:
                    logger.error(f"Error closing websocket: {e}")
                    
            logger.info("All resources cleaned up, attempting reconnection in 5 seconds...")
            await asyncio.sleep(5)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nScript terminated by user.")
    except Exception as e:
        logger.exception("An unhandled error occurred outside main:")
