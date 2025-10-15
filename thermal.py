import sys
print(sys.executable)
import asyncio
import logging
import json
import cv2
import numpy as np
from aiortc import RTCIceCandidate, RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from aiortc.mediastreams import VideoStreamTrack
from av import VideoFrame
import websockets
from ultralytics import YOLO
import time
import board
import busio
import adafruit_mlx90640
import os
from datetime import datetime
from typing import Optional, Dict, List, Tuple
import threading

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DualCameraCalibrator:
    """Handles calibration between RGB and thermal cameras"""
    def __init__(self, rgb_resolution=(320, 240), thermal_resolution=(32, 24)):
        self.rgb_resolution = rgb_resolution
        self.thermal_resolution = thermal_resolution
        self.homography_matrix = None
        self.calibration_file = "camera_calibration.json"
        self.load_calibration()
        
    def load_calibration(self) -> bool:
        """Load existing calibration from file"""
        try:
            if os.path.exists(self.calibration_file):
                with open(self.calibration_file, 'r') as f:
                    data = json.load(f)
                self.homography_matrix = np.array(data['homography_matrix'])
                logger.info(f"✅ Loaded calibration (error: {data.get('reprojection_error', 0):.2f}px)")
                return True
        except Exception as e:
            logger.warning(f"⚠️  No calibration loaded: {e}")
        return False
    
    def save_calibration(self, rgb_points: List, thermal_points: List) -> bool:
        """Calculate and save homography matrix"""
        try:
            if len(rgb_points) < 4 or len(thermal_points) < 4:
                logger.error("Need at least 4 point pairs")
                return False
                
            src_points = np.float32(rgb_points)
            dst_points = np.float32(thermal_points)
            
            self.homography_matrix, mask = cv2.findHomography(
                src_points, dst_points, cv2.RANSAC, ransacReprojThreshold=2.0
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
                }
                
                with open(self.calibration_file, 'w') as f:
                    json.dump(calibration_data, f, indent=2)
                
                logger.info(f"✅ Calibration saved (error: {error:.2f}px)")
                return True
        except Exception as e:
            logger.error(f"Calibration save failed: {e}")
        return False
    
    def _calculate_reprojection_error(self, rgb_points, thermal_points) -> float:
        """Calculate average reprojection error"""
        if self.homography_matrix is None:
            return float('inf')
        
        rgb_pts = np.float32(rgb_points).reshape(-1, 1, 2)
        projected_pts = cv2.perspectiveTransform(rgb_pts, self.homography_matrix)
        projected_pts = projected_pts.reshape(-1, 2)
        
        errors = [np.sqrt((tx - px)**2 + (ty - py)**2) 
                  for (tx, ty), (px, py) in zip(thermal_points, projected_pts)]
        return np.mean(errors)
    
    def map_rgb_bbox_to_thermal(self, rgb_bbox: Tuple[int, int, int, int]) -> Optional[Tuple[int, int, int, int]]:
        """Map RGB bounding box to thermal coordinates"""
        x1, y1, x2, y2 = rgb_bbox
        
        if self.homography_matrix is None:
            # Fallback to simple scaling
            thermal_x1 = max(0, min(31, int(x1 * 32 / 320)))
            thermal_y1 = max(0, min(23, int(y1 * 24 / 240)))
            thermal_x2 = max(0, min(31, int(x2 * 32 / 320)))
            thermal_y2 = max(0, min(23, int(y2 * 24 / 240)))
        else:
            # Use homography for accurate mapping
            corners = np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]).reshape(-1, 1, 2)
            mapped_corners = cv2.perspectiveTransform(corners, self.homography_matrix).reshape(-1, 2)
            
            thermal_x1 = max(0, min(31, int(np.min(mapped_corners[:, 0]))))
            thermal_y1 = max(0, min(23, int(np.min(mapped_corners[:, 1]))))
            thermal_x2 = max(0, min(31, int(np.max(mapped_corners[:, 0]))))
            thermal_y2 = max(0, min(23, int(np.max(mapped_corners[:, 1]))))
        
        if thermal_x2 > thermal_x1 and thermal_y2 > thermal_y1:
            return (thermal_x1, thermal_y1, thermal_x2, thermal_y2)
        return None

class ThermalDataBuffer:
    """Thread-safe buffer for thermal data sharing between tracks"""
    def __init__(self):
        self.thermal_data = None
        self.timestamp = 0
        self.lock = threading.Lock()
    
    def update(self, data: np.ndarray):
        """Update thermal data"""
        with self.lock:
            self.thermal_data = data.copy() if data is not None else None
            self.timestamp = time.time()
    
    def get(self) -> Optional[np.ndarray]:
        """Get latest thermal data"""
        with self.lock:
            return self.thermal_data.copy() if self.thermal_data is not None else None
    
    def get_age(self) -> float:
        """Get age of thermal data in seconds"""
        with self.lock:
            return time.time() - self.timestamp if self.timestamp > 0 else float('inf')

class CalibratedHumanDetector:
    """Analyzes human detections using thermal data"""
    def __init__(self, calibrator: DualCameraCalibrator):
        self.calibrator = calibrator
        self.TEMP_THRESHOLDS = {
            'human_min': 32.0,
            'human_normal': 35.0,
            'human_fever': 38.0,
            'ambient_diff_min': 3.0,
            'dead_threshold': 28.0,
        }
        
    def analyze_humans(self, yolo_results, thermal_data: Optional[np.ndarray]) -> List[Dict]:
        """Analyze all human detections"""
        if thermal_data is None or yolo_results is None:
            return []
        
        ambient_temp = np.percentile(thermal_data, 25)
        analyses = []
        
        for box in yolo_results.boxes.data:
            x1, y1, x2, y2, conf, cls = box
            if int(cls) == 0:  # Person class
                rgb_bbox = (int(x1), int(y1), int(x2), int(y2))
                analysis = self._analyze_single(rgb_bbox, thermal_data, ambient_temp)
                if analysis:
                    analysis['rgb_bbox'] = rgb_bbox
                    analysis['yolo_conf'] = float(conf)
                    analyses.append(analysis)
        
        return analyses
    
    def _analyze_single(self, rgb_bbox, thermal_data, ambient_temp) -> Optional[Dict]:
        """Analyze single human detection"""
        thermal_bbox = self.calibrator.map_rgb_bbox_to_thermal(rgb_bbox)
        if thermal_bbox is None:
            return None
        
        tx1, ty1, tx2, ty2 = thermal_bbox
        thermal_region = thermal_data[ty1:ty2, tx1:tx2]
        
        if thermal_region.size == 0:
            return None
        
        # Thermal statistics
        max_temp = np.max(thermal_region)
        mean_temp = np.mean(thermal_region)
        hot_pixels = np.sum(thermal_region >= self.TEMP_THRESHOLDS['human_min'])
        hot_ratio = hot_pixels / thermal_region.size
        temp_above_ambient = max_temp - ambient_temp
        
        # Determine status
        status, confidence = self._classify_status(max_temp, temp_above_ambient, hot_ratio)
        
        return {
            'status': status,
            'confidence': round(confidence, 2),
            'max_temp': round(max_temp, 1),
            'mean_temp': round(mean_temp, 1),
            'ambient_temp': round(ambient_temp, 1),
            'temp_above_ambient': round(temp_above_ambient, 1),
            'hot_ratio': round(hot_ratio, 2),
            'thermal_bbox': thermal_bbox
        }
    
    def _classify_status(self, max_temp, temp_above_ambient, hot_ratio) -> Tuple[str, float]:
        """Classify human status with confidence"""
        # Calculate confidence components
        temp_conf = 1.0 if max_temp >= self.TEMP_THRESHOLDS['human_normal'] else \
                   0.7 if max_temp >= self.TEMP_THRESHOLDS['human_min'] else \
                   0.3 if max_temp >= 30.0 else 0.0
        
        ambient_conf = 1.0 if temp_above_ambient >= self.TEMP_THRESHOLDS['ambient_diff_min'] else \
                      0.5 if temp_above_ambient >= 2.0 else 0.0
        
        pixel_conf = min(1.0, hot_ratio * 4)
        
        # Weighted confidence
        confidence = temp_conf * 0.5 + ambient_conf * 0.3 + pixel_conf * 0.2
        
        # Classify based on confidence and temperature
        if confidence >= 0.75:
            if max_temp >= self.TEMP_THRESHOLDS['human_fever']:
                return "ALIVE-Fever", confidence
            elif max_temp >= self.TEMP_THRESHOLDS['human_normal']:
                return "ALIVE-Normal", confidence
            else:
                return "ALIVE-Cool", confidence
        elif confidence >= 0.5:
            return "LIKELY ALIVE", confidence
        elif confidence >= 0.3:
            return "UNCERTAIN", confidence
        else:
            return "LIKELY DEAD" if max_temp <= self.TEMP_THRESHOLDS['dead_threshold'] else "UNKNOWN", confidence

class ThermalVideoStreamTrack(VideoStreamTrack):
    """Thermal camera video stream with data sharing"""
    def __init__(self, thermal_buffer: ThermalDataBuffer):
        super().__init__()
        self.width = 320
        self.height = 240
        self.thermal_buffer = thermal_buffer
        self.frame_interval = 1.0 / 8  # 8 FPS
        self.last_frame_time = 0
        self._initialize_thermal_camera()
        logger.info("✅ Thermal camera track initialized")

    def _initialize_thermal_camera(self):
        """Initialize MLX90640 thermal camera"""
        for attempt in range(3):
            try:
                self.i2c = busio.I2C(board.SCL, board.SDA, frequency=400000)
                self.mlx = adafruit_mlx90640.MLX90640(self.i2c)
                self.mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_8_HZ
                self.thermal_frame = np.zeros((24 * 32,), dtype=float)
                logger.info(f"Thermal camera ready (attempt {attempt + 1})")
                return
            except Exception as e:
                logger.error(f"Thermal init attempt {attempt + 1} failed: {e}")
                time.sleep(0.5)
        raise IOError("Cannot initialize thermal camera")

    def get_thermal_data(self) -> Optional[np.ndarray]:
        """Get raw thermal data"""
        try:
            self.mlx.getFrame(self.thermal_frame)
            return np.array(self.thermal_frame).reshape((24, 32))
        except Exception as e:
            logger.error(f"Thermal data read error: {e}")
            return None

    def thermal_to_colormap(self, thermal_data: np.ndarray) -> np.ndarray:
        """Convert thermal data to color image"""
        vmin, vmax = 20.0, 50.0
        scaled = np.clip((thermal_data - vmin) / (vmax - vmin), 0, 1)
        thermal_8bit = (scaled * 255).astype(np.uint8)
        thermal_resized = cv2.resize(thermal_8bit, (self.width, self.height), 
                                    interpolation=cv2.INTER_NEAREST)
        return cv2.applyColorMap(thermal_resized, cv2.COLORMAP_INFERNO)

    async def recv(self):
        """Receive thermal video frame"""
        try:
            # Frame rate control
            current_time = time.time()
            if current_time - self.last_frame_time < self.frame_interval:
                await asyncio.sleep(self.frame_interval - (current_time - self.last_frame_time))
            self.last_frame_time = time.time()
            
            pts, time_base = await self.next_timestamp()
            
            # Get thermal data
            thermal_data = self.get_thermal_data()
            
            if thermal_data is not None:
                # Update shared buffer for RGB camera
                self.thermal_buffer.update(thermal_data)
                
                # Create display frame
                frame = self.thermal_to_colormap(thermal_data)
                
                # Add overlays
                avg_temp = np.mean(thermal_data)
                min_temp = np.min(thermal_data)
                max_temp = np.max(thermal_data)
                
                cv2.putText(frame, "THERMAL", (10, 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Avg:{avg_temp:.1f}C", (10, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, f"Max:{max_temp:.1f}C", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.rectangle(frame, (0, 0), (self.width-1, self.height-1), (0, 165, 255), 2)
            else:
                frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                cv2.putText(frame, "NO THERMAL DATA", (10, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            # Convert and return
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_frame = VideoFrame.from_ndarray(frame, format="rgb24")
            video_frame.pts = pts
            video_frame.time_base = time_base
            return video_frame
            
        except Exception as e:
            logger.error(f"Thermal recv error: {e}")
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            video_frame = VideoFrame.from_ndarray(frame, format="rgb24")
            video_frame.pts = pts
            video_frame.time_base = time_base
            return video_frame

class CameraVideoStreamTrack(VideoStreamTrack):
    """RGB camera with integrated thermal analysis"""
    def __init__(self, thermal_buffer: ThermalDataBuffer):
        super().__init__()
        self.width = 320
        self.height = 240
        self.thermal_buffer = thermal_buffer
        self.frame_interval = 1.0 / 15  # 15 FPS
        self.last_frame_time = 0
        self.frame_counter = 0
        
        # Initialize components
        self.calibrator = DualCameraCalibrator()
        self.detector = CalibratedHumanDetector(self.calibrator)
        self._initialize_camera()
        
        # Load YOLO
        try:
            self.model = YOLO("yolov8n.pt")
            logger.info("✅ YOLO model loaded")
        except Exception as e:
            logger.error(f"YOLO load failed: {e}")
            raise
        
        # Cache
        self.last_results = None
        self.last_analyses = []
        
        logger.info("✅ RGB camera track initialized")

    def _initialize_camera(self):
        """Initialize camera"""
        for attempt in range(3):
            try:
                if hasattr(self, 'cap') and self.cap:
                    self.cap.release()
                    time.sleep(0.3)
                
                pipeline = (
                    f"libcamerasrc ! "
                    f"video/x-raw,width={self.width},height={self.height},format=YUY2,framerate=15/1 ! "
                    f"queue max-size-buffers=1 leaky=downstream ! "
                    f"videoconvert ! video/x-raw,format=BGR ! "
                    f"appsink sync=false drop=true max-buffers=1"
                )
                self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
                
                if self.cap.isOpened():
                    logger.info(f"RGB camera ready (attempt {attempt + 1})")
                    return
                time.sleep(0.3)
            except Exception as e:
                logger.error(f"RGB init attempt {attempt + 1} failed: {e}")
                time.sleep(0.3)
        raise IOError("Cannot initialize RGB camera")

    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'cap') and self.cap and self.cap.isOpened():
            self.cap.release()
            logger.info("RGB camera released")

    def __del__(self):
        self.cleanup()

    async def recv(self):
        """Receive RGB video frame with thermal analysis"""
        try:
            # Frame rate control
            current_time = time.time()
            if current_time - self.last_frame_time < self.frame_interval:
                await asyncio.sleep(self.frame_interval - (current_time - self.last_frame_time))
            self.last_frame_time = time.time()
            
            pts, time_base = await self.next_timestamp()
            
            # Read frame
            if not self.cap or not self.cap.isOpened():
                logger.warning("Camera disconnected, reinitializing...")
                self._initialize_camera()
            
            ret, frame = self.cap.read()
            if not ret:
                logger.error("Failed to read frame")
                frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                cv2.rectangle(frame, (50, 50), (100, 100), (0, 0, 255), -1)
            else:
                frame = self._process_frame(frame)
            
            # Convert and return
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_frame = VideoFrame.from_ndarray(frame, format="rgb24")
            video_frame.pts = pts
            video_frame.time_base = time_base
            return video_frame
            
        except Exception as e:
            logger.error(f"RGB recv error: {e}")
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            video_frame = VideoFrame.from_ndarray(frame, format="rgb24")
            video_frame.pts = pts
            video_frame.time_base = time_base
            return video_frame

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process frame with YOLO and thermal analysis"""
        self.frame_counter += 1
        
        # Run YOLO every 3 frames
        if self.frame_counter % 3 == 0:
            self.last_results = self.model(frame, verbose=False, imgsz=320, conf=0.5)[0]
            
            # Get thermal data (check if fresh)
            thermal_data = None
            thermal_age = self.thermal_buffer.get_age()
            if thermal_age < 1.0:  # Use thermal data if less than 1 second old
                thermal_data = self.thermal_buffer.get()
            
            # Analyze humans with thermal
            self.last_analyses = self.detector.analyze_humans(self.last_results, thermal_data)
        
        # Draw results
        return self._draw_results(frame)
    
    def _draw_results(self, frame: np.ndarray) -> np.ndarray:
        """Draw detection results on frame"""
        alive, dead, uncertain = 0, 0, 0
        
        # Draw human detections with thermal analysis
        for analysis in self.last_analyses:
            x1, y1, x2, y2 = analysis['rgb_bbox']
            status = analysis['status']
            conf = analysis['confidence']
            temp = analysis['max_temp']
            
            # Color based on status
            if "ALIVE" in status:
                color, alive = (0, 255, 0), alive + 1
            elif "DEAD" in status:
                color, dead = (0, 0, 255), dead + 1
            else:
                color, uncertain = (0, 255, 255), uncertain + 1
            
            # Draw bbox and info
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, status, (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            cv2.putText(frame, f"{temp}C", (x1, y1-15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            cv2.putText(frame, f"C:{conf:.2f}", (x1, y1-2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        # Draw other objects (chairs, tables)
        if self.last_results:
            for box in self.last_results.boxes.data:
                cls = int(box[5])
                if cls in [56, 60]:  # chair, table
                    x1, y1, x2, y2 = map(int, box[:4])
                    label = "chair" if cls == 56 else "table"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 128, 255), 2)
                    cv2.putText(frame, label, (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 128, 255), 1)
        
        # Status overlay
        thermal_age = self.thermal_buffer.get_age()
        calib_status = "Calibrated" if self.calibrator.homography_matrix is not None else "Scaling"
        thermal_status = f"Fresh" if thermal_age < 1.0 else f"Stale({thermal_age:.1f}s)"
        
        cv2.putText(frame, "RGB+THERMAL", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Alive:{alive} Dead:{dead} ?:{uncertain}", (10, 45), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"{calib_status}|{thermal_status}", (10, 65), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Border
        cv2.rectangle(frame, (0, 0), (self.width-1, self.height-1), (0, 255, 0), 2)
        
        return frame

async def run_webrtc(offer, pc, camera_track, thermal_track):
    """Handle WebRTC connection"""
    @pc.on("track")
    def on_track(track):
        logger.info(f"Received track: {track.kind}")

    @pc.on("iceconnectionstatechange")
    async def on_ice_state():
        logger.info(f"ICE state: {pc.iceConnectionState}")

    @pc.on("connectionstatechange")
    async def on_conn_state():
        logger.info(f"Connection state: {pc.connectionState}")

    try:
        if camera_track:
            pc.addTrack(camera_track)
            logger.info("✅ RGB track added")
        if thermal_track:
            pc.addTrack(thermal_track)
            logger.info("✅ Thermal track added")

        await pc.setRemoteDescription(RTCSessionDescription(sdp=offer, type="offer"))
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        logger.info("✅ WebRTC negotiation complete")
        return pc.localDescription
    except Exception as e:
        logger.error(f"WebRTC error: {e}")
        return None

def parse_ice_candidate(candidate_str: str) -> Dict:
    """Parse ICE candidate string"""
    if not candidate_str.startswith("candidate:"):
        raise ValueError("Invalid candidate format")
    
    parts = candidate_str[len("candidate:"):].split()
    if len(parts) < 8:
        raise ValueError("Insufficient candidate parts")
    
    result = {
        "foundation": parts[0],
        "component": int(parts[1]),
        "protocol": parts[2],
        "priority": int(parts[3]),
        "ip": parts[4],
        "port": int(parts[5]),
        "type": parts[7],
        "relatedAddress": None,
        "relatedPort": None
    }
    
    # Parse optional raddr/rport
    i = 8
    while i < len(parts):
        if parts[i] == "raddr" and i + 1 < len(parts):
            result["relatedAddress"] = parts[i+1]
            i += 2
        elif parts[i] == "rport" and i + 1 < len(parts):
            result["relatedPort"] = int(parts[i+1])
            i += 2
        else:
            i += 1
    
    return result

async def main():
    """Main application loop"""
    uri = "wss://websockettest-eggy.onrender.com"
    peer_id = "python-peer"
    
    print("="*50)
    print("THERMAL-RGB HUMAN DETECTION SYSTEM")
    print("="*50)
    
    # Create shared thermal buffer
    thermal_buffer = ThermalDataBuffer()
    
    while True:  # Reconnection loop
        websocket = None
        pc = None
        camera_track = None
        thermal_track = None
        
        try:
            pc = RTCPeerConnection()
            
            # Initialize thermal camera first
            try:
                thermal_track = ThermalVideoStreamTrack(thermal_buffer)
            except Exception as e:
                logger.error(f"❌ Thermal camera failed: {e}")
                await asyncio.sleep(5)
                continue
            
            # Initialize RGB camera
            try:
                camera_track = CameraVideoStreamTrack(thermal_buffer)
            except Exception as e:
                logger.error(f"❌ RGB camera failed: {e}")
                await asyncio.sleep(5)
                continue
            
            logger.info("✅ Both cameras initialized")
            
            # WebRTC handlers
            @pc.on("icecandidate")
            async def on_ice_candidate(candidate):
                if candidate and websocket and not websocket.closed:
                    msg = {
                        "SdpMid": candidate.sdpMid,
                        "SdpMLineIndex": candidate.sdpMLineIndex,
                        "Candidate": candidate.candidate
                    }
                    await websocket.send("CANDIDATE!" + json.dumps(msg))
            
            # Connect to signaling server
            async with websockets.connect(uri) as ws:
                websocket = ws
                logger.info(f"✅ Connected to {uri}")
                
                # Register
                await websocket.send(json.dumps({"type": "register", "peer_id": peer_id}))
                await websocket.send(json.dumps({"type": "peer_connected", "peer_id": peer_id}))
                logger.info("✅ Registered as python-peer")
                
                # Message loop
                while True:
                    try:
                        message = await websocket.recv()
                        message = message.decode('utf-8') if isinstance(message, bytes) else message
                        
                        if message.startswith("OFFER!"):
                            data = json.loads(message[6:])
                            logger.info("📨 Received offer")
                            answer = await run_webrtc(data["Sdp"], pc, camera_track, thermal_track)
                            if answer:
                                await websocket.send("ANSWER!" + json.dumps({
                                    "SessionType": answer.type.capitalize(),
                                    "Sdp": answer.sdp
                                }))
                                logger.info("✅ Sent answer")
                        
                        elif message.startswith("CANDIDATE!"):
                            data = json.loads(message[10:])
                            parsed = parse_ice_candidate(data["Candidate"])
                            candidate = RTCIceCandidate(
                                foundation=parsed["foundation"],
                                component=parsed["component"],
                                protocol=parsed["protocol"],
                                priority=parsed["priority"],
                                ip=parsed["ip"],
                                port=parsed["port"],
                                type=parsed["type"],
                                sdpMid=data["SdpMid"],
                                sdpMLineIndex=data["SdpMLineIndex"],
                                relatedAddress=parsed["relatedAddress"],
                                relatedPort=parsed["relatedPort"]
                            )
                            await pc.addIceCandidate(candidate)
                            logger.debug("Added ICE candidate")
                        
                        elif message == "bye":
                            logger.info("Received bye")
                            break
                    
                    except websockets.ConnectionClosedOK:
                        logger.info("WebSocket closed gracefully")
                        break
                    except websockets.ConnectionClosedError as e:
                        logger.error(f"WebSocket closed with error: {e}")
                        break
                    except Exception as e:
                        logger.error(f"Message handling error: {e}")
                        continue
        
        except Exception as e:
            logger.error(f"Main loop error: {e}")
        finally:
            # Cleanup
            if camera_track:
                camera_track.cleanup()
            if pc and pc.connectionState != "closed":
                await pc.close()
            if websocket:
                await websocket.close()
            
            logger.info("⏳ Reconnecting in 5 seconds...")
            await asyncio.sleep(5)

def run_calibration_interactive():
    """Run interactive calibration setup"""
    print("\n" + "="*50)
    print("CAMERA CALIBRATION SETUP")
    print("="*50)
    print("\nThis calibration will map RGB camera to thermal camera.")
    print("You'll need to place hot objects (warm mug, heated metal, etc.)")
    print("at visible positions in both camera views.\n")
    
    calibrator = DualCameraCalibrator()
    
    if calibrator.homography_matrix is not None:
        print("✅ Existing calibration found")
        recal = input("Recalibrate? (y/n): ").lower()
        if recal != 'y':
            print("Using existing calibration")
            return
    
    print("\nCalibration Mode:")
    print("1. Manual - Enter point coordinates directly")
    print("2. Skip - Use simple scaling (less accurate)")
    
    mode = input("Choose (1/2): ").strip()
    
    if mode == '1':
        print("\n📝 MANUAL CALIBRATION")
        print("Enter at least 4 corresponding point pairs")
        print("Format: x,y (e.g., 160,120)")
        print("\nTips:")
        print("- RGB range: x:0-319, y:0-239")
        print("- Thermal range: x:0-31, y:0-23")
        print("- Place objects at corners and center for best results\n")
        
        rgb_points = []
        thermal_points = []
        
        for i in range(4):
            print(f"\nPoint {i+1}:")
            try:
                rgb_input = input("  RGB (x,y): ").strip()
                thermal_input = input("  Thermal (x,y): ").strip()
                
                rgb_x, rgb_y = map(int, rgb_input.split(','))
                thermal_x, thermal_y = map(int, thermal_input.split(','))
                
                if (0 <= rgb_x < 320 and 0 <= rgb_y < 240 and 
                    0 <= thermal_x < 32 and 0 <= thermal_y < 24):
                    rgb_points.append([rgb_x, rgb_y])
                    thermal_points.append([thermal_x, thermal_y])
                    print(f"  ✅ Point {i+1} added")
                else:
                    print("  ❌ Out of range, skipping")
            except Exception as e:
                print(f"  ❌ Invalid input: {e}")
        
        if len(rgb_points) >= 4:
            if calibrator.save_calibration(rgb_points, thermal_points):
                print("\n✅ Calibration saved successfully!")
            else:
                print("\n❌ Calibration save failed")
        else:
            print("\n❌ Not enough valid points")
    else:
        print("\n⚠️  Using simple scaling (no calibration)")

if __name__ == "__main__":
    try:
        # Optional: Run calibration first
        print("\nDo you want to run calibration? (y/n/skip)")
        cal_choice = input("> ").lower()
        
        if cal_choice == 'y':
            run_calibration_interactive()
        
        print("\n🚀 Starting WebRTC streaming system...")
        print("Press Ctrl+C to stop\n")
        
        asyncio.run(main())
    
    except KeyboardInterrupt:
        print("\n\n⏹️  System stopped by user")
    except Exception as e:
        logger.exception(f"Fatal error: {e}")