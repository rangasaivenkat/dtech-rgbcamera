import sys
print(sys.executable)
import asyncio
import logging
import json
import cv2
import numpy as np
from aiortc import RTCIceCandidate, RTCPeerConnection, RTCSessionDescription
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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DualCameraCalibrator:
    def __init__(self, rgb_resolution=(320, 240), thermal_resolution=(32, 24)):
        self.rgb_resolution = rgb_resolution
        self.thermal_resolution = thermal_resolution
        self.homography_matrix = None
        self.calibration_file = "camera_calibration.json"
        self.load_calibration()
        
    def load_calibration(self):
        try:
            if os.path.exists(self.calibration_file):
                with open(self.calibration_file, 'r') as f:
                    data = json.load(f)
                self.homography_matrix = np.array(data['homography_matrix'])
                logger.info(f"Loaded calibration - Error: {data.get('reprojection_error', 'N/A'):.2f}px")
                return True
        except Exception as e:
            logger.error(f"Failed to load calibration: {e}")
        return False
    
    def map_rgb_bbox_to_thermal(self, rgb_bbox):
        if self.homography_matrix is None:
            x1, y1, x2, y2 = rgb_bbox
            thermal_x1 = max(0, min(31, int(x1 * 32 / 320)))
            thermal_y1 = max(0, min(23, int(y1 * 24 / 240)))
            thermal_x2 = max(0, min(31, int(x2 * 32 / 320)))
            thermal_y2 = max(0, min(23, int(y2 * 24 / 240)))
            if thermal_x2 > thermal_x1 and thermal_y2 > thermal_y1:
                return (thermal_x1, thermal_y1, thermal_x2, thermal_y2)
            return None
        
        x1, y1, x2, y2 = rgb_bbox
        corners = np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]).reshape(-1, 1, 2)
        mapped_corners = cv2.perspectiveTransform(corners, self.homography_matrix)
        mapped_corners = mapped_corners.reshape(-1, 2)
        
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
        self.TEMP_THRESHOLDS = {
            'human_min': 33.0, 'human_normal': 36.0, 'human_fever': 38.0,
            'ambient_diff_min': 4.0, 'dead_threshold': 30.0,
        }
        
    def analyze_human_regions(self, yolo_results, thermal_data, ambient_temp=None):
        if ambient_temp is None:
            ambient_temp = np.percentile(thermal_data, 25)
        
        human_analyses = []
        for box in yolo_results.boxes.data:
            x1, y1, x2, y2, conf, cls = box
            if int(cls) == 0:  # Person
                rgb_bbox = (int(x1), int(y1), int(x2), int(y2))
                analysis = self._analyze_single_human(rgb_bbox, thermal_data, ambient_temp)
                if analysis:
                    analysis['rgb_bbox'] = rgb_bbox
                    analysis['yolo_confidence'] = float(conf)
                    human_analyses.append(analysis)
        return human_analyses
    
    def _analyze_single_human(self, rgb_bbox, thermal_data, ambient_temp):
        thermal_bbox = self.calibrator.map_rgb_bbox_to_thermal(rgb_bbox)
        if thermal_bbox is None:
            return None
        
        tx1, ty1, tx2, ty2 = thermal_bbox
        thermal_region = thermal_data[ty1:ty2, tx1:tx2]
        if thermal_region.size == 0:
            return None
        
        max_temp = np.max(thermal_region)
        mean_temp = np.mean(thermal_region)
        hot_pixels = np.sum(thermal_region >= self.TEMP_THRESHOLDS['human_min'])
        hot_pixel_ratio = hot_pixels / thermal_region.size
        temp_above_ambient = max_temp - ambient_temp
        
        status, confidence = self._determine_human_status(max_temp, temp_above_ambient, hot_pixel_ratio)
        
        return {
            'status': status, 'confidence': confidence,
            'max_temp': round(max_temp, 1), 'mean_temp': round(mean_temp, 1),
            'temp_above_ambient': round(temp_above_ambient, 1),
            'hot_pixel_ratio': round(hot_pixel_ratio, 2),
            'thermal_bbox': thermal_bbox, 'ambient_temp': round(ambient_temp, 1)
        }
    
    def _determine_human_status(self, max_temp, temp_above_ambient, hot_pixel_ratio):
        temp_conf = 1.0 if max_temp >= 36.0 else (0.7 if max_temp >= 33.0 else (0.3 if max_temp >= 30.0 else 0.0))
        ambient_conf = 1.0 if temp_above_ambient >= 4.0 else (0.5 if temp_above_ambient >= 2.0 else 0.0)
        pixel_conf = min(1.0, hot_pixel_ratio * 5)
        confidence = (temp_conf * 0.5 + ambient_conf * 0.3 + pixel_conf * 0.2)
        
        if confidence >= 0.8:
            if max_temp >= 38.0:
                return "ALIVE - Fever", confidence
            elif max_temp >= 36.0:
                return "ALIVE - Normal", confidence
            else:
                return "ALIVE - Cool", confidence
        elif confidence >= 0.6:
            return "LIKELY ALIVE", confidence
        elif confidence >= 0.4:
            return "UNCERTAIN", confidence
        else:
            return ("LIKELY DEAD" if max_temp <= 30.0 else "UNKNOWN"), confidence

class ThermalVideoStreamTrack(VideoStreamTrack):
    def __init__(self):
        super().__init__()
        self.width = 320
        self.height = 240
        self.thermal_available = False
        self.last_valid_data = None
        self.consecutive_errors = 0
        self.max_errors = 15
        self.last_frame_time = 0
        self.frame_interval = 1.0 / 8
        self._initialize_thermal()
        logger.info("ThermalVideoStreamTrack initialized")

    def _initialize_thermal(self):
        max_retries = 5
        for attempt in range(max_retries):
            try:
                logger.info(f"Initializing thermal camera (attempt {attempt + 1}/{max_retries})")
                self.i2c = busio.I2C(board.SCL, board.SDA, frequency=100000)  # Lower frequency
                time.sleep(0.3)
                self.mlx = adafruit_mlx90640.MLX90640(self.i2c)
                logger.info(f"MLX90640 serial: {[hex(i) for i in self.mlx.serial_number]}")
                self.mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_8_HZ
                self.thermal_frame = np.zeros((24 * 32,), dtype=float)
                
                # Test read
                for test_attempt in range(3):
                    try:
                        self.mlx.getFrame(self.thermal_frame)
                        break
                    except:
                        if test_attempt < 2:
                            time.sleep(0.2)
                        else:
                            raise
                
                self.thermal_available = True
                logger.info("Thermal camera initialized successfully")
                return
            except Exception as e:
                logger.error(f"Thermal init attempt {attempt + 1} failed: {e}")
                time.sleep(1.0)
        
        logger.warning("Thermal camera unavailable - running in RGB-only mode")
        self.thermal_available = False

    def thermal_to_colormap(self, thermal_data):
        vmin, vmax = 20.0, 50.0
        scaled = np.clip((thermal_data - vmin) / (vmax - vmin), 0, 1)
        thermal_8bit = (scaled * 255).astype(np.uint8)
        thermal_resized = cv2.resize(thermal_8bit, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        return cv2.applyColorMap(thermal_resized, cv2.COLORMAP_INFERNO)

    def get_thermal_data(self):
        if not self.thermal_available:
            return self.last_valid_data
        
        for attempt in range(5):
            try:
                self.mlx.getFrame(self.thermal_frame)
                thermal_data = np.array(self.thermal_frame).reshape((24, 32))
                
                if np.isnan(thermal_data).any() or np.isinf(thermal_data).any():
                    raise ValueError("Invalid thermal data")
                
                self.last_valid_data = thermal_data
                self.consecutive_errors = 0
                return thermal_data
            except Exception as e:
                if attempt < 4:
                    time.sleep(0.05 * (attempt + 1))
                else:
                    self.consecutive_errors += 1
                    if self.consecutive_errors >= self.max_errors:
                        logger.warning("Too many thermal errors, attempting reinit")
                        self.consecutive_errors = 0
                        self._initialize_thermal()
                    return self.last_valid_data

    async def recv(self):
        try:
            current_time = time.time()
            if current_time - self.last_frame_time < self.frame_interval:
                await asyncio.sleep(self.frame_interval - (current_time - self.last_frame_time))
            self.last_frame_time = time.time()
            
            pts, time_base = await self.next_timestamp()

            try:
                thermal_data = self.get_thermal_data()
                if thermal_data is not None:
                    frame = self.thermal_to_colormap(thermal_data)
                    avg_temp = np.mean(thermal_data)
                    min_temp = np.min(thermal_data)
                    max_temp = np.max(thermal_data)
                    
                    status = "THERMAL - OK" if self.consecutive_errors == 0 else f"THERMAL - ERR:{self.consecutive_errors}"
                    color = (255, 255, 255) if self.consecutive_errors == 0 else (0, 165, 255)
                    
                    cv2.putText(frame, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    cv2.putText(frame, f"Avg: {avg_temp:.1f}C", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(frame, f"Min: {min_temp:.1f}C", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    cv2.putText(frame, f"Max: {max_temp:.1f}C", (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    
                    border_color = (0, 165, 255) if self.consecutive_errors == 0 else (0, 0, 255)
                    cv2.rectangle(frame, (0, 0), (frame.shape[1]-1, frame.shape[0]-1), border_color, 3)
                else:
                    raise Exception("No thermal data")
            except:
                frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                cv2.putText(frame, "THERMAL UNAVAILABLE", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "RGB-only mode", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.rectangle(frame, (0, 0), (frame.shape[1]-1, frame.shape[0]-1), (0, 0, 255), 3)

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
    def __init__(self, thermal_track=None):
        super().__init__()
        self.cap = None
        self.width = 320
        self.height = 240
        self.thermal_track = thermal_track
        self._initialize_camera()
        
        self.calibrator = DualCameraCalibrator()
        self.human_detector = CalibratedHumanDetector(self.calibrator)
        
        try:
            self.model = YOLO("yolov8n.pt")
            logger.info("YOLO model loaded")
        except Exception as e:
            logger.error(f"Failed to load YOLO: {e}")
            raise
            
        self.target_class_ids = {0: "person", 56: "chair", 60: "dining table"}
        self.last_frame_time = 0
        self.frame_interval = 1.0 / 15
        self.frame_counter = 0
        self.last_results = None
        self.last_human_analyses = []
        logger.info("CameraVideoStreamTrack initialized")

    def _initialize_camera(self):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if self.cap:
                    self.cap.release()
                    time.sleep(0.5)
                    
                pipeline = (
                    f"libcamerasrc ! "
                    f"video/x-raw, width={self.width}, height={self.height}, format=YUY2, framerate=15/1 ! "
                    f"queue max-size-buffers=1 leaky=downstream ! "
                    f"videoconvert ! video/x-raw, format=BGR ! "
                    f"appsink sync=false drop=true max-buffers=1"
                )
                self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
                if self.cap.isOpened():
                    logger.info(f"Camera initialized (attempt {attempt + 1})")
                    return
                time.sleep(0.5)
            except Exception as e:
                logger.error(f"Camera init error (attempt {attempt + 1}): {e}")
                time.sleep(0.5)
        raise IOError("Cannot open camera")

    def cleanup(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
            time.sleep(0.5)
            logger.info("Camera released")
        self.cap = None

    def __del__(self):
        self.cleanup()
        
    async def recv(self):
        try:
            current_time = time.time()
            if current_time - self.last_frame_time < self.frame_interval:
                await asyncio.sleep(self.frame_interval - (current_time - self.last_frame_time))
            self.last_frame_time = current_time
            
            pts, time_base = await self.next_timestamp()

            if not self.cap or not self.cap.isOpened():
                logger.warning("Camera reinitializing")
                self._initialize_camera()

            ret, frame = self.cap.read()
            if not ret:
                logger.error("Failed to read frame")
                frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                cv2.rectangle(frame, (50, 50), (100, 100), (0, 0, 255), -1)
            else:
                frame = self._process_frame(frame)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_frame = VideoFrame.from_ndarray(frame, format="rgb24")
            video_frame.pts = pts
            video_frame.time_base = time_base
            return video_frame
        except Exception as e:
            logger.error(f"recv error: {e}")
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            video_frame = VideoFrame.from_ndarray(frame, format="rgb24")
            video_frame.pts = pts
            video_frame.time_base = time_base
            return video_frame

    def _process_frame(self, frame):
        self.frame_counter += 1
        human_analyses = self.last_human_analyses

        if self.frame_counter % 3 == 0:
            results = self.model(frame, verbose=False, imgsz=320, conf=0.5)[0]
            self.last_results = results
            
            thermal_data = None
            if self.thermal_track and self.thermal_track.thermal_available:
                thermal_data = self.thermal_track.get_thermal_data()
            
            if thermal_data is not None:
                human_analyses = self.human_detector.analyze_human_regions(results, thermal_data)
            else:
                human_analyses = []
                for box in results.boxes.data:
                    if int(box[5]) == 0:
                        human_analyses.append({
                            'rgb_bbox': (int(box[0]), int(box[1]), int(box[2]), int(box[3])),
                            'status': 'DETECTED - No Thermal',
                            'confidence': float(box[4]),
                            'max_temp': 0.0,
                            'yolo_confidence': float(box[4])
                        })
            self.last_human_analyses = human_analyses
        
        return self._draw_results(frame, human_analyses)
    
    def _draw_results(self, frame, human_analyses):
        alive_count = dead_count = uncertain_count = 0
        
        for analysis in human_analyses:
            x1, y1, x2, y2 = analysis['rgb_bbox']
            status = analysis['status']
            confidence = analysis['confidence']
            max_temp = analysis.get('max_temp', 0.0)
            
            if "ALIVE" in status:
                color, alive_count = (0, 255, 0), alive_count + 1
            elif "DEAD" in status:
                color, dead_count = (0, 0, 255), dead_count + 1
            else:
                color, uncertain_count = (0, 255, 255), uncertain_count + 1
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            temp_text = f"{max_temp:.1f}C" if max_temp > 0 else "No Temp"
            
            cv2.putText(frame, status, (x1, y1 - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.putText(frame, temp_text, (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.putText(frame, f"Conf: {confidence:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        if self.last_results:
            for box in self.last_results.boxes.data:
                class_id = int(box[5])
                if class_id in self.target_class_ids and class_id != 0:
                    x1, y1, x2, y2 = map(int, box[:4])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 128, 255), 2)
                    cv2.putText(frame, self.target_class_ids[class_id], (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 128, 255), 2)
        
        mode = "RGB + THERMAL" if (self.thermal_track and self.thermal_track.thermal_available) else "RGB ONLY"
        cv2.putText(frame, mode, (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Alive: {alive_count}", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Dead: {dead_count}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Uncertain: {uncertain_count}", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        calib = "Calibrated" if self.calibrator.homography_matrix is not None else "Not Calibrated"
        cv2.putText(frame, f"Status: {calib}", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.rectangle(frame, (0, 0), (frame.shape[1]-1, frame.shape[0]-1), (0, 255, 0), 3)
        
        return frame

def parse_ice_candidate_string(candidate_str):
    if not candidate_str.startswith("candidate:"):
        raise ValueError(f"Invalid ICE candidate: {candidate_str}")
    
    parts = candidate_str[len("candidate:"):].split()
    if len(parts) < 8:
        raise ValueError(f"Invalid ICE candidate format")

    related_address = related_port = None
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
        "foundation": parts[0], "component": int(parts[1]), "protocol": parts[2],
        "priority": int(parts[3]), "ip": parts[4], "port": int(parts[5]),
        "type": parts[7], "relatedAddress": related_address, "relatedPort": related_port
    }

async def run(offer, pc, camera_track, thermal_track):
    @pc.on("track")
    def on_track(track):
        logger.info(f"Received track: {track.kind}")

    @pc.on("iceconnectionstatechange")
    async def on_ice_connection_state_change():
        logger.info(f"ICE state: {pc.iceConnectionState}")

    @pc.on("connectionstatechange")
    async def on_connection_state_change():
        logger.info(f"Connection state: {pc.connectionState}")

    try:
        if camera_track:
            pc.addTrack(camera_track)
            logger.info("RGB track added (Track 0)")
        if thermal_track:
            pc.addTrack(thermal_track)
            logger.info("Thermal track added (Track 1)")

        if not offer or "m=video" not in offer:
            logger.error("Invalid offer SDP")
            return None

        await pc.setRemoteDescription(RTCSessionDescription(sdp=offer, type="offer"))
        logger.info("Set remote description")
        
        answer = await pc.createAnswer()
        if not answer:
            logger.error("Failed to create answer")
            return None
        
        await pc.setLocalDescription(answer)
        logger.info("Set local description")
        return pc.localDescription
    except Exception as e:
        logger.error(f"Error in run: {e}")
        return None

async def main():
    uri = "wss://websockettest-eggy.onrender.com"
    peer_id = "python-peer"
    
    print("="*60)
    print("THERMAL-RGB HUMAN DETECTION SYSTEM")
    print("="*60)
    print("\nSystem will handle thermal errors gracefully")
    print("If thermal camera fails, RGB detection will continue\n")
    
    while True:
        websocket = pc = camera_track = thermal_track = None
        
        try:
            pc = RTCPeerConnection()
            
            try:
                print("Initializing thermal camera...")
                thermal_track = ThermalVideoStreamTrack()
                if thermal_track.thermal_available:
                    print("OK - Thermal camera ready")
                else:
                    print("WARNING - Thermal unavailable, using RGB-only mode")
            except Exception as e:
                logger.error(f"Thermal init failed: {e}")
                print("WARNING - Thermal failed, using RGB-only mode")
                
            try:
                print("Initializing RGB camera...")
                camera_track = CameraVideoStreamTrack(thermal_track)
                print("OK - RGB camera ready")
            except Exception as e:
                logger.error(f"RGB init failed: {e}")
                print("ERROR - RGB camera failed, retrying in 5s")
                await asyncio.sleep(5)
                continue

            print("\nStarting WebRTC streaming...\n")

            @pc.on("datachannel")
            def on_datachannel(channel):
                logger.info(f"Data channel '{channel.label}' received")

            @pc.on("icecandidate")
            async def on_ice_candidate(candidate):
                if candidate and websocket and not websocket.closed:
                    candidate_data = {
                        "SdpMid": candidate.sdpMid,
                        "SdpMLineIndex": candidate.sdpMLineIndex,
                        "Candidate": candidate.candidate
                    }
                    try:
                        await websocket.send("CANDIDATE!" + json.dumps(candidate_data))
                        logger.info("Sent ICE candidate")
                    except Exception as e:
                        logger.error(f"Failed to send ICE: {e}")

            async with websockets.connect(uri) as ws_conn:
                websocket = ws_conn
                logger.info(f"Connected to signaling server")
                
                await websocket.send(json.dumps({"type": "register", "peer_id": peer_id}))
                await websocket.send(json.dumps({"type": "peer_connected", "peer_id": peer_id}))
                logger.info("Registered with server")

                while True:
                    try:
                        message = await websocket.recv()
                        message = message.decode('utf-8') if isinstance(message, bytes) else message
                        
                        if message.startswith("OFFER!"):
                            try:
                                data = json.loads(message[6:])
                                logger.info("Received offer")
                                answer = await run(data["Sdp"], pc, camera_track, thermal_track)
                                if answer:
                                    answer_data = {"SessionType": answer.type.capitalize(), "Sdp": answer.sdp}
                                    await websocket.send("ANSWER!" + json.dumps(answer_data))
                                    logger.info("Sent answer")
                            except Exception as e:
                                logger.error(f"Error processing offer: {e}")
                                
                        elif message.startswith("CANDIDATE!"):
                            try:
                                data = json.loads(message[10:])
                                logger.info("Received ICE candidate")
                                parsed = parse_ice_candidate_string(data["Candidate"])
                                ice_candidate = RTCIceCandidate(
                                    foundation=parsed["foundation"], component=parsed["component"],
                                    protocol=parsed["protocol"], priority=parsed["priority"],
                                    ip=parsed["ip"], port=parsed["port"], type=parsed["type"],
                                    sdpMid=data["SdpMid"], sdpMLineIndex=data["SdpMLineIndex"],
                                    relatedAddress=parsed["relatedAddress"], relatedPort=parsed["relatedPort"]
                                )
                                await pc.addIceCandidate(ice_candidate)
                                logger.info("Added ICE candidate")
                            except Exception as e:
                                logger.error(f"Error processing candidate: {e}")
                                
                        elif message == "bye":
                            logger.info("Received bye")
                            return
                            
                    except websockets.ConnectionClosed:
                        logger.info("WebSocket closed")
                        break
                    except Exception as e:
                        logger.exception("WebSocket error")
                        break

        except websockets.exceptions.ConnectionClosedError as e:
            logger.error(f"Failed to connect to signaling server: {e}")
        except Exception as e:
            logger.exception("Error in main function")
        finally:
            if camera_track:
                camera_track.cleanup()
            if pc and pc.connectionState != "closed":
                await pc.close()
            if websocket and hasattr(websocket, 'close'):
                await websocket.close()
            logger.info("Connections closed, reconnecting in 5 seconds...")
            await asyncio.sleep(5)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nScript terminated by user")
    except Exception as e:
        logger.exception("Unhandled error")
