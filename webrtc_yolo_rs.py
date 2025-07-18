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
import os
import subprocess
from aiortc.contrib.media import MediaPlayer


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CameraVideoStreamTrack(VideoStreamTrack):
    def __init__(self):
        super().__init__()
        self.cap = None
        self.width = 640
        self.height = 480
        self._initialize_camera()
        
        # Load YOLOv8 model
        try:
            self.model = YOLO("yolov8n.pt")
            logger.info("YOLO model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise
            
        self.target_class_ids = {0: "person", 56: "chair", 60: "dining table"}
        
        # Motion tracking parameters
        self.motion_threshold = 7
        self.dead_time_seconds = 5
        self.human_tracker = {}
        self.prev_gray = None
        
        # Frame rate control
        self.last_frame_time = 0
        self.frame_interval = 1.0 / 30  # 30 FPS
        
        # Frame fallback
        self.last_good_frame = None
        self.consecutive_failures = 0
        self.max_consecutive_failures = 5
        
        logger.info("CameraVideoStreamTrack with YOLOv8 initialized.")

    def _find_available_cameras(self):
        """Find all available camera devices"""
        available_cameras = []
        
        # Check /dev/video* devices
        for i in range(10):  # Check first 10 video devices
            device_path = f"/dev/video{i}"
            if os.path.exists(device_path):
                available_cameras.append(i)
                logger.info(f"Found video device: {device_path}")
        
        # Try to get more info about video devices
        try:
            result = subprocess.run(['v4l2-ctl', '--list-devices'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                logger.info("Available video devices:")
                logger.info(result.stdout)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.warning("v4l2-ctl not available or timed out")
        
        return available_cameras

    def _try_realsense_pyrealsense2(self):
        """Try to initialize using pyrealsense2 library with improved error handling"""
        try:
            import pyrealsense2 as rs
            logger.info("pyrealsense2 library found, attempting RealSense initialization")
            
            # Configure depth and color streams
            pipeline = rs.pipeline()
            config = rs.config()
            
            # Get device product line for setting a supporting resolution
            pipeline_wrapper = rs.pipeline_wrapper(pipeline)
            pipeline_profile = config.resolve(pipeline_wrapper)
            device = pipeline_profile.get_device()
            device_product_line = str(device.get_info(rs.camera_info.product_line))
            
            logger.info(f"RealSense device detected: {device_product_line}")
            
            # Enable streams with more conservative settings
            config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, 30)
            
            # Start streaming
            pipeline.start(config)
            
            # Wait for a few frames to stabilize
            logger.info("Waiting for camera to stabilize...")
            for i in range(5):
                try:
                    frames = pipeline.wait_for_frames(timeout_ms=10000)  # 10 second timeout
                    if frames.get_color_frame():
                        logger.info(f"Stabilization frame {i+1}/5 captured")
                        break
                except Exception as e:
                    logger.warning(f"Stabilization frame {i+1} failed: {e}")
                    if i == 4:  # Last attempt
                        raise
            
            # Create a custom capture object that works with RealSense
            class RealSenseCapture:
                def __init__(self, pipeline):
                    self.pipeline = pipeline
                    self.is_opened = True
                    self.frame_count = 0
                    self.last_successful_frame = None
                
                def read(self):
                    try:
                        # Use a shorter timeout for regular frame capture
                        frames = self.pipeline.wait_for_frames(timeout_ms=1000)  # 1 second timeout
                        color_frame = frames.get_color_frame()
                        if not color_frame:
                            logger.warning("No color frame received")
                            return self._return_fallback_frame()
                        
                        # Convert to numpy array
                        frame = np.asanyarray(color_frame.get_data())
                        
                        # Store as fallback
                        self.last_successful_frame = frame.copy()
                        self.frame_count += 1
                        
                        return True, frame
                    except RuntimeError as e:
                        if "Frame didn't arrive within" in str(e):
                            logger.warning(f"Frame timeout: {e}")
                            return self._return_fallback_frame()
                        else:
                            logger.error(f"RealSense runtime error: {e}")
                            return self._return_fallback_frame()
                    except Exception as e:
                        logger.error(f"Error reading RealSense frame: {e}")
                        return self._return_fallback_frame()
                
                def _return_fallback_frame(self):
                    """Return the last good frame or a black frame if none available"""
                    if self.last_successful_frame is not None:
                        logger.debug("Returning last successful frame")
                        return True, self.last_successful_frame.copy()
                    else:
                        logger.debug("No fallback frame available, returning black frame")
                        black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                        return False, black_frame
                
                def isOpened(self):
                    return self.is_opened
                
                def get(self, prop):
                    if prop == cv2.CAP_PROP_FPS:
                        return 30
                    elif prop == cv2.CAP_PROP_FRAME_WIDTH:
                        return 640
                    elif prop == cv2.CAP_PROP_FRAME_HEIGHT:
                        return 480
                    return 0
                
                def set(self, prop, value):
                    # RealSense properties are set during initialization
                    return True
                
                def release(self):
                    try:
                        self.pipeline.stop()
                        self.is_opened = False
                        logger.info("RealSense pipeline stopped")
                    except Exception as e:
                        logger.error(f"Error stopping RealSense pipeline: {e}")
            
            self.cap = RealSenseCapture(pipeline)
            logger.info("RealSense camera initialized successfully using pyrealsense2")
            
            # Test initial frame capture
            ret, test_frame = self.cap.read()
            if ret:
                logger.info("Initial frame capture successful")
                return True
            else:
                logger.warning("Initial frame capture failed")
                return False
            
        except ImportError:
            logger.warning("pyrealsense2 library not found. Install with: pip install pyrealsense2")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize RealSense with pyrealsense2: {e}")
            return False

    def _initialize_camera(self):
        """Initialize camera with comprehensive retry logic for RealSense"""
        logger.info("Initializing camera for Intel RealSense D435i...")
        
        # First, try to find available cameras
        available_cameras = self._find_available_cameras()
        
        # Try pyrealsense2 first (best for RealSense cameras)
        if self._try_realsense_pyrealsense2():
            return
        
        # If pyrealsense2 fails, try OpenCV with different backends and indices
        backends_to_try = [
            cv2.CAP_V4L2,
            cv2.CAP_GSTREAMER,
            cv2.CAP_FFMPEG,
            cv2.CAP_ANY
        ]
        
        # Try different camera indices (RealSense might not be at index 0)
        camera_indices = available_cameras if available_cameras else [0, 1, 2, 4, 6, 8]
        
        for backend in backends_to_try:
            backend_name = self._get_backend_name(backend)
            logger.info(f"Trying backend: {backend_name}")
            
            for camera_index in camera_indices:
                try:
                    if self.cap:
                        self.cap.release()
                    
                    logger.info(f"Attempting camera index {camera_index} with {backend_name}")
                    self.cap = cv2.VideoCapture(camera_index, backend)
                    
                    if self.cap.isOpened():
                        # Test if we can actually read a frame
                        ret, test_frame = self.cap.read()
                        if ret and test_frame is not None:
                            logger.info(f"Successfully opened camera {camera_index} with {backend_name}")
                            
                            # Set properties
                            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                            self.cap.set(cv2.CAP_PROP_FPS, 30)
                            
                            # Verify the settings
                            actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                            actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
                            
                            logger.info(f"Camera configured: {actual_width}x{actual_height} @ {actual_fps} FPS")
                            return
                        else:
                            logger.warning(f"Camera {camera_index} opened but couldn't read frame")
                            self.cap.release()
                    else:
                        logger.warning(f"Failed to open camera {camera_index} with {backend_name}")
                        
                except Exception as e:
                    logger.warning(f"Error with camera {camera_index} and {backend_name}: {e}")
                    if self.cap:
                        self.cap.release()
                    continue
        
        # If all attempts fail, try GStreamer pipeline for RealSense
        self._try_gstreamer_pipeline()
        
        if not self.cap or not self.cap.isOpened():
            raise IOError("Cannot open Intel RealSense D435i camera. Please ensure:\n"
                         "1. Camera is connected via USB 3.0\n"
                         "2. Install Intel RealSense SDK 2.0\n"
                         "3. Install pyrealsense2: pip install pyrealsense2\n"
                         "4. Check camera permissions: sudo chmod 666 /dev/video*\n"
                         "5. Try different USB ports (preferably USB 3.0)\n"
                         "6. Check if other applications are using the camera")

    def _try_gstreamer_pipeline(self):
        """Try to initialize using GStreamer pipeline"""
        try:
            # GStreamer pipeline for RealSense (if available)
            gst_pipeline = (
                "v4l2src device=/dev/video0 ! "
                "video/x-raw, width=640, height=480, framerate=30/1 ! "
                "videoconvert ! appsink"
            )
            
            logger.info("Trying GStreamer pipeline for RealSense")
            self.cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
            
            if self.cap.isOpened():
                ret, test_frame = self.cap.read()
                if ret and test_frame is not None:
                    logger.info("GStreamer pipeline initialized successfully")
                    return
                else:
                    logger.warning("GStreamer pipeline opened but couldn't read frame")
                    self.cap.release()
            else:
                logger.warning("Failed to open GStreamer pipeline")
                
        except Exception as e:
            logger.error(f"GStreamer pipeline failed: {e}")
            if self.cap:
                self.cap.release()

    def _get_backend_name(self, backend):
        """Get human-readable backend name"""
        backend_names = {
            cv2.CAP_V4L2: "V4L2",
            cv2.CAP_GSTREAMER: "GStreamer",
            cv2.CAP_FFMPEG: "FFmpeg",
            cv2.CAP_ANY: "Any"
        }
        return backend_names.get(backend, f"Backend_{backend}")

    def cleanup(self):
        """Clean up camera resources"""
        if self.cap:
            try:
                if hasattr(self.cap, 'isOpened') and self.cap.isOpened():
                    self.cap.release()
                elif hasattr(self.cap, 'is_opened') and self.cap.is_opened:
                    self.cap.release()
                logger.info("Camera released.")
            except Exception as e:
                logger.error(f"Error releasing camera: {e}")
        self.cap = None

    def __del__(self):
        self.cleanup()
        
    async def recv(self):
        """Receive and process video frames with improved error handling"""
        try:
            # Frame rate control
            current_time = time.time()
            if current_time - self.last_frame_time < self.frame_interval:
                await asyncio.sleep(self.frame_interval - (current_time - self.last_frame_time))
            
            self.last_frame_time = time.time()
            
            pts, time_base = await self.next_timestamp()
            timestamp = time.time()

            # Check if camera is still open
            if not self.cap or not (hasattr(self.cap, 'isOpened') and self.cap.isOpened()) and not (hasattr(self.cap, 'is_opened') and self.cap.is_opened):
                logger.warning("Camera not available, attempting to reinitialize...")
                try:
                    self._initialize_camera()
                except Exception as e:
                    logger.error(f"Failed to reinitialize camera: {e}")

            ret, frame = self.cap.read()
            if not ret:
                self.consecutive_failures += 1
                logger.warning(f"Failed to read frame from camera (failure #{self.consecutive_failures})")
                
                # If we have too many consecutive failures, try to reinitialize
                if self.consecutive_failures >= self.max_consecutive_failures:
                    logger.warning("Too many consecutive failures, attempting camera reinitialization...")
                    try:
                        self._initialize_camera()
                        self.consecutive_failures = 0
                    except Exception as e:
                        logger.error(f"Camera reinitialization failed: {e}")
                
                # Return black frame or last good frame
                if self.last_good_frame is not None:
                    frame = self.last_good_frame.copy()
                    # Add "CAMERA ERROR" text overlay
                    cv2.putText(frame, "CAMERA ERROR - USING LAST FRAME", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                    cv2.putText(frame, "CAMERA ERROR - NO SIGNAL", 
                               (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                # Successful frame capture
                self.consecutive_failures = 0
                self.last_good_frame = frame.copy()
                frame = self._process_frame(frame, timestamp)

            # Convert BGR to RGB for WebRTC
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_frame = VideoFrame.from_ndarray(frame, format="rgb24")
            video_frame.pts = pts
            video_frame.time_base = time_base

            return video_frame
            
        except Exception as e:
            logger.error(f"Error in recv(): {e}")
            # Return a black frame with error message
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            cv2.putText(frame, "PROCESSING ERROR", 
                       (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_frame = VideoFrame.from_ndarray(frame, format="rgb24")
            video_frame.pts = pts
            video_frame.time_base = time_base
            return video_frame

    def _process_frame(self, frame, timestamp):
        """Process frame with YOLO detection and motion tracking"""
        # Convert to grayscale for motion tracking
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # Run YOLOv8 detection
        results = self.model(frame, verbose=False)[0]
        humans_detected = 0
        possibly_dead = 0
        new_tracker = {}

        for box in results.boxes.data:
            x1, y1, x2, y2, conf, cls = box
            class_id = int(cls)
            if class_id not in self.target_class_ids:
                continue

            label_name = self.target_class_ids[class_id]
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            color = (255, 255, 255)

            if class_id == 0:  # Person: motion tracking
                humans_detected += 1
                roi_now = gray[y1:y2, x1:x2]
                person_id = f"{x1}-{y1}-{x2}-{y2}"

                moved = False
                if self.prev_gray is not None:
                    roi_prev = self.prev_gray[y1:y2, x1:x2]
                    moved = self.is_moving(roi_now, roi_prev)

                last_move_time = self.human_tracker.get(person_id, 0)
                if moved:
                    last_move_time = timestamp

                new_tracker[person_id] = last_move_time
                time_inactive = timestamp - last_move_time
                is_dead = time_inactive > self.dead_time_seconds

                label = "Dead" if is_dead else ("Alive" if moved else "No Motion")
                color = (0, 0, 255) if is_dead else ((0, 255, 255) if not moved else (0, 255, 0))

                if is_dead:
                    possibly_dead += 1
            else:
                label = label_name
                color = (0, 128, 255)

            # Draw detection and label on frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Display stats on frame
        cv2.putText(frame, f"Humans: {humans_detected}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Possibly Dead: {possibly_dead}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        fps = 30  # Default FPS for display
        if hasattr(self.cap, 'get'):
            fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        cv2.putText(frame, f"FPS: {fps:.2f}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        self.prev_gray = gray.copy()
        self.human_tracker = new_tracker
        
        return frame

    def is_moving(self, current_roi, previous_roi):
        if current_roi.shape != previous_roi.shape:
            return False
        diff = cv2.absdiff(current_roi, previous_roi)
        _, motion_mask = cv2.threshold(diff, self.motion_threshold, 255, cv2.THRESH_BINARY)
        return np.sum(motion_mask > 0) > 50


class DummyAudioTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self, sample_rate=48000, amplitude=0.1):
        super().__init__()
        self.sample_rate = sample_rate
        self.amplitude = amplitude
        self._counter = 0

    async def recv(self):
        await asyncio.sleep(0.02)  # Simulate 20ms audio chunk
        samples = np.zeros(960, dtype=np.int16)  # 960 samples for 20ms at 48kHz
        t = np.arange(self._counter, self._counter + 960) / self.sample_rate
        samples = (self.amplitude * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
        self._counter += 960
        return samples.tobytes()  # Convert to bytes for WebRTC


# Replace the run() function with this updated version:
async def run(offer, pc, camera_track):
    """
    Handles the WebRTC offer/answer exchange and track reception.
    """
    @pc.on("track")
    def on_track(track):
        """
        Callback when a remote track is received.
        """
        logger.info(f"Python received track {track.kind}")

    @pc.on("iceconnectionstatechange")
    async def on_ice_connection_state_change():
        logger.info(f"ICE connection state changed to: {pc.iceConnectionState}")
        if pc.iceConnectionState == "failed":
            logger.warning("ICE connection failed.")
        elif pc.iceConnectionState == "disconnected":
            logger.warning("ICE connection disconnected.")
        elif pc.iceConnectionState == "closed":
            logger.info("ICE connection closed.")

    @pc.on("connectionstatechange")
    async def on_connection_state_change():
        logger.info(f"Connection state changed to: {pc.connectionState}")
        if pc.connectionState == "failed":
            logger.error("WebRTC connection failed")
        elif pc.connectionState == "closed":
            logger.info("WebRTC connection closed")

    try:
        # Set the remote description (the offer received from browser)
        await pc.setRemoteDescription(RTCSessionDescription(sdp=offer, type="offer"))
        logger.info("Set remote description successfully")
        
        # Add the camera track after setting remote description
        if camera_track:
            pc.addTrack(camera_track)
            logger.info("Camera track added to peer connection")
        
        # Create and set the local description (the answer to browser's offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        logger.info("Created and set local description (answer)")

        # Return the local description (answer) to be sent back via signaling
        return pc.localDescription
        
    except Exception as e:
        logger.error(f"Error in run() function: {e}")
        raise

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
        
        try:
            # Create new peer connection for each attempt
            pc = RTCPeerConnection()
            
            # Initialize camera track
            try:
                camera_track = CameraVideoStreamTrack()
                logger.info("Camera track initialized")
            except Exception as e:
                logger.error(f"Failed to initialize camera track: {e}")
                continue

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

                while True:
                    try:
                        message = await websocket.recv()
                        message = message.decode('utf-8')
                        logger.info(f"Raw message received: {message}")
                        
                        if message.startswith("OFFER!"):
                            json_str = message[len("OFFER!"):]
                            try:
                                data = json.loads(json_str)
                                logger.info("Received offer from Unity")
                                offer_sdp = data["Sdp"]
                                
                                answer = await run(offer_sdp, pc, camera_track)
                                answer_message_data = {
                                    "SessionType": answer.type.capitalize(),
                                    "Sdp": answer.sdp
                                }
                                full_answer_message = "ANSWER!" + json.dumps(answer_message_data)
                                await websocket.send(full_answer_message)
                                logger.info("Sent answer to Unity")
                                
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
                                logger.info("Received ICE candidate from Unity")
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
                                logger.info("Added ICE candidate from Unity")
                                
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

