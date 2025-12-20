import cv2
import numpy as np
import argparse
import time
from datetime import datetime
from pathlib import Path
from pathlib import Path
from pathlib import Path
from detector import TrashDetector, EnsembleDetector
import csv
import subprocess
import threading
import random
import math
from collections import OrderedDict

# --- Helper Functions ---

class CentroidTracker:
    def __init__(self, maxDisappeared=50):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects

        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            
            D = []
            for i in range(len(objectCentroids)):
                row = []
                for j in range(len(inputCentroids)):
                    dist = math.sqrt((objectCentroids[i][0] - inputCentroids[j][0])**2 + 
                                   (objectCentroids[i][1] - inputCentroids[j][1])**2)
                    row.append(dist)
                D.append(row)
            D = np.array(D)
            
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue

                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0
                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])

        return self.objects

class DetectionSmoother:
    def __init__(self, history_size=3, min_hits=2):
        self.history = [] # List of detection lists
        self.history_size = history_size
        self.min_hits = min_hits
        
    def update(self, current_detections):
        self.history.append(current_detections)
        if len(self.history) > self.history_size:
            self.history.pop(0)
            
        if len(self.history) < self.history_size:
            return current_detections # Not enough history yet
            
        # Box Smoothing (Exponential Moving Average)
        # We need to track boxes by ID. Since we don't have IDs here yet (Tracker runs after),
        # we will just do a simple IoU matching to find the "same" box in the previous frame.
        
        smoothed_detections = []
        prev_detections = self.history[-2] if len(self.history) >= 2 else []
        
        for curr in current_detections:
            # Find best match in previous frame
            best_match = None
            max_iou = 0.0
            
            curr_box = curr['box']
            cx1, cy1, cx2, cy2 = curr_box
            curr_area = (cx2 - cx1) * (cy2 - cy1)
            
            for prev in prev_detections:
                px1, py1, px2, py2 = prev['box']
                
                # Calculate IoU
                xx1 = max(cx1, px1)
                yy1 = max(cy1, py1)
                xx2 = min(cx2, px2)
                yy2 = min(cy2, py2)
                
                w = max(0, xx2 - xx1)
                h = max(0, yy2 - yy1)
                inter_area = w * h
                
                prev_area = (px2 - px1) * (py2 - py1)
                union_area = curr_area + prev_area - inter_area
                
                if union_area > 0:
                    iou = inter_area / union_area
                    if iou > 0.5 and iou > max_iou:
                        max_iou = iou
                        best_match = prev
            
            if best_match:
                # Smooth the box
                alpha = 0.3 # Smoothing factor (lower = smoother but more lag)
                px1, py1, px2, py2 = best_match['box']
                
                nx1 = int(px1 * (1-alpha) + cx1 * alpha)
                ny1 = int(py1 * (1-alpha) + cy1 * alpha)
                nx2 = int(px2 * (1-alpha) + cx2 * alpha)
                ny2 = int(py2 * (1-alpha) + cy2 * alpha)
                
                curr['box'] = (nx1, ny1, nx2, ny2)
                
            smoothed_detections.append(curr)
            
        return smoothed_detections

def _ensure_dir(p: Path):
    """Create directory if it doesn't exist"""
    p.mkdir(parents=True, exist_ok=True)

def _init_writer(out_path: Path, frame_shape, fps: float):
    """Initialize video writer for MP4 recording"""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    h, w = frame_shape[:2]
    return cv2.VideoWriter(str(out_path), fourcc, fps if fps > 0 else 30.0, (w, h))

def _draw_hud(frame, fps: float, threshold: float, cam_label: str, recording: bool, show_help: bool, camera_id: int = 0):
    """Draw professional HUD overlay with status and help"""
    overlay = frame.copy()
    # Top bar
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 50), (32, 32, 32), -1)
    # Bottom help bar
    if show_help:
        cv2.rectangle(overlay, (0, frame.shape[0]-80), (frame.shape[1], frame.shape[0]), (32, 32, 32), -1)
    alpha = 0.5
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Top bar text
    title = 'Ocean Trash Detector'
    status = f"FPS: {fps:4.1f}  thr={threshold:.2f}  Mode={cam_label}  CamID={camera_id}"
    rec_txt = 'REC' if recording else ''
    cv2.putText(frame, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, status, (320, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 200), 2)
    if recording:
        cv2.putText(frame, rec_txt, (frame.shape[1]-100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Help at bottom
    if show_help:
        help_lines = [
            "Shortcuts: q=Quit | f=Fullscreen | r=Record | s=Snapshot",
            "m=Switch Model | c=Switch Camera | h=Toggle Help"
        ]
        y0 = frame.shape[0]-60
        for i, line in enumerate(help_lines):
            cv2.putText(frame, line, (10, y0 + i*24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

class MissionLogger:
    def __init__(self):
        # Save to Desktop for easy access
        self.log_dir = Path.home() / "Desktop" / "OceanTrashLogs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = self.log_dir / f"mission_{timestamp}.csv"
        
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Frame_ID", "Label", "Score", "Box"])
            
    def log(self, frame_id, detections):
        if not detections:
            return
            
        timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            for d in detections:
                writer.writerow([timestamp, frame_id, d['label'], f"{d['score']:.2f}", str(d['box'])])

class ThreadedCamera:
    """Turbo Mode: Reads frames in a separate thread to prevent lag"""
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        
        # Check if camera opened successfully
        if not self.capture.isOpened():
            self.status = False
            self.frame = None
        else:
            self.status = True
            # Read first frame
            self.status, self.frame = self.capture.read()
            
        self.stopped = False
        self.lock = threading.Lock()
        
    def start(self):
        if self.status:
            t = threading.Thread(target=self.update, args=())
            t.daemon = True
            t.start()
        return self
        
    def update(self):
        while not self.stopped:
            if self.capture.isOpened():
                status, frame = self.capture.read()
                with self.lock:
                    self.status = status
                    self.frame = frame
            else:
                time.sleep(0.1)
                
    def read(self):
        with self.lock:
            return self.status, self.frame
            
    def stop(self):
        self.stopped = True
        if self.capture.isOpened():
            self.capture.release()
            
    def isOpened(self):
        return self.capture.isOpened()

    def switch_camera(self, new_src):
        """Switch to a different camera source at runtime"""
        with self.lock:
            if self.capture.isOpened():
                self.capture.release()
            
            self.capture = cv2.VideoCapture(new_src)
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
            
            if not self.capture.isOpened():
                self.status = False
                self.frame = None
                print(f"❌ Failed to open camera {new_src}")
                return False
            else:
                self.status = True
                self.status, self.frame = self.capture.read()
                print(f"✅ Switched to camera {new_src}")
                return True

class ThreadedDetector:
    """Async Detector: Runs AI in a separate thread so video stays smooth"""
    def __init__(self, detector, threshold=0.5):
        self.detector = detector
        self.threshold = threshold
        self.current_frame = None
        self.latest_detections = []
        self.stopped = False
        self.lock = threading.Lock()
        self.new_frame_event = threading.Event()
        
    def set_detector(self, detector):
        with self.lock:
            self.detector = detector
        
    def start(self):
        t = threading.Thread(target=self.loop, args=())
        t.daemon = True
        t.start()
        return self
        
    def update_frame(self, frame):
        with self.lock:
            self.current_frame = frame.copy()
        self.new_frame_event.set()
        
    def get_detections(self):
        with self.lock:
            return self.latest_detections
            
    def loop(self):
        while not self.stopped:
            # Wait for a new frame
            if self.new_frame_event.wait(timeout=0.1):
                self.new_frame_event.clear()
                
                with self.lock:
                    if self.current_frame is None: continue
                    frame_to_process = self.current_frame
                
                # Run heavy detection (Tiling)
                # Reverting Tiling as per user request (too slow/laggy)
                results = self.detector.detect(frame_to_process, self.threshold)
                
                with self.lock:
                    self.latest_detections = results
            
    def stop(self):
        self.stopped = True

# --- Main Application ---

def main(camera_id: int, threshold: float, fullscreen: bool = False):
    print("=== Ocean Trash Detector ===")
    print("Press 'q' to Quit")
    
    # Initialize Detector
    available_modes = [] # List of (name, detector_instance)
    
    try:
        # 1. Load Primary Model (TACO)
        model_path = "backend/models/model.tflite"
        if not Path(model_path).exists():
            # Fallback for running from different dir
            model_path = "models/model.tflite"
            
        # Resolve Label Path
        label_path = "backend/models/labels.txt"
        if not Path(label_path).exists():
            label_path = "models/labels.txt"
            
        primary_detector = TrashDetector(model_path=model_path, label_path=label_path)
        print(f"✓ Loaded Primary Model: {model_path}")
        available_modes.append(("Model A (TACO)", primary_detector))
        
        # 2. Load Secondary Model (Old/Backup)
        secondary_model_path = "backend/models/model_old.tflite"
        if not Path(secondary_model_path).exists():
             secondary_model_path = "models/model_old.tflite"
             
        if Path(secondary_model_path).exists():
            # Check for separate labels
            secondary_label_path = "backend/models/labels_old.txt"
            if not Path(secondary_label_path).exists():
                secondary_label_path = "models/labels_old.txt"
            
            if not Path(secondary_label_path).exists():
                print("⚠️  Warning: 'labels_old.txt' not found. Using default labels for Model B.")
                secondary_label_path = "models/labels.txt" # Fallback
                
            secondary_detector = TrashDetector(model_path=secondary_model_path, label_path=secondary_label_path)
            print(f"✓ Loaded Secondary Model: {secondary_model_path} (Labels: {secondary_label_path})")
            available_modes.append(("Model B (Old)", secondary_detector))
            
            # Create Ensemble
            ensemble_detector = EnsembleDetector([primary_detector, secondary_detector])
            available_modes.append(("Ensemble (A+B)", ensemble_detector))
            print("🚀 Ensemble Mode Activated (2 Models)")
            
            # Default to Ensemble
            current_mode_idx = 2 
        else:
            print("ℹ️  Single Model Mode (Add 'model_old.tflite' to enable Ensemble)")
            current_mode_idx = 0

        # Set initial detector
        current_detector_name = available_modes[current_mode_idx][0]
        detector = available_modes[current_mode_idx][1]

    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # Initialize Camera (Turbo Mode)
    cap = ThreadedCamera(camera_id).start()
    
    if not cap.isOpened():
        print(f"❌ Could not open camera {camera_id}. Using dummy stream.")
        cap = None
    else:
        print(f"✓ Camera {camera_id} started (Turbo Mode 🚀)")

    # Window Setup
    window_name = 'Ocean Trash Detector'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    if fullscreen:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.resizeWindow(window_name, 960, 540)

    # State
    last_t = time.time()
    fps_ma = 0.0
    recording = False
    writer = None
    show_help = True
    
    # New Features
    logger = MissionLogger()
    frame_count = 0
    last_alert_time = 0.0
    
    # Initialize Tracker
    tracker = CentroidTracker()
    
    # Initialize Async Detector
    async_detector = ThreadedDetector(detector, threshold).start()
    
    # Initialize Smoother
    smoother = DetectionSmoother()
    
    # Auto-Capture Directory
    capture_dir = Path.home() / "Desktop" / "OceanTrashDataset"
    _ensure_dir(capture_dir)
    
    print("✓ Tracker, Auto-Capture & Async AI initialized")

    try:
        while True:
            if cap:
                ret, frame = cap.read()
                if not ret or frame is None:
                    # If frame is None, wait a bit and try again (camera might be initializing)
                    time.sleep(0.01)
                    continue
            else:
                # Dummy frame
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "No Camera Found", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                # Simulate trash
                cv2.circle(frame, (320, 240), 30, (0, 0, 255), -1)
                time.sleep(0.03)

            # FPS calculation
            now = time.time()
            dt = now - last_t
            last_t = now
            inst_fps = 1.0 / max(dt, 1e-6)
            fps_ma = fps_ma * 0.9 + inst_fps * 0.1 if fps_ma > 0 else inst_fps

            # Detection
            # Send frame to async detector
            async_detector.update_frame(frame)
            # Get latest results (instant)
            raw_detections = async_detector.get_detections()
            
            # Apply Smoothing
            # We need to initialize smoother outside loop first!
            # Assuming 'smoother' is initialized in main
            detections = smoother.update(raw_detections)
            
            # Update Tracker
            rects = []
            for d in detections:
                rects.append(d['box'])
            
            objects = tracker.update(rects)
            
            # Match IDs to detections (Simple heuristic)
            for d in detections:
                box = d['box']
                cX = int((box[0] + box[2]) / 2.0)
                cY = int((box[1] + box[3]) / 2.0)
                
                # Find closest object ID
                best_id = -1
                min_dist = 99999
                for obj_id, centroid in objects.items():
                    dist = math.sqrt((cX - centroid[0])**2 + (cY - centroid[1])**2)
                    if dist < 50: # Threshold to associate
                        if dist < min_dist:
                            min_dist = dist
                            best_id = obj_id
                
                d['id'] = best_id if best_id != -1 else "?"
                
                # Auto-Capture Logic
                if d['score'] > 0.80 and best_id != -1:
                    # Check if we already captured this ID recently (simplified: just capture randomly for now to avoid spam)
                    if random.random() < 0.1: # 10% chance per frame to capture high confidence object
                        obj_img = frame[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                        if obj_img.size > 0:
                            filename = capture_dir / f"{d['label']}_{best_id}_{int(time.time())}.jpg"
                            cv2.imwrite(str(filename), obj_img)
                            # Draw "CAPTURED" indicator
                            cv2.putText(frame, "CAPTURED", (int(box[0]), int(box[1])-20), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Trash Pile Alert=None)
            
            # Log detections
            frame_count += 1
            logger.log(frame_count, detections)
            
            # Annotation
            # frame = detector.draw_detections(frame, detections) # Custom draw for IDs
            
            for d in detections:
                box = d['box']
                label = f"ID:{d['id']} {d['label']} {d['score']:.2f}"
                color = (0, 255, 0) if d['label'] == 'Plastic' else (0, 165, 255)
                
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
                cv2.putText(frame, label, (int(box[0]), int(box[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # 4. Draw HUDTRASH PILE DETECTION ---
            if len(detections) > 5:
                # Calculate union bounding box
                min_x = min(d['box'][0] for d in detections)
                min_y = min(d['box'][1] for d in detections)
                max_x = max(d['box'][2] for d in detections)
                max_y = max(d['box'][3] for d in detections)
                
                # Draw Big Red Box
                cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (0, 0, 255), 4)
                
                # Draw Warning Label
                label = "WARNING: TRASH PILE DETECTED"
                label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
                
                # Ensure label is visible (not off-screen)
                label_y = max(min_y - 10, label_size[1] + 10)
                
                # Draw label background
                cv2.rectangle(frame, (min_x, label_y - label_size[1] - 10), (min_x + label_size[0] + 20, label_y + base_line + 10), (0, 0, 255), cv2.FILLED)
                # Draw text
                cv2.putText(frame, label, (min_x + 10, label_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

                # Sound Alert (Max once every 2 seconds)
                now = time.time()
                if now - last_alert_time > 2.0:
                    try:
                        subprocess.Popen(['afplay', '/System/Library/Sounds/Ping.aiff'])
                        last_alert_time = now
                    except Exception:
                        pass
            
            # HUD
            _draw_hud(frame, fps_ma, threshold, current_detector_name, recording, show_help, camera_id)

            # Recording
            if recording and writer:
                writer.write(frame)

            # Display
            cv2.imshow(window_name, frame)

            # Input Handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('f'):
                fullscreen = not fullscreen
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
            elif key == ord('r'):
                if recording:
                    recording = False
                    if writer:
                        writer.release()
                        writer = None
                        print("⏹️  Recording stopped")
                else:
                    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                    out = Path('recordings') / f'demo_{ts}.mp4'
                    _ensure_dir(out.parent)
                    writer = _init_writer(out, frame.shape, fps_ma)
                    recording = True
                    print(f"⏺️  Recording started: {out}")
            elif key == ord('s'):
                ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                out = Path('screenshots') / f'snap_{ts}.png'
                _ensure_dir(out.parent)
                cv2.imwrite(str(out), frame)
                print(f"📸 Screenshot saved: {out}")
            elif key == ord('m'):
                # Switch Model
                if len(available_modes) > 1:
                    current_mode_idx = (current_mode_idx + 1) % len(available_modes)
                    current_detector_name = available_modes[current_mode_idx][0]
                    new_detector = available_modes[current_mode_idx][1]
                    async_detector.set_detector(new_detector)
                    print(f"🔄 Switched to: {current_detector_name}")
            elif key == ord('c'):
                # Switch Camera
                # Cycle 0 -> 1 -> 2 -> 0
                next_cam = (camera_id + 1) % 3 
                if cap.switch_camera(next_cam):
                    camera_id = next_cam
                else:
                    # If failed (e.g. cam 2 doesn't exist), try 0
                    if cap.switch_camera(0):
                        camera_id = 0
            elif key == ord('h'):
                show_help = not show_help

    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        if cap: cap.stop()
        async_detector.stop()
        if writer: writer.release()
        cv2.destroyAllWindows()
        print("✓ Program finished")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0, help="Camera ID")
    parser.add_argument("--threshold", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--fullscreen", action='store_true', help="Start in fullscreen")
    args = parser.parse_args()

    main(args.camera, args.threshold, args.fullscreen)
