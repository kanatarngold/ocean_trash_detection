from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import cv2
import asyncio
import json
import numpy as np
from detector import TrashDetector, EnsembleDetector
from pathlib import Path
import threading
import time
import math
from collections import OrderedDict

app = FastAPI()

# Initialize detector
# --- Shared Classes (Copied from desktop_main.py for now) ---
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

class ThreadedCamera:
    """Turbo Mode: Reads frames in a separate thread to prevent lag"""
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640) # Lower res for streaming
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        
        if not self.capture.isOpened():
            self.status = False
            self.frame = None
        else:
            self.status = True
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

# Initialize detector (Ensemble if available)
try:
    # 1. Load Primary
    model_path = "backend/models/model.tflite"
    if not Path(model_path).exists(): model_path = "models/model.tflite"
    
    # Resolve Label Path
    label_path = "backend/models/labels.txt"
    if not Path(label_path).exists(): label_path = "models/labels.txt"
        
    primary_detector = TrashDetector(model_path=model_path, label_path=label_path)
    
    # 2. Load Secondary
    secondary_model_path = "backend/models/model_old.tflite"
    if not Path(secondary_model_path).exists(): secondary_model_path = "models/model_old.tflite"
    
    if Path(secondary_model_path).exists():
        # Check for separate labels
        secondary_label_path = "backend/models/labels_old.txt"
        if not Path(secondary_label_path).exists(): secondary_label_path = "models/labels_old.txt"
        if not Path(secondary_label_path).exists(): secondary_label_path = "models/labels.txt"

        secondary_detector = TrashDetector(model_path=secondary_model_path, label_path=secondary_label_path)
        detector = EnsembleDetector([primary_detector, secondary_detector])
        print("🚀 Web Backend: Ensemble Mode Activated")
    else:
        detector = primary_detector
        print("ℹ️  Web Backend: Single Model Mode")
        
except Exception as e:
    print(f"❌ Web Backend Error: {e}")
    detector = TrashDetector() # Fallback

tracker = CentroidTracker()

# Mount static files (Frontend)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def get():
    with open("static/index.html") as f:
        return HTMLResponse(f.read())

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # Open webcam (0) or video file
    cap = ThreadedCamera(0).start()
    
    if not cap.isOpened():
        print("Warning: Could not open webcam. Using dummy stream.")
        # Create a dummy video capture object or just handle it in the loop
        cap = None

    try:
        while True:
            if cap:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Failed to read frame from camera. Switching to dummy stream.")
                    cap.release()
                    cap = None
            if cap:
                ret, frame = cap.read()
                if not ret or frame is None:
                    # print("Error: Failed to read frame from camera. Switching to dummy stream.")
                    # cap.stop()
                    # cap = None
                    time.sleep(0.01)
                    continue
            else:
                # Generate dummy frame (black background)
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "No Camera Found", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                # Simulate "trash" for demo purposes
                cv2.circle(frame, (320, 240), 30, (0, 0, 255), -1) # Red circle
                
            # Detect objects (only if real frame, or skip for dummy)
            if cap:
                detections = detector.detect(frame)
            else:
                detections = [] # Or fake detections for demo
            
            # Update Tracker
            rects = []
            for d in detections:
                rects.append(d['box'])
            objects = tracker.update(rects)
            
            # Match IDs
            for d in detections:
                box = d['box']
                cX = int((box[0] + box[2]) / 2.0)
                cY = int((box[1] + box[3]) / 2.0)
                
                best_id = -1
                min_dist = 99999
                for obj_id, centroid in objects.items():
                    dist = math.sqrt((cX - centroid[0])**2 + (cY - centroid[1])**2)
                    if dist < 50:
                        if dist < min_dist:
                            min_dist = dist
                            best_id = obj_id
                d['id'] = best_id if best_id != -1 else "?"

            # Draw on frame
            # frame_annotated = detector.draw_detections(frame.copy(), detections)
            frame_annotated = frame.copy()
            for d in detections:
                box = d['box']
                label = f"ID:{d['id']} {d['label']} {d['score']:.2f}"
                color = (0, 255, 0) if d['label'] == 'Plastic' else (0, 165, 255)
                cv2.rectangle(frame_annotated, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
                cv2.putText(frame_annotated, label, (int(box[0]), int(box[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Encode frame to JPEG
            _, buffer = cv2.imencode('.jpg', frame_annotated)
            jpg_as_text = buffer.tobytes()
            
            # Send frame bytes
            await websocket.send_bytes(jpg_as_text)
            
            # Optionally send stats as text (interleaved or separate channel)
            # For simplicity, we just stream video now. 
            # To add stats, we could send a JSON text message every N frames.
            stats = {
                "count": len(detections),
                "objects": [d['label'] for d in detections]
            }
            await websocket.send_text(json.dumps(stats))
            
            # Control FPS
            await asyncio.sleep(0.03) # ~30 FPS
            
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if cap: cap.stop()
