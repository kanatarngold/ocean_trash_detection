import cv2
import time
import sys
import os

# Füge aktuellen Ordner zum Pfad hinzu
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from detector import TrashDetector
    from visualizer import Visualizer 
    from tracker import DetectionSmoother # RE-ENABLED
except ImportError:
    print("Fehler: konnte 'detector.py' nicht finden. Stelle sicher, dass alle Dateien im selben Ordner sind.")
    sys.exit(1)

# Konfiguration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_int8.tflite")
LABEL_PATH = os.path.join(BASE_DIR, "models", "labels.txt")

def main():
    print("🌊 Ocean Trash Detector Pi (vPRO + Stable + Overlap Fix) wird gestartet...")

    # 1. Detector laden
    try:
        detector = TrashDetector(model_path=MODEL_PATH, label_path=LABEL_PATH)
        visualizer = Visualizer()
        smoother = DetectionSmoother(history_size=5, smoothing_factor=0.3)
        print("✅ Modell & UI erfolgreich geladen.")
    except Exception as e:
        print(f"❌ Kritisches Fehler beim Laden des Modells: {e}")
        return

    # 2. Kamera Setup (Picamera2 - Native RPi Solution)
    try:
        from picamzero import Camera # Try simplest lib first if available
    except ImportError:
        pass

    try:
        # We try to import Picamera2. 
        # CAUTION: It must be installed via apt: sudo apt install python3-picamera2
        from picamera2 import Picamera2
        print("📷 Starte Picamera2 (Native Mode)...")
        
        # Configure camera
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(main={"size": (640, 480), "format": "BGR888"})
        picam2.configure(config)
        picam2.start()
        
        picam2.start()
        
        using_picamera = True
        print("✅ Picamera2 läuft!")
        
    except Exception as e:
        print(f"⚠️ Picamera2 Fehler: {e}")
        print("Versuche Fallback auf OpenCV (GStreamer)...")
        using_picamera = False
        
        # Fallback to GStreamer
        gstreamer_pipeline = (
            "libcamerasrc ! video/x-raw, width=640, height=480, framerate=30/1 ! "
            "videoconvert ! appsink"
        )
        cap = cv2.VideoCapture(gstreamer_pipeline, cv2.CAP_GSTREAMER)
        if not cap.isOpened():
             cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

    # 3. FPS Counter
    start_time = time.time()
    frame_count = 0
    fps = 0.0

    print("🚀 System bereit! Drücke 'q' zum Beenden.")
    
    # Store last detections to reuse them between inference frames
    last_detections = []
    
    # Create window explicitly to allow moving/resizing/closing
    window_name = 'Ocean Sentry AI (Pi Edition)'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 640, 480)

    while True:
        # Frame Capture
        if using_picamera:
            frame = picam2.capture_array()
        else:
            ret, frame = cap.read()
            if not ret:
                print("Fehler beim Lesen des Frames.")
                time.sleep(1)
                continue

        # Resize if needed (Picamera already gives 640x480)
        # frame = cv2.resize(frame, (640, 480))

        # Inference only every 3rd frame (Boosts UI FPS x3)
        if frame_count % 3 == 0:
            detections = detector.detect(frame, threshold=0.45, enhance=True)
            detections = smoother.update(detections)
            last_detections = detections
        else:
            detections = last_detections

        # 1. Upscale for HD UI (Text looks crisp even on large screens)
        # We process detection on small image (fast), but draw on big image (pretty)
        hd_frame = cv2.resize(frame, (1280, 960), interpolation=cv2.INTER_LINEAR)
        
        # Scale detections to match HD resolution (multiply coordinates by 2)
        hd_detections = []
        for det in detections:
            scaled_det = det.copy()
            scaled_det['box'] = [c * 2 for c in det['box']]
            hd_detections.append(scaled_det)

        # Visualisierung (Draw on HD Frame)
        visualizer.draw_tracker_overlay(hd_frame, hd_detections, fps)
        
        # Display
        cv2.imshow(window_name, hd_frame)
        
        # Check if window was closed by clicking 'X'
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

        # FPS Stats
        frame_count += 1
        elapsed = time.time() - start_time
        if elapsed > 1.0:
            fps = frame_count / elapsed
            print(f"FPS: {fps:.2f} | Objekte: {len(detections)}")
            frame_count = 0
            start_time = time.time()

        # Controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('f'):
            # Toggle Fullscreen
            prop = cv2.getWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN)
            if prop == cv2.WINDOW_FULLSCREEN:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            else:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    if using_picamera:
        try:
            picam2.stop()
        except:
            pass
    elif 'cap' in locals():
        cap.release()
    
    cv2.destroyAllWindows()
    print("👋 Programm beendet.")

if __name__ == '__main__':
    main()
