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

    # 2. Kamera starten
    print("📷 Starte Kamera...")
    cap = cv2.VideoCapture(0)
    
    # Auflösung für Pi V1.3
    cap.set(3, 640)
    cap.set(4, 480)

    if not cap.isOpened():
        print("❌ Fehler: Kamera konnte nicht geöffnet werden.")
        return

    print("🚀 System bereit! Drücke 'q' zum Beenden.")

    frame_count = 0
    start_time = time.time()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Fehler beim Lesen des Frames.")
            break

        # 3. Erkennung (Inference)
        # Threshold 0.25 -> "Scout Mode" sieht auch unsichere Objekte (grau markiert)
        detections = detector.detect(frame, threshold=0.25)
        
        # 4. Stabilisierung
        detections = smoother.update(detections)

        # FPS Stats
        frame_count += 1
        elapsed = time.time() - start_time
        if elapsed > 1.0:
            fps = frame_count / elapsed
            frame_count = 0
            start_time = time.time()

        # 5. Zeichnen (Professional Overlay)
        frame = visualizer.draw_tracker_overlay(frame, detections, fps)

        # Anzeigen
        cv2.imshow('Ocean Trash Detector', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
