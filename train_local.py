from ultralytics import YOLO
from roboflow import Roboflow
import torch

def train_local():
    # 1. Prüfen ob Mac GPU (MPS) verfügbar ist
    if torch.backends.mps.is_available():
        device = "mps"
        print("🚀 Apple Silicon GPU (M1/M2/M3) gefunden! Nutze Metal Performance Shaders.")
    else:
        device = "cpu"
        print("⚠️ Keine Apple GPU gefunden. Training läuft auf CPU (kann langsam sein).")

    # 2. Dataset laden
    rf = Roboflow(api_key="lkKTlQssHi1GPc8BYGsi")
    project = rf.workspace("research-7dj8h").project("ocean-waste")
    version = project.version(1)
    dataset = version.download("yolov8")

    # 3. Modell laden
    model = YOLO('yolov8s.pt')

    # 4. Training starten
    print(f"Starte Training auf {device}...")
    results = model.train(
        data=dataset.location + "/data.yaml",
        epochs=50,           # 50 Epochen für mehr Genauigkeit
        imgsz=640,
        device=device,       # Nutzt 'mps' auf Mac
        plots=True,
        name='ocean_trash_mac2' # Neuer Name zur Sicherheit
    )

    # 5. Exportieren
    print("Exportiere Modell...")
    model.export(format='tflite', int8=True)
    print("✅ Fertig! Modell liegt im Ordner 'runs/detect/ocean_trash_mac2/weights'")

if __name__ == '__main__':
    train_local()
