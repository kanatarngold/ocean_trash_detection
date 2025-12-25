# 🌊 Ocean Trash Detector - Team Guide

Hier ist eine kurze Anleitung, wie du das Projekt auf deinem Laptop zum Laufen bekommst.

## 1. Voraussetzungen
Du brauchst **Python 3.9** oder neuer.

## 2. Installation
Öffne ein Terminal (oder CMD auf Windows) in diesem Ordner (`backend`) und installiere die Abhängigkeiten:

```bash
pip install -r requirements.txt
```

*Falls es Probleme mit TensorFlow auf Mac gibt (M1/M2/M3), nutze:*
```bash
pip install tensorflow-macos opencv-python numpy
```

## 3. Starten
Um die Erkennung mit deiner Webcam zu starten, führe einfach dieses Skript aus:

**Mac / Linux:**
```bash
python3 test_mac.py
```

**Windows:**
```bash
python test_mac.py
```

## 4. Features steuern
Das System hat folgende Features aktiviert:
*   **Sticky Boxes:** Stabilisiert das Bild (weniger Zittern).
*   **Identity Shield:** Ignoriert Plastik, das dein Gesicht verdeckt.
*   **Crystal Vision:** Bessert den Kontrast bei trübem Wasser auf.
*   **Scout Mode:** Zeigt unsichere Objekte als graue "?"-Boxen an.

Viel Spaß beim Testen! 🚀
