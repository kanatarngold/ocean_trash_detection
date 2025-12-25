import cv2
import numpy as np

class Visualizer:
    def __init__(self):
        # --- 1. FARBKONZEPT (Industrial) ---
        # Format: BGR (OpenCV uses Blue-Green-Red)
        self.colors = {
            "Background": (26, 26, 26),     # #1A1A1A (Dark Neutral Grey)
            "Accent":     (0, 215, 244),    # #F4D700 (Industrial Yellow)
            "Text":       (220, 220, 220),  # #DCDCDC (Light Grey)
            "Success":    (50, 205, 50),    # #32CD32 (Lime Green)
            "Alarm":      (50, 50, 220),    # #DC3232 (Crimson Red)
            "Grid":       (60, 60, 60)      # #3C3C3C (Subtle Grid Lines)
        }
        
        # --- 2. TYPOGRAFIE ---
        # Neutral Sans-Serif, gut lesbar
        self.font = cv2.FONT_HERSHEY_SIMPLEX 
        self.font_scale_ui = 0.5
        self.font_scale_label = 0.6
        self.thick_ui = 1
        self.thick_label = 2

    def draw_tracker_overlay(self, canvas, detections, fps):
        """
        Zeichnet das komplette 'Industrial Dashboard' auf den 1920x1080 Canvas.
        Erwartet: canvas (1920x1080), detections (skaliert), fps (float)
        """
        h_canvas, w_canvas, _ = canvas.shape # Sollte 1080x1920 sein
        
        # --- STEP 1: HINTERGRUND & RAHMEN ---
        # Falls Canvas nicht schwarz ist, füllen wir die Ränder auf
        # (Wird normalerweise vom Inference-Skript gemacht, aber sicher ist sicher)
        # Hier zeichnen wir nur den Rahmen um den Kamera-Bereich
        
        # Kamera-Bereich: 1280x960, zentriert
        cam_w, cam_h = 1280, 960
        x_off = (w_canvas - cam_w) // 2
        y_off = (h_canvas - cam_h) // 2
        
        # Dezenter Industrierahmen (2px Gelb)
        cv2.rectangle(canvas, 
                      (x_off - 2, y_off - 2), 
                      (x_off + cam_w + 2, y_off + cam_h + 2), 
                      self.colors["Accent"], 2)
        
        # --- STEP 2: DETECTIONS (Im Kamera-Bereich) ---
        for det in detections:
            self.draw_single_detection(canvas, det)

        # --- STEP 3: STATUSLEISTE (Footer) ---
        # Fixe Höhe am unteren Rand: 60px
        bar_h = 60
        bar_y = h_canvas - bar_h
        
        # Leiste Hintergrund (Dunkelgrau)
        cv2.rectangle(canvas, (0, bar_y), (w_canvas, h_canvas), self.colors["Background"], -1)
        # Trennlinie oben (Gelb)
        cv2.line(canvas, (0, bar_y), (w_canvas, bar_y), self.colors["Accent"], 2)

        # status metrics helper
        y_text = bar_y + 40
        
        # A) System ID (Links)
        cv2.putText(canvas, "SYSTEM: OCEAN SENTRY MK-1", (40, y_text), 
                    self.font, 0.7, self.colors["Accent"], 2, cv2.LINE_AA)

        # B) Live Status (Mitte Links)
        cv2.putText(canvas, "STATUS: ONLINE", (500, y_text), 
                    self.font, 0.6, self.colors["Success"], 1, cv2.LINE_AA)
                    
        # C) AI Metrics (Mitte Rechts)
        obj_count = len(detections)
        ai_color = self.colors["Text"]
        if obj_count > 0:
            ai_color = self.colors["Accent"] # Gelb wenn Objekte da sind
            
        cv2.putText(canvas, f"DETECTED OBJECTS: {obj_count}", (1100, y_text),
                    self.font, 0.6, ai_color, 1, cv2.LINE_AA)

        # D) FPS Counter (Rechts)
        fps_color = self.colors["Text"]
        if fps < 5.0: fps_color = self.colors["Alarm"] # Rot bei Lag
        
        cv2.putText(canvas, f"FPS: {fps:.1f}", (1700, y_text),
                    self.font, 0.6, fps_color, 1, cv2.LINE_AA)

        return canvas

    def draw_single_detection(self, canvas, det):
        """Minimalistische Boxen für Industrial Look"""
        left, top, right, bottom = det['box']
        label = det['label']
        score = det['score']
        
        # Farbe basierend auf Label (Hier halten wir es einheitlich "Industrial"?)
        # Oder nutzen wir Akzentfarbe für alles? 
        # User sagte: "industrielles Gelb-auf-Schwarz als Akzent"
        # -> Wir machen alle Boxen Gelb für Uniformität, oder subtil unterschiedlich.
        # Lass uns Klassen-spezifisch bleiben, aber gedämpft.
        
        color = self.colors["Accent"] # Default Yellow
        if label == "Glass": color = (200, 200, 200) # Whiteish
        elif label == "Metal": color = (70, 70, 180) # Rusty red/blue?
        elif label == "Plastic": color = (0, 165, 255) # Orange
        
        # Scout Mode Override (Low conf)
        if score < 0.45:
            color = (100, 100, 100) # Dark Grey
            label = "?"

        # 1. Feiner Rahmen (2px)
        cv2.rectangle(canvas, (left, top), (right, bottom), color, 2)
        
        # 2. Label Tag (Pill Style, but rectangular for Industrial)
        label_text = f"{label} {score:.0%}"
        (text_w, text_h), _ = cv2.getTextSize(label_text, self.font, 0.5, 1)
        
        # Background Rect (Black)
        cv2.rectangle(canvas, 
                      (left, top - 25), 
                      (left + text_w + 10, top), 
                      (10, 10, 10), -1) # Almost black fill
        # Border for Label (Same color as box)
        cv2.rectangle(canvas, 
                      (left, top - 25), 
                      (left + text_w + 10, top), 
                      color, 1)

        # Text (Yellow/Color)
        cv2.putText(canvas, label_text, (left + 5, top - 8), 
                    self.font, 0.5, color, 1, cv2.LINE_AA)
