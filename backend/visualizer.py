import cv2
import numpy as np

class Visualizer:
    def __init__(self):
        # --- FARBKONZEPT (Strict Industrial) ---
        # BGR Werte
        self.colors = {
            "Background": (30, 30, 30),     # #1E1E1E Anthrazit (Basis-Hintergrund)
            "Surface":    (40, 40, 40),     # #282828 Leichte Variation für Footer
            "Accent":     (21, 195, 229),   # #E5C315 Caterpillar-Gelb (Markierungen)
            "TextLight":  (230, 230, 230),  # #E6E6E6 Fast Weiß (Haupttext)
            "TextDim":    (160, 160, 160),  # #A0A0A0 Grau (Metadaten)
            "Alert":      (50, 50, 200),    # #C83232 Rot (Nur für Status-Dots, nicht Hintergrund)
            "Green":      (80, 180, 80)     # #50B450 Grün (Status OK)
        }
        
        # --- TYPOGRAFIE ---
        # Inter/Roboto Ersatz -> Hershey Sans-Serif
        self.font_main = cv2.FONT_HERSHEY_SIMPLEX
        self.font_tech = cv2.FONT_HERSHEY_PLAIN
        
        # Layout Metriken
        self.footer_h = 42

    def draw_tracker_overlay(self, canvas, detections, fps):
        """
        Zeichnet das finale Layout.
        WICHTIG: Erwartet, dass 'canvas' bereits mit Background-Farbe (30,30,30) gefüllt ist!
        """
        h_cam, w_cam, _ = canvas.shape
        
        # 1. Kamera-Container Design
        # Wir zeichnen den 1px Rahmen um den zentralen Bereich (1280x960)
        # Offset berechnen (Kamera ist 1280x960)
        cam_w, cam_h = 1280, 960
        x_off = (w_cam - cam_w) // 2
        y_off = (h_cam - cam_h) // 2
        
        # Der Bereich außerhalb der Kamera ist bereits "Background", 
        # wir zeichnen nur den feinen Rahmen.
        # Exakt 1px.
        cv2.rectangle(canvas, 
                      (x_off - 1, y_off - 1), 
                      (x_off + cam_w, y_off + cam_h), 
                      self.colors["Accent"], 1)

        # 2. Detections
        for det in detections:
            self.draw_single_detection(canvas, det)

        # 3. Statusleiste (Footer)
        # Separierter Bereich unten
        y_foot = h_cam - self.footer_h
        
        # Hintergrund Footer (Surface)
        cv2.rectangle(canvas, (0, y_foot), (w_cam, h_cam), self.colors["Surface"], -1)
        # Haarlinie oben (Accent) - nur ganz dezent
        # cv2.line(canvas, (0, y_foot), (w_cam, y_foot), self.colors["Accent"], 1) 
        # User wollte: "Gelb wird nur für Akzente verwendet... keine zufälligen Linien".
        # Eine Linie als Trennung ist okay, aber vielleicht zu dominant? 
        # Wir lassen sie weg für maximalen Minimalismus oder machen sie dunkelgrau.
        cv2.line(canvas, (0, y_foot), (w_cam, y_foot), (60, 60, 60), 1)

        # --- INHALT FOOTER ---
        mid_y = y_foot + 28
        
        # A. System-Name (Links)
        # Großbuchstaben, Clean
        cv2.putText(canvas, "OCEAN SENTRY", (30, mid_y), 
                    self.font_main, 0.6, self.colors["TextLight"], 1, cv2.LINE_AA)
        
        # B. Status Indikatoren (Mitte / Rechts-orientiert)
        
        # Dot Helper
        def draw_dot(cx, color):
            cv2.circle(canvas, (cx, mid_y - 5), 4, color, -1)

        # Live Feed Status
        draw_dot(300, self.colors["Green"])
        cv2.putText(canvas, "SIGNAL ACTIVE", (315, mid_y),
                    self.font_tech, 1.0, self.colors["TextDim"], 1, cv2.LINE_AA)

        # AI Status
        ai_active = len(detections) > 0
        ai_col = self.colors["Green"] if ai_active else self.colors["TextDim"]
        draw_dot(500, ai_col)
        cv2.putText(canvas, "AI RÜSTUNG", (515, mid_y),
                    self.font_tech, 1.0, self.colors["TextDim"], 1, cv2.LINE_AA)

        # C. FPS (Ganz Rechts)
        fps_text = f"{fps:.1f} FPS"
        (w_txt, _), _ = cv2.getTextSize(fps_text, self.font_tech, 1.2, 1)
        x_fps = w_cam - 30 - w_txt
        
        cv2.putText(canvas, fps_text, (x_fps, mid_y), 
                    self.font_tech, 1.2, self.colors["TextDim"], 1, cv2.LINE_AA)

        return canvas

    def draw_single_detection(self, canvas, det):
        """Minimalistisch: 1px Box, Text 'schwebt' oder cleanes Label."""
        l, t, r, b = det['box']
        label = det['label']
        score = det['score']
        
        color = self.colors["Accent"]
        
        # Scout Mode (unsicher)
        if score < 0.45:
            color = self.colors["TextDim"]
            label = "?"
            
        # 1. Box (1px)
        cv2.rectangle(canvas, (l, t), (r, b), color, 1)
        
        # 2. Label
        if label != "?":
            txt = f"{label} {int(score*100)}%"
        else:
            txt = "?"
            
        # Text Größe
        (tw, th), _ = cv2.getTextSize(txt, self.font_tech, 1.0, 1)
        
        # Label Hintergrund (Surface Color, clean)
        # Position: Oben Links, innen oder außen? Außen ist besser für Übersicht.
        # Wir setzen es "auf" die Linie oben.
        
        cv2.rectangle(canvas, (l, t - th - 8), (l + tw + 8, t), self.colors["Background"], -1) # "Freistellen"
        
        cv2.putText(canvas, txt, (l + 4, t - 4), 
                    self.font_tech, 1.0, self.colors["TextLight"], 1, cv2.LINE_AA)
        
    def draw_offline_screen(self, canvas):
        """Zeichnet einen 'No Signal' Screen, falls Kamera ausfällt."""
        h, w, _ = canvas.shape
        # Background fill
        canvas[:] = self.colors["Background"]
        
        # Zentrierter Text
        text = "NO SIGNAL"
        (tw, th), _ = cv2.getTextSize(text, self.font_main, 1.5, 2)
        cx, cy = w // 2, h // 2
        
        cv2.putText(canvas, text, (cx - tw//2, cy + th//2), 
                    self.font_main, 1.5, self.colors["TextDim"], 2, cv2.LINE_AA)
        
        return canvas
