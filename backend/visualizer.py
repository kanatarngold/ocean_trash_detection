import cv2
import numpy as np

class Visualizer:
    def __init__(self):
        # --- 1. FARBKONZEPT (Industrial Precision) ---
        # Alle Farben entsättigt und augenschonend für Dauerbetrieb.
        # Format: BGR (Blue-Green-Red)
        self.colors = {
            "Background": (30, 30, 30),     # #1E1E1E (Anthrazit, neutral)
            "Surface":    (45, 45, 45),     # #2D2D2D (UI Flächen)
            "Accent":     (21, 195, 229),   # #E5C315 (Industrial Yellow, BGR: 21,195,229)
            "TextMain":   (204, 204, 204),  # #CCCCCC (Hellgrau/Silber)
            "TextDim":    (150, 150, 150),  # #969696 (Dunkelgrau/Dim)
            "StatusGood": (106, 187, 102),  # #66BB6A (Sanftes Indikator-Grün)
            "StatusWarn": (60, 130, 240),   # #F0823C (Sanftes Orange)
            "StatusBad":  (80, 80, 220)     # #DC5050 (Sanftes Rot)
        }
        
        # --- 2. TYPOGRAFIE ---
        # Simplex für UI (Clean), Plain für technische Daten
        self.font_ui = cv2.FONT_HERSHEY_SIMPLEX 
        self.font_tech = cv2.FONT_HERSHEY_PLAIN
        
        # Visuelle Konstanten
        self.grid_size = 8
        self.footer_height = 40  # Schlanke Statusleiste
        
    def hex_to_bgr(self, hex_val):
        """Hilfsmethode, falls man Hex nutzen will (hier manuell gemacht)"""
        pass

    def draw_tracker_overlay(self, canvas, detections, fps):
        """
        Zeichnet das 'Precision Dashboard' auf den Canvas.
        Fokus: Ruhe, Ordnung, Lesbarkeit.
        """
        h_canvas, w_canvas, _ = canvas.shape
        
        # --- BACKGROUND RESET ---
        # Wir überschreiben den Rand mit unserem exakten Industrie-Grau
        # (Da inference_pi.py nur schwarz macht, färben wir hier nach)
        
        # Kamera-Zentrum berechnen (1280x960)
        cam_w, cam_h = 1280, 960
        x_off = (w_canvas - cam_w) // 2
        y_off = (h_canvas - cam_h) // 2
        
        # Linker Rand
        cv2.rectangle(canvas, (0, 0), (x_off, h_canvas), self.colors["Background"], -1)
        # Rechter Rand
        cv2.rectangle(canvas, (x_off + cam_w, 0), (w_canvas, h_canvas), self.colors["Background"], -1)
        # Oberer Rand
        cv2.rectangle(canvas, (0, 0), (w_canvas, y_off), self.colors["Background"], -1)
        # Unterer Rand (bis Footer)
        cv2.rectangle(canvas, (0, y_off + cam_h), (w_canvas, h_canvas - self.footer_height), self.colors["Background"], -1)

        # --- KAMERA RAHMEN (1px Akzent) ---
        cv2.rectangle(canvas, 
                      (x_off - 1, y_off - 1), 
                      (x_off + cam_w + 1, y_off + cam_h + 1), 
                      self.colors["Accent"], 1)

        # --- DETECTIONS ---
        # Falls keine Detections da sind, aber Kamera läuft -> Clean lassen
        for det in detections:
            self.draw_single_detection(canvas, det)

        # --- STATUSLEISTE (Footer) ---
        footer_y = h_canvas - self.footer_height
        
        # Hintergrund Leiste
        cv2.rectangle(canvas, 
                      (0, footer_y), 
                      (w_canvas, h_canvas), 
                      self.colors["Surface"], -1)
        
        # Trennlinie (Top Border of Footer)
        cv2.line(canvas, (0, footer_y), (w_canvas, footer_y), (60, 60, 60), 1)

        # --- STATUS ELEMENTE (Grid Aligned) ---
        # Vertikal zentriert im Footer
        mid_y = footer_y + 25 
        
        # 1. System Name (Links)
        cv2.putText(canvas, "OCEAN SENTRY", (self.grid_size * 4, mid_y), 
                    self.font_ui, 0.5, self.colors["TextMain"], 1, cv2.LINE_AA)
        
        # 2. Live Indikator (Links)
        self._draw_status_dot(canvas, self.grid_size * 25, mid_y - 5, self.colors["StatusGood"])
        cv2.putText(canvas, "LIVE FEED", (self.grid_size * 28, mid_y), 
                    self.font_tech, 1.0, self.colors["TextDim"], 1, cv2.LINE_AA)

        # 3. AI Status (Mitte)
        ai_active = len(detections) > 0
        ai_col = self.colors["StatusGood"] if ai_active else self.colors["TextDim"]
        ai_txt = "AI ACTIVE" if ai_active else "AI IDLE"
        
        # Dot
        self._draw_status_dot(canvas, w_canvas // 2 - 60, mid_y - 5, ai_col)
        # Text
        cv2.putText(canvas, ai_txt, (w_canvas // 2 - 40, mid_y), 
                    self.font_tech, 1.0, self.colors["TextMain"], 1, cv2.LINE_AA)

        # 4. FPS (Rechts)
        fps_txt = f"FPS: {fps:.1f}"
        (tw, th), _ = cv2.getTextSize(fps_txt, self.font_tech, 1.0, 1)
        fps_x = w_canvas - (self.grid_size * 4) - tw
        
        # FPS Dot Color
        fps_col = self.colors["StatusGood"]
        if fps < 10: fps_col = self.colors["StatusWarn"]
        if fps < 5: fps_col = self.colors["StatusBad"]
        
        self._draw_status_dot(canvas, fps_x - 15, mid_y - 5, fps_col)
        cv2.putText(canvas, fps_txt, (fps_x, mid_y), 
                    self.font_tech, 1.0, self.colors["TextDim"], 1, cv2.LINE_AA)

        return canvas

    def draw_single_detection(self, canvas, det):
        """Minimalistische Boxen: 1px Linien, dezente Labels"""
        left, top, right, bottom = det['box']
        label = det['label']
        score = det['score']
        
        # Farbe: Standard Akzent (Gelb) für alles, für Ruhe.
        # Nur bei Scout (Unsicher) grau.
        color = self.colors["Accent"]
        
        is_scout = False
        if score < 0.45:
            is_scout = True
            color = self.colors["TextDim"] # Grau
            label = "?"

        # 1. Rahmen (1px, sehr fein)
        cv2.rectangle(canvas, (left, top), (right, bottom), color, 1)
        
        # 2. Label (Text only, oder sehr dezenter Background)
        # Wir machen Text direkt über Box, mit kleinem Rechteck als BG für Lesbarkeit
        if not is_scout:
            label_text = f"{label.upper()} {score:.0%}"
        else:
            label_text = "?"
            
        (tw, th), base = cv2.getTextSize(label_text, self.font_tech, 1.0, 1)
        
        # Label Background (Surface Color, 80% Opacity wär cool, aber CV2 kann nur solid)
        # Wir nehmen Surface Farbe
        cv2.rectangle(canvas, 
                      (left, top - th - 6), 
                      (left + tw + 6, top), 
                      self.colors["Surface"], -1)
        
        # Kleiner farbiger Strich oben am Label (Akzent)
        cv2.line(canvas, (left, top - th - 6), (left + tw + 6, top - th - 6), color, 1)

        # Text
        cv2.putText(canvas, label_text, (left + 3, top - 4), 
                    self.font_tech, 1.0, self.colors["TextMain"], 1, cv2.LINE_AA)

        # Ecken andeuten? (Brackets) - Optional, aktuell 1px Rahmen clean.

    def _draw_status_dot(self, canvas, x, y, color):
        """Zeichnet einen kleinen Indikator-Punkt (besser als Textstatus)"""
        cv2.circle(canvas, (x, y), 3, color, -1)
