import cv2
import numpy as np

class Visualizer:
    def __init__(self):
        # --- FARBKONZEPT: DEEP OCEAN (Marine Standard) ---
        # Palette inspiriert von modernen Schiffsbrücken (OpenBridge)
        # Format: BGR (Blue-Green-Red)
        self.colors = {
            "Background": (36, 23, 15),     # #0F1724 (Deep Navy / Midnight Blue)
            "Surface":    (59, 41, 30),     # #1E293B (Slate Navy - für Footer)
            "Accent":     (248, 189, 56),   # #38BDF8 (Cyan / Sky Blue - Hauptakzent)
            "TextLight":  (249, 245, 241),  # #F1F5F9 (Slate White - bester Kontrast auf Blau)
            "TextDim":    (184, 163, 148),  # #94A3B8 (Slate Grey - für Metadaten)
            "Success":    (128, 222, 74),   # #4ADE80 (Eco Green)
            "Warning":    (36, 191, 251),   # #FBBF24 (Amber - soft warning)
            "Alert":      (113, 113, 248)   # #F87171 (Soft Red)
        }
        
        # --- TYPOGRAFIE ---
        # Clean & Nautisch
        self.font_main = cv2.FONT_HERSHEY_SIMPLEX
        self.font_tech = cv2.FONT_HERSHEY_PLAIN
        
        self.footer_h = 42

    def draw_tracker_overlay(self, canvas, detections, fps):
        """
        Zeichnet das 'Deep Ocean' Interface.
        Erwartet Canvas in 'Background'-Farbe (36, 23, 15).
        """
        h_cam, w_cam, _ = canvas.shape
        
        # 1. Kamera-Rahmen (Cyan)
        cam_w, cam_h = 1280, 960
        x_off = (w_cam - cam_w) // 2
        y_off = (h_cam - cam_h) // 2
        
        # 1px Cyan Border (sehr technisch)
        cv2.rectangle(canvas, 
                      (x_off - 1, y_off - 1), 
                      (x_off + cam_w, y_off + cam_h), 
                      self.colors["Accent"], 1)

        # 2. Detections
        for det in detections:
            self.draw_single_detection(canvas, det)

        # 3. Statusleiste (Navy Surface)
        y_foot = h_cam - self.footer_h
        
        # Leiste: Etwas heller als Background für Trennung
        cv2.rectangle(canvas, (0, y_foot), (w_cam, h_cam), self.colors["Surface"], -1)
        
        # Keine Linie, nur Farbflächen-Trennung (sehr modern)
        # Optional: Ganz feine Linie in DimColor
        cv2.line(canvas, (0, y_foot), (w_cam, y_foot), (50, 40, 30), 1)

        # --- FOOTER INHALT ---
        mid_y = y_foot + 28
        
        # A. System-Name (Links)
        cv2.putText(canvas, "OCEAN SENTRY NAV", (30, mid_y), 
                    self.font_main, 0.6, self.colors["TextLight"], 1, cv2.LINE_AA)
        
        # B. Indikatoren
        def draw_dot(cx, color):
            cv2.circle(canvas, (cx, mid_y - 5), 4, color, -1)
            # Schein-Effekt (Glow) für Marine-Look? (Zu teuer für CPU? Checken wir mal: 1 extra circle)
            # cv2.circle(canvas, (cx, mid_y - 5), 7, color, 1) # Ring

        # Live Status
        draw_dot(340, self.colors["Success"])
        cv2.putText(canvas, "LIVE FEED", (355, mid_y),
                    self.font_tech, 1.0, self.colors["TextDim"], 1, cv2.LINE_AA)

        # AI Status
        ai_active = len(detections) > 0
        ai_col = self.colors["Accent"] if ai_active else self.colors["TextDim"] # Cyan wenn aktiv
        
        draw_dot(550, ai_col)
        cv2.putText(canvas, "AI SCANNED", (565, mid_y),
                    self.font_tech, 1.0, self.colors["TextDim"], 1, cv2.LINE_AA)

        # C. FPS
        fps_text = f"{fps:.1f} FPS"
        (w_txt, _), _ = cv2.getTextSize(fps_text, self.font_tech, 1.2, 1)
        x_fps = w_cam - 30 - w_txt
        
        cv2.putText(canvas, fps_text, (x_fps, mid_y), 
                    self.font_tech, 1.2, self.colors["Accent"], 1, cv2.LINE_AA)

        return canvas

    def draw_single_detection(self, canvas, det):
        """Marine Style: Cyan Boxes, Slate Labels"""
        l, t, r, b = det['box']
        label = det['label']
        score = det['score']
        
        color = self.colors["Accent"] # Cyan
        
        # Scout
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
            
        (tw, th), _ = cv2.getTextSize(txt, self.font_tech, 1.0, 1)
        
        # Label Background (Surface Navy)
        cv2.rectangle(canvas, (l, t - th - 8), (l + tw + 8, t), self.colors["Background"], -1)
        
        # Text in Cyan oder Hellweiß?
        # Cyan Text auf Navy wirkt sehr "Tron" / Marine.
        cv2.putText(canvas, txt, (l + 4, t - 4), 
                    self.font_tech, 1.0, color, 1, cv2.LINE_AA)
        
    def draw_offline_screen(self, canvas):
        """Offline Screen in Navy"""
        h, w, _ = canvas.shape
        canvas[:] = self.colors["Background"]
        
        text = "SONAR OFFLINE"
        (tw, th), _ = cv2.getTextSize(text, self.font_main, 1.5, 2)
        cx, cy = w // 2, h // 2
        
        cv2.putText(canvas, text, (cx - tw//2, cy + th//2), 
                    self.font_main, 1.5, self.colors["TextDim"], 2, cv2.LINE_AA)
        
        return canvas
