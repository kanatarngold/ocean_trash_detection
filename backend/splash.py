import tkinter as tk
from tkinter import ttk
import subprocess
import threading
import time
import sys
import os

class SplashScreen:
    def __init__(self):
        self.root = tk.Tk()
        
        # Professional Frameless Window
        self.root.overrideredirect(True)
        
        # Geometry: Center on Screen
        width = 600
        height = 300
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width // 2) - (width // 2)
        y = (screen_height // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')
        
        # Styling (Cyberpunk / High Tech)
        bg_color = "#050510" # Very Dark Blue/Black
        self.root.configure(bg=bg_color)
        
        # Header
        self.label_title = tk.Label(self.root, text="OCEAN SENTRY AI", font=("Courier", 34, "bold"), fg="#00FFFF", bg=bg_color)
        self.label_title.pack(pady=(60, 10))
        
        # Subtitle
        self.label_sub = tk.Label(self.root, text="INITIALIZING NEURAL NETWORK...", font=("Arial", 12), fg="#AAAAAA", bg=bg_color)
        self.label_sub.pack(pady=(0, 40))
        
        # Progress Bar (Custom Style)
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("green.Horizontal.TProgressbar", foreground='#00FFFF', background='#00FFFF', troughcolor='#222222', bordercolor='#000000', lightcolor='#00FFFF', darkcolor='#00FFFF')
        
        self.progress = ttk.Progressbar(self.root, style="green.Horizontal.TProgressbar", orient="horizontal", length=400, mode="indeterminate")
        self.progress.pack()
        self.progress.start(15) # Speed of animation
        
        # Status Text
        self.status_label = tk.Label(self.root, text="Loading TensorFlow Modules...", font=("Consolas", 10), fg="#555555", bg=bg_color)
        self.status_label.pack(side="bottom", pady=20)

    def launch_main_app(self):
        # Start the heavy inference script
        # We use Popen to readstdout line by line
        creationflags = 0
        self.process = subprocess.Popen(
            [sys.executable, "inference_pi.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1,
            creationflags=creationflags
        )
        
        # Build a thread to monitor output
        t = threading.Thread(target=self.monitor_output)
        t.daemon = True
        t.start()

    def monitor_output(self, timeout=30):
        # Read lines until we see "System bereit" or timeout
        start = time.time()
        ready = False
        
        while True:
            line = self.process.stdout.readline()
            if not line:
                break
            
            print(f"[Launcher] {line.strip()}") # Forward to internal log
            
            # Simple keyword matching to update UI status
            if "TensorFlow" in line:
                 self.update_status("Loading TensorFlow Engines...")
            if "Picamera" in line:
                 self.update_status("Accessing Camera Sensors...")
            if "System bereit" in line or "Ocean Sentry AI" in line:
                # SUCCESS!
                ready = True
                break
                
            if time.time() - start > timeout:
                break

        # Wait a tiny bit for the window to actually appear
        time.sleep(2.0) 
        
        # Destroy the splash window but keep this script running
        # because it is the parent of the inference process
        self.root.destroy()
        
        # Wait for the main app to finish (User presses 'q')
        self.process.wait()

    def update_status(self, text):
        # Check if window exists before updating
        if self.root:
            try:
                self.status_label.config(text=text)
            except:
                pass

    def run(self):
        # Start the app launch in a separate thread so UI doesn't freeze
        t = threading.Thread(target=self.launch_main_app)
        t.daemon = True
        t.start()
        
        # Start UI loop
        self.root.mainloop()

if __name__ == "__main__":
    app = SplashScreen()
    app.run()
