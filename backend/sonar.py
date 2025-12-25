import wave
import math
import struct
import os
import subprocess
import threading
import time

class Sonar:
    def __init__(self, filename="sonar_ping.wav"):
        self.filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
        self.last_ping = 0
        self.cooldown = 2.0 # Seconds between pings
        
        # Generate sound if missing
        if not os.path.exists(self.filename):
            self.generate_ping()
            
    def generate_ping(self):
        """Generates a synthetic sonar ping WAV file."""
        print("🔊 Generiere Sonar-Sound...")
        framerate = 44100
        duration = 0.15 # seconds
        freq = 880.0 # Hz (High pitch A5)
        
        # Open file
        with wave.open(self.filename, 'w') as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(framerate)
            
            # Generate sine wave with decay (Ping effect)
            for i in range(int(duration * framerate)):
                t = i / framerate
                # Sine wave
                value = math.sin(2 * math.pi * freq * t)
                # Exponential decay volume
                volume = 32767.0 * math.exp(-t * 20) 
                data = struct.pack('<h', int(value * volume))
                f.writeframesraw(data)
                
    def ping(self):
        """Plays the sound asynchronously if cooldown allowed."""
        now = time.time()
        if now - self.last_ping > self.cooldown:
            self.last_ping = now
            # Uses 'aplay' (standard on Raspberry Pi / Linux)
            # Run in thread to not block main loop!
            threading.Thread(target=self._play_thread).start()
            
    def _play_thread(self):
        try:
            subprocess.run(["aplay", "-q", self.filename], check=False)
        except Exception:
            # Silence errors (e.g. on Mac without aplay)
            pass

if __name__ == "__main__":
    # Test
    s = Sonar()
    s.ping()
    time.sleep(1)
