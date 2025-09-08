import os
import time
import warnings
import smtplib
import subprocess
import atexit
from threading import Thread, Event
from urllib.parse import quote
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

import torch
import cv2
import rumps
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore", category=FutureWarning)

IP_ADDRESS = os.getenv("IP", "")
USERNAME = os.getenv("CAM_USERNAME", "")
PASSWORD = os.getenv("CAM_PASSWORD", "")
RTSP_URL = f'rtsp://{quote(USERNAME)}:{quote(PASSWORD)}@{IP_ADDRESS}:554/stream1'

SNIPPET_DURATION = int(os.getenv("SNIPPET_DURATION", "30"))
FRAME_WIDTH = int(os.getenv("FRAME_WIDTH", "640"))
FRAME_HEIGHT = int(os.getenv("FRAME_HEIGHT", "480"))

RECORDINGS_DIR = os.path.join(os.getcwd(), "recordings")
os.makedirs(RECORDINGS_DIR, exist_ok=True)

SENDER_EMAIL = os.getenv("SENDER_EMAIL", "")
SENDER_PASSWORD = os.getenv("SENDER_PASSWORD", "")
RECIPIENT_EMAILS = [e.strip() for e in os.getenv("RECIPIENT_EMAILS", "").split(",") if e.strip()]
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
EMAIL_SUBJECT = "Video Clip From Home Camera"
EMAIL_BODY = "Eine Aufnahme wurde genommen"

def send_email_with_video(video_path):
    message = MIMEMultipart()
    message['From'] = SENDER_EMAIL
    message['To'] = ', '.join(RECIPIENT_EMAILS)
    message['Subject'] = EMAIL_SUBJECT
    message.attach(MIMEText(EMAIL_BODY, 'plain'))
    with open(video_path, 'rb') as video_file:
        part = MIMEBase('video', 'mp4')
        part.set_payload(video_file.read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', f'attachment; filename={os.path.basename(video_path)}')
        message.attach(part)
    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(message)
            print("Email sent successfully to:", ', '.join(RECIPIENT_EMAILS))
    except Exception as e:
        print("Failed to send email:", e)

print("Loading YOLOv5 model, please wait...")
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
print("YOLOv5 model loaded.")

class VideoWriter:
    def __init__(self, filename, fps, frame_size):
        self.filename = filename
        self.fps = fps
        self.frame_size = frame_size
        self.writer = None
        self.active = False
    def start(self, codec='mp4v'):
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self.writer = cv2.VideoWriter(self.filename, fourcc, self.fps, self.frame_size)
        self.active = True
    def write(self, frame):
        if self.active and self.writer is not None:
            self.writer.write(frame)
    def stop(self):
        if self.writer is not None:
            self.writer.release()
        self.active = False

def detect_person(frame):
    results = model(frame)
    detections = results.xyxy[0]
    for *_, conf, cls in detections:
        if int(cls) == 0:
            return True
    return False

class CameraWorker(Thread):
    def __init__(self):
        super().__init__()
        self._stop_event = Event()
        self.recording = False
        self.writer = None
        self.start_time = 0
    def run(self):
        print("CameraWorker started.")
        cap = cv2.VideoCapture(RTSP_URL)
        if not cap.isOpened():
            print("Error: Cannot access RTSP stream.")
            return
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 10
            print(f"Warning: FPS from camera is invalid. Using fallback = {fps}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        while not self._stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                print("Error: Cannot read frame from RTSP stream.")
                time.sleep(0.5)
                continue
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            person_detected = detect_person(frame)
            if person_detected and not self.recording:
                filename = os.path.join(RECORDINGS_DIR, f"snippet_{int(time.time())}.mp4")
                self.writer = VideoWriter(filename, fps, (FRAME_WIDTH, FRAME_HEIGHT))
                self.writer.start(codec='mp4v')
                self.recording = True
                self.start_time = time.time()
                print(f"Recording started: {filename}")
            if self.recording and self.writer:
                self.writer.write(frame)
                if time.time() - self.start_time >= SNIPPET_DURATION:
                    self.writer.stop()
                    print(f"Video saved: {self.writer.filename}")
                    self.recording = False
                    send_email_with_video(self.writer.filename)
        if cap is not None:
            cap.release()
        print("CameraWorker ended.")
    def stop(self):
        self._stop_event.set()
        if self.recording and self.writer is not None:
            self.writer.stop()
            print(f"Video saved (on stop): {self.writer.filename}")
            self.recording = False

def keep_mac_awake():
    return subprocess.Popen(["caffeinate", "-i"])

@atexit.register
def cleanup_caffeinate():
    global caffeinate_proc
    if 'caffeinate_proc' in globals() and caffeinate_proc:
        print("Stopping caffeinate...")
        caffeinate_proc.terminate()

class MyMenuBarApp(rumps.App):
    def __init__(self):
        super(MyMenuBarApp, self).__init__("MyCam")
        self.camera_worker = None
        self.toggle_item = rumps.MenuItem(title="🟢 START", callback=self.toggle_camera)
        self.menu.clear()
        self.menu.add(self.toggle_item)
    def toggle_camera(self, sender):
        if not self.camera_worker or not self.camera_worker.is_alive():
            self.camera_worker = CameraWorker()
            self.camera_worker.start()
            sender.title = "🔴 STOP"
        else:
            self.camera_worker.stop()
            self.camera_worker.join()
            self.camera_worker = None
            sender.title = "🟢 START"

def main():
    global caffeinate_proc
    caffeinate_proc = keep_mac_awake()
    MyMenuBarApp().run()

if __name__ == "__main__":
    main()

