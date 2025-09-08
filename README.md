## Homecam
This is a simple script that utilizes pretrained AI model `yolov5s.pt` to detect humans in real time video streams. It can be configured to automatically record clips when detection occurs and send them to target recipients.

## Config (`.env`)
```
IP=192.168.1.10
CAM_USERNAME=your_camera_username
CAM_PASSWORD=your_camera_password

SENDER_EMAIL=you@example.com
SENDER_PASSWORD=your_app_password
RECIPIENT_EMAILS=recipient1@example.com,recipient2@example.com

SNIPPET_DURATION=30
FRAME_WIDTH=640
FRAME_HEIGHT=480
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
```
