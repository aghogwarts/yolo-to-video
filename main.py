import cv2
import mediapipe as mp
from ultralytics import YOLO
import os

# Load your 1080p video
# Make sure to replace the path with the correct one for your system
input_video = ""
output_video = "yolo.mp4"  # Output video path in the same directory
USE_MEDIAPIPE = True
USE_YOLO = True

# Check if the output video already exists
if os.path.exists(output_video):
    print(
        f"Output video {output_video} already exists. Please delete it or choose a different name or a different file."
    )
    exit()

# Initlalize MediaPipe Pose with custom configuration
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = (
    mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        smooth_landmarks=True,
        # enable_segmentation=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    if USE_MEDIAPIPE
    else None
)

# Initialize YOLOv8 model
model = YOLO("yolov8n.pt") if USE_YOLO else None
