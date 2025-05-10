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

# Open the video file
cap = cv2.VideoCapture(input_video)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"-- Analysing {frame_count} frames in {input_video}")

# Process the video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if model:
        results = model(frame, verbose=False)
        frame = results[0].plot()
