import cv2
import mediapipe as mp
from ultralytics import YOLO
import os

# Load your 1080p video
# Make sure to replace the path with the correct one for your system
input_video = ""
output_video = os.path.join("./edits", f"{os.path.splitext(input_video)[0]}.mp4")
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

    if pose:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

        # Process the frame and get pose landmarks
        results = pose.process(rgb_frame)

        # Draw pose landmarks on the original frame
        if results.pose_landmarks:
            landmark_spec = mp_drawing.DrawingSpec(
                color=(0, 255, 0), thickness=2, circle_radius=2
            )
            connection_spec = mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)

            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_spec,
                connection_spec,
            )

            # Draw additional pose connections
            landmarks = results.pose_landmarks.landmark

            def draw_extra_connection(p1, p2):
                x1 = int(landmarks[p1].x * width)
                y1 = int(landmarks[p1].y * height)
                x2 = int(landmarks[p2].x * width)
                y2 = int(landmarks[p2].y * height)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            extra_connctions = [
                (0, 1),  # Nose to inner eye
                (0, 4),  # Nose to outer eye
                (9, 10),  # Shoulders connection
                (11, 13),  # Upper arms connections
                (12, 14),
                (13, 15),  # Lower arms connections
                (14, 16),
                (23, 24),  # Hips connection
                (11, 23),  # Shoulders to hips
                (12, 14),
                (23, 25),  # Upper legs connections
                (24, 26),
                (25, 27),  # Lower legs connections
                (26, 28),
                (27, 31),  # Foot connections
                (28, 32),
            ]
            for connection in extra_connctions:
                draw_extra_connection(*connection)

    # Write the processed frame to the output video
    out.write(frame)

# Release resources
cap.release()
out.release()
pose.close()
cv2.destroyAllWindows()

print(f"Output video saved as {output_video}")
