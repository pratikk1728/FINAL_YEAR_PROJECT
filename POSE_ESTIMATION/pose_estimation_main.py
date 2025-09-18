#pip install ultralytics opencv-python-headless numpy torch torchvision torchaudio

import cv2
import torch
import numpy as np
from ultralytics import YOLO

# Load the YOLOv8-Pose model
model = YOLO("yolov8n-pose.pt")

# Set device (GPU if available)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Input & output video paths
input_video_path = "GIJoe.mp4"  # Replace with your video file
output_video_path = "pose_detected_output.mp4"

# Open the input video
cap = cv2.VideoCapture(input_video_path)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))        # Frames per second
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Frame width
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Frame height

# Define the video writer (XVID codec for saving video)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run pose estimation
    results = model(frame)

    # Get the annotated frame
    annotated_frame = results[0].plot()

    # Write the annotated frame to the output video
    out.write(annotated_frame)

cap.release()
out.release()

print(f"✅ Pose estimation video saved as: {output_video_path}")


