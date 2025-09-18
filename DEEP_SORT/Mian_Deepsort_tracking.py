import cv2
import torch
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


torch.set_num_threads(1)
cv2.setNumThreads(1)

# Load YOLOv8 model (pre-trained on COCO dataset)
model = YOLO('yolov8n.pt')

# Initialize DeepSORT tracker with increased max_age
tracker = DeepSort(max_age=300, n_init=3)  # max_age increased for better persistence

# Load video
video_path = r'C:\Users\Prati\Desktop\BE Project\@MAIN_3\DEEP_SORT\GIJoe.mp4'
output_path = r'C:\Users\Prati\Desktop\BE Project\@MAIN_3\DEEP_SORT\opp_just.mp4'
cap = cv2.VideoCapture(video_path)

# Get video properties
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = cap.get(5)
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Global variables for target tracking
target_id = None
target_features = None

# Function to calculate Intersection over Union (IoU)
def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    xx1, yy1, xx2, yy2 = box2

    xi1 = max(x1, xx1)
    yi1 = max(y1, yy1)
    xi2 = min(x2, xx2)
    yi2 = min(y2, yy2)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (xx2 - xx1) * (yy2 - yy1)

    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou

# Mouse click callback to select a person
def select_person(event, x, y, flags, param):
    global target_id, target_features
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Selecting person at: ({x}, {y})")
        target_id = identify_person(x, y)

# Identify the person by location
def identify_person(x, y):
    for track in active_tracks:
        if not track.is_confirmed():
            continue

        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)

        if x1 <= x <= x2 and y1 <= y <= y2:
            print(f"Target ID: {track.track_id}")
            return track.track_id

    return None

cv2.namedWindow('Tracking')
cv2.setMouseCallback('Tracking', select_person)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 inference
    results = model(frame, verbose=False)

    detections = []

    # Extract person detections
    for result in results:
        for box in result.boxes.data:
            x1, y1, x2, y2, score, class_id = box.cpu().numpy()
            if int(class_id) == 0 and score > 0.5:  # Class 0 = 'person'
                detections.append(([x1, y1, x2, y2], score, 'person'))

    # Update DeepSORT tracker
    active_tracks = tracker.update_tracks(detections, frame=frame)

    # Ensure target ID persistence by matching bounding box
    if target_id is not None:
        best_match = None
        best_iou = 0.0

        for track in active_tracks:
            if not track.is_confirmed():
                continue

            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)

            # Match bounding box to the target
            if track.track_id == target_id:
                best_match = track
                break

            # Use IoU to find the best match
            iou = compute_iou(ltrb, [x1, y1, x2, y2])
            if iou > best_iou:
                best_match = track
                best_iou = iou

        if best_match:
            target_id = best_match.track_id

    # Draw bounding boxes and track IDs
    for track in active_tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)

        # Different color for the target person
        if track_id == target_id:
            color = (0, 0, 255)  # Red for target
            label = f'TARGET ID: {track_id}'
        else:
            color = (0, 255, 0)  # Green for others
            label = f'ID: {track_id}'

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Display frame
    cv2.imshow('Tracking', frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
