import cv2
import torch
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


torch.set_num_threads(1)
cv2.setNumThreads(1)

# Load models
weapon_detector = YOLO(r"C:\Users\Prati\Desktop\BE Project\@MAIN_3\DETECTION_WEAPON_MAIN\best.pt")  # YOLOv8 for weapon detection
pose_estimator = YOLO("yolov8n-pose.pt")  # YOLOv8-Pose for keypoints
tracker = DeepSort(max_age=300, n_init=3)  # DeepSORT tracker

device = "cuda" if torch.cuda.is_available() else "cpu"
weapon_detector.to(device)
pose_estimator.to(device)

# Video paths
input_video_path = r"C:\Users\Prati\Desktop\BE Project\@MAIN_3\DETECTION_WEAPON_MAIN\GIJoe.mp4"
output_video_path = "LEtS_SEEiT.mp4"

cap = cv2.VideoCapture(input_video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Weapon detection
    weapon_results = weapon_detector(frame)
    weapons = []
    for result in weapon_results:
        for box in result.boxes.data:
            x1, y1, x2, y2, conf, cls = box.cpu().numpy()
            if int(cls) == 0 and conf > 0.5:  # Assuming class 0 is 'gun'
                weapons.append((x1, y1, x2, y2))
    
    # Pose estimation
    # Pose estimation with error handling
    if weapons:
        pose_results = pose_estimator(frame)
        for result in pose_results:
            keypoints = result.keypoints.xy.cpu().numpy()

            # Ensure keypoints exist before accessing them
            if keypoints.shape[0] == 0:
                continue  # Skip if no keypoints are detected

            for weapon in weapons:
                wx1, wy1, wx2, wy2 = weapon
                closest_hand = None
                min_distance = float("inf")

                # Ensure there are at least 10 keypoints before accessing index 9 (right wrist)
                if len(keypoints) > 9:
                    hand_x, hand_y = keypoints[9]  # Right wrist

                    dist = np.linalg.norm([hand_x - wx1, hand_y - wy1])
                    if dist < min_distance:
                        min_distance = dist
                        closest_hand = (hand_x, hand_y)

                # Draw bounding box and hand point
                if closest_hand:
                    cv2.rectangle(frame, (int(wx1), int(wy1)), (int(wx2), int(wy2)), (0, 0, 255), 2)
                    cv2.circle(frame, (int(closest_hand[0]), int(closest_hand[1])), 5, (255, 0, 0), -1)

                    
    # Person tracking
    detections = [[(x1, y1, x2, y2), 0.9, "armed_person"] for x1, y1, x2, y2 in weapons]
    active_tracks = tracker.update_tracks(detections, frame=frame)
    
    for track in active_tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'Track ID: {track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    out.write(frame)
    
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"✅ Integrated tracking video saved as: {output_video_path}")
