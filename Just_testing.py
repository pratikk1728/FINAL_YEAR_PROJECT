import cv2
import torch
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 models (custom-trained for weapon detection and pose estimation)
weapon_model = YOLO(r'DETECTION_WEAPON_MAIN\best.pt')  # Replace with your trained weapon model
pose_model = YOLO('yolov8n-pose.pt')  # Use YOLOv8 pre-trained pose estimation model

# Load video
video_path = r'DEEP_SORT\GIJoe.mp4'  # Replace with actual video file
output_path = 'output_video.mp4'
cap = cv2.VideoCapture(video_path)

# Video properties
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = cap.get(5)
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

def find_nearest_keypoint(weapon_box, keypoints):
    wx, wy = (weapon_box[0] + weapon_box[2]) // 2, (weapon_box[1] + weapon_box[3]) // 2
    min_distance = float('inf')
    nearest_point = None
    
    for kp in keypoints:
        if kp[2] > 0.5:  # Confidence threshold
            distance = np.sqrt((kp[0] - wx) ** 2 + (kp[1] - wy) ** 2)
            if distance < min_distance:
                min_distance = distance
                nearest_point = kp
    
    return nearest_point, min_distance

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO inference for weapon detection
    weapon_results = weapon_model(frame, verbose=False)
    pose_results = pose_model(frame, verbose=False)
    
    weapons = []
    armed_persons = []
    
    # Process weapon detections
    for result in weapon_results:
        for box in result.boxes.data:
            x1, y1, x2, y2, score, class_id = box[:6].cpu().numpy()
            if score > 0.5:  # Adjust confidence threshold
                weapons.append(([int(x1), int(y1), int(x2), int(y2)], score))

    # Process pose detections
    for result in pose_results:
        for keypoints in result.keypoints.data.cpu().numpy():
            if keypoints.shape[0] < 11:
                continue  # Ensure there are enough keypoints
            
            left_wrist, right_wrist = keypoints[9], keypoints[10]  # Wrist keypoints
            
            for weapon in weapons:
                weapon_box, _ = weapon
                nearest_wrist, distance = find_nearest_keypoint(weapon_box, [left_wrist, right_wrist])
                if nearest_wrist is not None and distance < 50:  # Threshold distance
                    armed_persons.append(nearest_wrist)
                    
                    # Capture and store the person image
                    person_crop = frame[max(0, weapon_box[1] - 50):min(frame_height, weapon_box[3] + 50), 
                                        max(0, weapon_box[0] - 50):min(frame_width, weapon_box[2] + 50)]
                    cv2.imwrite(f'armed_person_{len(armed_persons)}.jpg', person_crop)
                    break

    # Draw detections
    for weapon in weapons:
        x1, y1, x2, y2, _ = weapon[0]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, 'Weapon', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    
    for person in armed_persons:
        cv2.circle(frame, (int(person[0]), int(person[1])), 10, (255, 0, 0), -1)
    
    # Display frame
    cv2.imshow('Weapon Detection & Pose Estimation', frame)
    out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
