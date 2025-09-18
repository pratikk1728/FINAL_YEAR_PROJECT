import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque

# Load the fine-tuned model
model = YOLO(r'C:\Users\Prati\Desktop\BE Project\@MAIN_3\DETECTION_WEAPON_MAIN\Weapon_detection\best.pt')  # Replace with your model path

# Video setup
video_path = r'C:\Users\Prati\Desktop\BE Project\@MAIN_3\DETECTION_WEAPON_MAIN\Weapon_detection\GIJoe.mp4'  # Input video path
output_path = r'@MAIN_3/DETECTION_WEAPON_MAIN/OP_1.mp4'

# Open video capture and get properties
cap = cv2.VideoCapture(video_path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define codec and create VideoWriter object for output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Initialize a queue to store results across frames for smoothing
detections_queue = deque(maxlen=5)  # Holds detections from last 5 frames

# Process each frame in the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO inference with custom thresholds
    results = model(frame, conf=0.5, iou=0.4)  # Increase confidence threshold, lower IoU threshold if needed

    # Extract detections and filter them
    detections = []
    for result in results[0].boxes:
        if result.conf.item() > 0.5:  # Check confidence
            # Append bounding box, class, and confidence to detections
            x1, y1, x2, y2 = map(int, result.xyxy[0])  # Get box coordinates
            class_id = int(result.cls.item())  # Get class ID
            conf = result.conf.item()  # Get confidence score
            detections.append((x1, y1, x2, y2, class_id, conf))

    # Add detections for this frame to the queue for smoothing
    detections_queue.append(detections)

    # Smooth detections across frames
    smoothed_detections = []
    for i in range(len(detections)):
        # Calculate average box coordinates and confidence over stored frames
        x1_avg = np.mean([det[i][0] for det in detections_queue if len(det) > i])
        y1_avg = np.mean([det[i][1] for det in detections_queue if len(det) > i])
        x2_avg = np.mean([det[i][2] for det in detections_queue if len(det) > i])
        y2_avg = np.mean([det[i][3] for det in detections_queue if len(det) > i])
        conf_avg = np.mean([det[i][5] for det in detections_queue if len(det) > i])

        # Only keep the detection if the averaged confidence is above threshold
        if conf_avg > 0.5:
            smoothed_detections.append((int(x1_avg), int(y1_avg), int(x2_avg), int(y2_avg), conf_avg))

    # Draw smoothed detections on the frame
    for (x1, y1, x2, y2, conf) in smoothed_detections:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'Weapon: {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


    out.write(frame)  # Write frame to output video

    # Press 'q' to exit
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

# Release resources
cap.release()
out.release()
# cv2.destroyAllWindows()
