# FINAL_YEAR_PROJECT
# 🔫 Weapon Detection and Person Tracking System

A real-time surveillance system built using two separate models: **YOLOv8** for weapon detection and **DeepSort** for person tracking.

## 🚀 Key Features

- Developed a real-time surveillance system using **YOLOv8** for weapon detection and **DeepSort** as a separate model for tracking individuals in video streams.  
- Trained a custom YOLOv8 model to detect firearms and knives, achieving high accuracy across varied lighting and background conditions.  
- Implemented **DeepSort** to assign consistent IDs to individuals and accurately track their movement across frames, even in crowded or occluded scenes.  

## 🛠 Technologies Used

- Python
- YOLOv8 (Ultralytics)
- DeepSort
- OpenCV
- NumPy

## 📂 Project Structure

```
weapon-detection-tracking/
├── models/
│   └── yolov8_custom.pt
├── deep_sort/
│   └── deep_sort_files/
├── images/
│   ├── weapon_detected.jpg
│   └── person_tracked.jpg
├── video/
│   └── input_video.mp4
├── main.py
└── README.md
```

## ▶️ How to Run

```bash
git clone https://github.com/your-username/weapon-detection-tracking.git
cd weapon-detection-tracking

# Install dependencies
pip install -r requirements.txt

# Run detection and tracking
python main.py --source video/input_video.mp4 --model models/yolov8_custom.pt
```

## 📷 Sample Output

| Weapon Detection | Person Tracking |
|------------------|------------------|
| ![Detection](![Screenshot 2025-03-23 173849](https://github.com/user-attachments/assets/24f455be-26f6-4765-8350-765e4653dbd1)
) | ![Tracking](![Screenshot 2025-03-23 173951](https://github.com/user-attachments/assets/a5e1d22b-28af-4737-a191-6659a9c64239)
) |

## 📌 Future Improvements

- Integration of both the models for high accuracy
- Add audio or visual alerts when a weapon is detected
- Integrate RTSP/CCTV live stream support
- Deploy the system on edge devices like Jetson Nano or Raspberry Pi

## 👤 Author

- **Pratik Kamble and group** – [GitHub]((https://github.com/pratikk1728))



