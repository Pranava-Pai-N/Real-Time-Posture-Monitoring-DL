import cv2
import math
import time
import numpy as np
from collections import deque
from ultralytics import YOLO
from plyer import notification 

model = YOLO("yolov8m-pose.pt")

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle


class AutoCalibrator:
    def __init__(self, duration_seconds=5.0):
        self.duration = duration_seconds
        self.is_calibrated = False
        self.start_time = None
        self.dist_samples = []
        self.angle_samples = []
        self.too_close_thresh = 0
        self.slouch_thresh = 0

    def start(self):
        """Reset and begin calibration."""
        self.is_calibrated = False
        self.start_time = time.time()
        self.dist_samples.clear()
        self.angle_samples.clear()

    def update(self, current_dist, current_angle):
        if self.is_calibrated:
            return True, "Calibrated"

        if self.start_time is None:
            self.start()

        elapsed = time.time() - self.start_time

        if elapsed < self.duration:
            self.dist_samples.append(current_dist)
            self.angle_samples.append(current_angle)
            time_left = max(0, int(np.ceil(self.duration - elapsed)))
            return False, f"CALIBRATING... Sit straight! ({time_left}s)"
        else:
            if len(self.dist_samples) > 0:
                base_dist = sum(self.dist_samples) / len(self.dist_samples)
                base_angle = sum(self.angle_samples) / len(self.angle_samples)

                self.too_close_thresh = base_dist * 1.20
                self.slouch_thresh = base_angle - 12.0
                self.is_calibrated = True
                return True, "CALIBRATION COMPLETE!"
            else:
                self.start()
                return False, "Recalibrating..."


calibrator = AutoCalibrator(duration_seconds=5.0)
ear_dist_history = deque(maxlen=8)
angle_history = deque(maxlen=8)


slouch_start_time = None
last_notification_time = 0
SLOUCH_TIME_TRIGGER = 5.0  
NOTIFICATION_COOLDOWN = 10.0 

cap = cv2.VideoCapture(0)

print("Starting Ergonomic Monitor with System Notifications...")
print("Press 'c' to Recalibrate | Press 'q' to Quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    results = model(frame, conf=0.5, verbose=False)

    for r in results:
        if r.keypoints is not None and len(r.keypoints.xy) > 0:
            kpts = r.keypoints.xy[0].cpu().numpy()

            if len(kpts) >= 7:
                l_ear, r_ear = kpts[3], kpts[4]
                l_shoulder, r_shoulder = kpts[5], kpts[6]

                if l_ear[0] > 0 and r_ear[0] > 0 and l_shoulder[0] > 0 and r_shoulder[0] > 0:
                    raw_ear_dist = math.hypot(r_ear[0] - l_ear[0], r_ear[1] - l_ear[1])

                    mid_ear = [(l_ear[0] + r_ear[0]) / 2, (l_ear[1] + r_ear[1]) / 2]
                    mid_shoulder = [(l_shoulder[0] + r_shoulder[0]) / 2, (l_shoulder[1] + r_shoulder[1]) / 2]
                    reference_vertical = [mid_shoulder[0], 0]

                    raw_posture_angle = calculate_angle(reference_vertical, mid_shoulder, mid_ear)

                    ear_dist_history.append(raw_ear_dist)
                    angle_history.append(raw_posture_angle)

                    ear_distance = sum(ear_dist_history) / len(ear_dist_history)
                    posture_angle = sum(angle_history) / len(angle_history)

                    is_ready, cal_status_msg = calibrator.update(ear_distance, posture_angle)

                    cv2.circle(frame, (int(mid_ear[0]), int(mid_ear[1])), 6, (255, 0, 0), -1)
                    cv2.circle(frame, (int(mid_shoulder[0]), int(mid_shoulder[1])), 6, (0, 255, 0), -1)
                    cv2.line(frame, (int(mid_ear[0]), int(mid_ear[1])), 
                             (int(mid_shoulder[0]), int(mid_shoulder[1])), (0, 255, 255), 2)

                    if not is_ready:
                        cv2.rectangle(frame, (0, 0), (w, 60), (0, 165, 255), -1)
                        cv2.putText(frame, cal_status_msg, (20, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                        slouch_start_time = None  
                    else:
                        cv2.putText(frame, f"Eye Dist: {int(ear_distance)}px (Threshold: >{int(calibrator.too_close_thresh)}px)", 
                                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                        cv2.putText(frame, f"Posture Angle: {int(posture_angle)}deg (Threshold: <{int(calibrator.slouch_thresh)}deg)", 
                                    (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

                        is_too_close = ear_distance > calibrator.too_close_thresh
                        is_slouching = posture_angle < calibrator.slouch_thresh

                        current_time = time.time()

                        if is_slouching or is_too_close:
                            if slouch_start_time is None:
                                slouch_start_time = current_time
                            
                            slouch_duration = current_time - slouch_start_time

                            cv2.putText(frame, f"Bad Posture Duration: {slouch_duration:.1f}s", (20, 120),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

                            if slouch_duration >= SLOUCH_TIME_TRIGGER:
                                if current_time - last_notification_time > NOTIFICATION_COOLDOWN:
                                    notification.notify(
                                        title="Ergonomic Alert",
                                        message=f"You have been slouching for over {slouch_duration} seconds! Sit up straight.",
                                        app_name="Posture Monitor",
                                        timeout=4  
                                    )
                                    last_notification_time = current_time
                        else:
                            slouch_start_time = None

    cv2.putText(frame, "Press 'c' to Recalibrate | Press 'q' to Quit", (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow("Touchless Ergonomic Monitor", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        calibrator.start()
        slouch_start_time = None
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()