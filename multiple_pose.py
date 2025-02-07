import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO
import torch

class MultiPersonPoseDetector:
    def __init__(self, width=1920, height=1080, fps=30):
        self.width = width
        self.height = height
        self.fps = fps
        
        # Initialize YOLO for person detection
        self.yolo_model = YOLO('yolov8n.pt')
        
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.poses = [self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1
        ) for _ in range(10)]  # Support up to 10 people
        self.mp_draw = mp.solutions.drawing_utils
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        # Colors for different people
        self.colors = [
            (245, 117, 66),   # Orange
            (245, 66, 230),   # Pink
            (66, 245, 200),   # Turquoise
            (245, 245, 66),   # Yellow
            (66, 117, 245),   # Blue
            (188, 66, 245),   # Purple
            (66, 245, 123),   # Green
            (245, 173, 66),   # Gold
            (245, 66, 66),    # Red
            (66, 245, 245)    # Cyan
        ]
    
    def process_frame(self, frame):
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect people using YOLO
        results = self.yolo_model(frame, classes=[0])  # class 0 is person in COCO
        boxes = results[0].boxes
        
        person_count = 0
        
        # Process each detected person
        for i, box in enumerate(boxes):
            if i >= len(self.poses):  # Skip if more people than available pose detectors
                break
                
            # Get person crop coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            
            # Only process high confidence detections
            if confidence < 0.5:
                continue
                
            # Crop and process person
            person_crop = rgb_frame[y1:y2, x1:x2]
            if person_crop.size == 0:
                continue
                
            # Get pose landmarks for this person
            pose_results = self.poses[i].process(person_crop)
            
            if pose_results.pose_landmarks:
                person_count += 1
                color = self.colors[i % len(self.colors)]
                
                # Draw person bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw person ID
                cv2.putText(
                    frame,
                    f'Person {person_count} ({confidence:.2f})',
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )
                
                # Adjust landmark coordinates to original frame
                for landmark in pose_results.pose_landmarks.landmark:
                    landmark.x = landmark.x * (x2 - x1) / self.width + x1 / self.width
                    landmark.y = landmark.y * (y2 - y1) / self.height + y1 / self.height
                
                # Draw pose landmarks
                self.mp_draw.draw_landmarks(
                    frame,
                    pose_results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_draw.DrawingSpec(
                        color=color,
                        thickness=2,
                        circle_radius=2
                    ),
                    connection_drawing_spec=self.mp_draw.DrawingSpec(
                        color=color,
                        thickness=2,
                        circle_radius=2
                    )
                )
        
        # Draw total person count
        cv2.putText(
            frame,
            f'People detected: {person_count}',
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        
        # Draw FPS
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        cv2.putText(
            frame,
            f'FPS: {int(fps)}',
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        
        return frame
    
    def run(self):
        try:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                
                # Process frame
                processed_frame = self.process_frame(frame)
                
                # Display frame
                cv2.imshow('Multi-Person Pose Detection', processed_frame)
                
                # Break loop on 'q' press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except Exception as e:
            print(f"Error during camera operation: {str(e)}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()
        for pose in self.poses:
            pose.close()

if __name__ == "__main__":
    try:
        # Create and run detector instance
        detector = MultiPersonPoseDetector(
            width=1920,
            height=1080,
            fps=30
        )
        print("Starting true multi-person pose detection...")
        print("Press 'q' to quit")
        detector.run()
    except Exception as e:
        print(f"Error initializing detector: {str(e)}")