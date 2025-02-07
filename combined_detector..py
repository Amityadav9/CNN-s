import cv2
import mediapipe as mp
from ultralytics import YOLO
import numpy as np

class CombinedDetector:
    def __init__(self, confidence_threshold=0.5):
        # Initialize YOLO for object detection
        self.yolo_model = YOLO('yolov8n.pt')
        self.conf_threshold = confidence_threshold
        
        # Initialize MediaPipe for multiple pose detection
        self.mp_pose = mp.solutions.pose
        self.poses = [self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1
        ) for _ in range(10)]  # Support up to 10 people
        self.mp_draw = mp.solutions.drawing_utils
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Colors for visualization
        self.pose_colors = [
            (245, 117, 66),  # Orange
            (117, 245, 66),  # Green
            (66, 117, 245),  # Blue
            (245, 66, 117),  # Pink
            (245, 245, 66),  # Yellow
            (66, 245, 245),  # Cyan
            (245, 66, 245),  # Magenta
            (66, 245, 117),  # Light Green
            (117, 66, 245),  # Purple
            (245, 117, 245)  # Light Pink
        ]
        self.object_color = (66, 245, 200)

    def process_frame(self, frame):
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Object Detection with YOLO
        yolo_results = self.yolo_model(frame, task='detect')[0]
        
        # Process YOLO detections
        for box in yolo_results.boxes:
            confidence = float(box.conf[0])
            
            if confidence < self.conf_threshold:
                continue
            
            class_id = int(box.cls[0])
            class_name = self.yolo_model.names[class_id]
            
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.object_color, 2)
            cv2.putText(frame, f'{class_name}: {confidence:.2f}',
                       (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                       self.object_color, 2)
        
        # Multi-Person Pose Detection with MediaPipe
        person_count = 0
        for pose_detector in self.poses:
            pose_results = pose_detector.process(rgb_frame)
            
            if pose_results.pose_landmarks:
                color = self.pose_colors[person_count % len(self.pose_colors)]
                
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
                
                # Draw person count
                cv2.putText(frame, f'Person {person_count + 1}',
                           (10, 60 + 30 * person_count),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                person_count += 1
                if person_count >= len(self.poses):
                    break
        
        # Add FPS counter
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        cv2.putText(
            frame,
            f'FPS: {int(fps)} | People: {person_count}',
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        
        return frame

    def run(self):
        print("Starting combined detection...")
        print("Press 'q' to quit")
        
        try:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Process frame with both detections
                processed_frame = self.process_frame(frame)
                
                # Display frame
                cv2.imshow('Combined Detection', processed_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except Exception as e:
            print(f"Error during detection: {str(e)}")
        finally:
            self.cleanup()

    def cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()
        for pose in self.poses:
            pose.close()

if __name__ == "__main__":
    try:
        detector = CombinedDetector(confidence_threshold=0.5)
        detector.run()
    except Exception as e:
        print(f"Error initializing detector: {str(e)}")