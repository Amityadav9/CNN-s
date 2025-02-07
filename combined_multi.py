import cv2
from ultralytics import YOLO
import numpy as np

class CombinedDetector:
    def __init__(self, confidence_threshold=0.5):
        # Initialize YOLO models
        self.object_model = YOLO('yolov8n.pt')  # for object detection
        self.pose_model = YOLO('yolov8n-pose.pt')  # for pose detection
        self.conf_threshold = confidence_threshold
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Colors for visualization
        self.pose_colors = [
            (245, 117, 66),   # Orange
            (117, 245, 66),   # Green
            (66, 117, 245),   # Blue
            (245, 66, 117),   # Pink
            (245, 245, 66),   # Yellow
            (66, 245, 245),   # Cyan
            (245, 66, 245),   # Magenta
            (66, 245, 117),   # Light Green
            (117, 66, 245),   # Purple
            (245, 117, 245)   # Light Pink
        ]
        self.object_color = (66, 245, 200)
        
        # Keypoint connections for pose
        self.skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], 
                        [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], 
                        [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

    def draw_pose(self, frame, keypoints, color):
        """Draw pose skeleton on frame"""
        for p1, p2 in self.skeleton:
            # Get coordinates
            x1, y1 = int(keypoints[p1-1][0]), int(keypoints[p1-1][1])
            x2, y2 = int(keypoints[p2-1][0]), int(keypoints[p2-1][1])
            
            # Draw line if both points are detected
            if (x1, y1) != (0, 0) and (x2, y2) != (0, 0):
                cv2.line(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw keypoints
        for x, y, conf in keypoints:
            if conf > self.conf_threshold:
                cv2.circle(frame, (int(x), int(y)), 4, color, -1)

    def process_frame(self, frame):
        # Object Detection
        object_results = self.object_model(frame, verbose=False)[0]
        
        # Process object detections
        for box in object_results.boxes:
            confidence = float(box.conf[0])
            
            if confidence < self.conf_threshold:
                continue
            
            class_id = int(box.cls[0])
            class_name = self.object_model.names[class_id]
            
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.object_color, 2)
            cv2.putText(frame, f'{class_name}: {confidence:.2f}',
                       (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                       self.object_color, 2)
        
        # Pose Detection
        pose_results = self.pose_model(frame, verbose=False)[0]
        person_count = 0
        
        # Process pose detections
        for person in pose_results.keypoints:
            if person_count >= len(self.pose_colors):
                break
                
            color = self.pose_colors[person_count]
            self.draw_pose(frame, person.data[0], color)
            
            # Add person counter
            cv2.putText(frame, f'Person {person_count + 1}',
                       (10, 60 + 30 * person_count),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            person_count += 1
        
        # Add FPS and person count
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
                
                # Process frame
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

if __name__ == "__main__":
    try:
        detector = CombinedDetector(confidence_threshold=0.5)
        detector.run()
    except Exception as e:
        print(f"Error initializing detector: {str(e)}")