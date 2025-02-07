import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime

class PoseDetectionCamera:
    def __init__(self, width=1920, height=1080, fps=30):
        self.width = width
        self.height = height
        self.fps = fps
        
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)  # Use default camera
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        
    def process_frame(self, frame):
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame and detect poses
        results = self.pose.process(rgb_frame)    # This finds all the landmarks in the image
        
        # Draw pose landmarks
        if results.pose_landmarks:
            self.mp_draw.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_draw.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                connection_drawing_spec=self.mp_draw.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )
        
        return frame
    
    def run(self):
        try:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                
                # Process frame with pose detection
                processed_frame = self.process_frame(frame)
                
                # Display frame
                cv2.imshow('Pose Detection', processed_frame)
                
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
        self.pose.close()

if __name__ == "__main__":
    try:
        # Create and run camera instance
        camera = PoseDetectionCamera(
            width=1920,
            height=1080,
            fps=30
        )
        print("Starting pose detection camera...")
        print("Press 'q' to quit")
        camera.run()
    except Exception as e:
        print(f"Error initializing camera: {str(e)}")