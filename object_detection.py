import cv2
import numpy as np
from ultralytics import YOLO
import time

class ObjectDetector:
    def __init__(self, confidence_threshold=0.5, width=1280, height=720):
        """
        Initialize the object detector
        confidence_threshold: minimum confidence score to consider a detection valid
        width, height: camera resolution
        """
        # Load the YOLO model - using YOLOv8 nano version for good balance of speed and accuracy
        self.model = YOLO('yolov8n.pt')
        
        # Set confidence threshold for detections
        self.conf_threshold = confidence_threshold
        
        # Initialize video capture (0 is default webcam)
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        # COCO dataset class names that YOLO can detect
        # YOLO is trained on COCO dataset which has 80 different object classes
        self.class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 
                           'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 
                           'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 
                           'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 
                           'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 
                           'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 
                           'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 
                           'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 
                           'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 
                           'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 
                           'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 
                           'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 
                           'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 
                           'scissors', 'teddy bear', 'hair drier', 'toothbrush']
        
        # Generate random colors for each class
        # Each object class will have its own consistent color
        self.colors = np.random.randint(0, 255, size=(len(self.class_names), 3))

    def draw_detection(self, frame, box, class_id, confidence):
        """
        Draw bounding box and label for each detected object
        frame: image to draw on
        box: bounding box coordinates (x1, y1, x2, y2)
        class_id: index of detected class
        confidence: detection confidence score
        """
        # Extract coordinates
        x1, y1, x2, y2 = map(int, box)
        
        # Get color for this class
        color = self.colors[class_id].tolist()
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Create label with class name and confidence
        label = f'{self.class_names[class_id]}: {confidence:.2f}'
        
        # Calculate label size for background rectangle
        (label_width, label_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
        )
        
        # Draw filled rectangle for label background
        cv2.rectangle(
            frame, 
            (x1, y1 - label_height - 10), 
            (x1 + label_width + 10, y1), 
            color, 
            -1  # -1 means filled rectangle
        )
        
        # Draw label text
        cv2.putText(
            frame, label, (x1 + 5, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
        )

    def process_frame(self, frame):
        """
        Process a single frame and detect objects
        frame: input image frame from camera
        returns: frame with detections drawn
        """
        # Run YOLO detection on the frame
        # task='detect' means we're doing object detection (as opposed to segmentation etc)
        results = self.model(frame, task='detect')[0]
        
        # Process each detection
        for box in results.boxes:
            # Get confidence score
            confidence = float(box.conf[0])
            
            # Only process detections above our confidence threshold
            if confidence < self.conf_threshold:
                continue
            
            # Get the predicted class ID
            class_id = int(box.cls[0])
            
            # Draw this detection on the frame
            self.draw_detection(frame, box.xyxy[0], class_id, confidence)
        
        # Add FPS counter
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        cv2.putText(
            frame, f'FPS: {int(fps)}',
            (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )
        
        return frame

    def run(self):
        """
        Main loop to continuously capture and process frames
        """
        print("Starting object detection...")
        print(f"Can detect {len(self.class_names)} different objects")
        print("Press 'q' to quit")
        
        try:
            while self.cap.isOpened():
                # Read frame from camera
                success, frame = self.cap.read()
                if not success:
                    print("Failed to grab frame")
                    break
                
                # Process frame and get detections
                processed_frame = self.process_frame(frame)
                
                # Show the processed frame
                cv2.imshow('Object Detection', processed_frame)
                
                # Break loop on 'q' press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except Exception as e:
            print(f"Error during detection: {str(e)}")
        finally:
            # Clean up
            self.cleanup()

    def cleanup(self):
        """
        Release resources
        """
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        # Create detector with custom settings
        detector = ObjectDetector(
            confidence_threshold=0.5,  # Minimum confidence score (0-1)
            width=1280,               # Camera width
            height=720                # Camera height
        )
        # Start detection
        detector.run()
    except Exception as e:
        print(f"Error initializing detector: {str(e)}")
