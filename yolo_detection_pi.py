import cv2
from picamera2 import Picamera2
import time
from ultralytics import YOLO

# --- Initial Settings ---
# Set video resolution
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

def main():
    # 1. Initialize Picamera2
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "RGB888"})
    picam2.configure(config)
    picam2.start()
    time.sleep(1) # Wait for the camera to stabilize

    # 2. Load the YOLO model
    # 'yolov8n.pt' is the smallest and fastest model, suitable for Raspberry Pi.
    # The model file will be downloaded automatically on the first run.
    model = YOLO('yolov8n.pt')
    print("YOLO model loaded successfully. Starting person detection...")

    while True:
        # 3. Capture a frame from the camera
        frame = picam2.capture_array()

        # 4. Perform object detection with the YOLO model
        # The model() function returns a list of detected objects.
        results = model(frame)

        # 5. Process detection results
        # results[0].boxes contains information about all detected objects.
        for box in results[0].boxes:
            # Check the class ID ('person' is class ID 0 in the COCO dataset)
            if box.cls == 0:
                # box.xyxy[0] contains coordinates in the format [x1, y1, x2, y2].
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Draw a green rectangle around the person
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Display the class name and confidence score
                confidence = box.conf[0]
                label = f"Person: {confidence:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 6. Display the resulting frame (requires Raspberry Pi Desktop environment)
        cv2.imshow("YOLO Person Detection", frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup on exit
    picam2.stop()
    cv2.destroyAllWindows()
    print("Program terminated.")

if __name__ == "__main__":
    main()
