import cv2
from picamera2 import Picamera2
import time
from ultralytics import YOLO
from collections import defaultdict

# --- Initial Settings (★★★★★ ADJUST THESE VALUES FOR YOUR ENVIRONMENT ★★★★★) ---
# 1. Video Resolution
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# 2. Counting Line (Y-coordinate)
LINE_Y_POSITION = 240

# 3. Model and Confidence Settings
# Use a more powerful model for better top-down view detection.
# 'yolov8s.pt' (small) is a good balance of accuracy and speed for RPi 5.
# If 'yolov8s.pt' is too slow, you can switch back to 'yolov8n.pt'.
MODEL_NAME = 'yolov8s.pt' 

# Lower the confidence threshold to detect people even if the model is not very sure.
# Default is 0.25. Start with 0.2 or 0.15.
CONFIDENCE_THRESHOLD = 0.2
# --------------------------------------------------------------------

def main():
    # 1. Initialize Picamera2
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "RGB888"})
    picam2.configure(config)
    picam2.start()
    time.sleep(1) # Wait for the camera to stabilize

    # 2. Load the specified YOLO model
    model = YOLO(MODEL_NAME)
    print(f"YOLO model '{MODEL_NAME}' loaded. Starting person counting.")

    # 3. Initialize variables
    track_history = defaultdict(lambda: [])
    in_count = 0
    out_count = 0

    while True:
        # 4. Capture a frame from the camera
        frame = picam2.capture_array()

        # 5. Perform object tracking for 'person' class with the custom confidence threshold
        # 'conf=CONFIDENCE_THRESHOLD' is the key change to apply our setting.
        results = model.track(frame, persist=True, verbose=False, classes=0, conf=CONFIDENCE_THRESHOLD)

        # Draw the bounding boxes and tracking IDs on the frame.
        annotated_frame = results[0].plot()

        # Check if any people are being tracked
        if results[0].boxes.id is not None:
            # 6. Get tracking data for the detected people
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            # 7. Iterate over each tracked person
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                center_x = int(x)
                center_y = int(y)

                # Append the person's center point to their tracking history
                track = track_history[track_id]
                track.append((center_x, center_y))
                if len(track) > 30:
                    track.pop(0)

                # 8. Check if the person has crossed the counting line
                if len(track) > 1:
                    prev_y = track[-2][1]
                    
                    if prev_y < LINE_Y_POSITION and center_y >= LINE_Y_POSITION:
                        in_count += 1
                        track_history[track_id] = []
                    
                    elif prev_y > LINE_Y_POSITION and center_y <= LINE_Y_POSITION:
                        out_count += 1
                        track_history[track_id] = []
        
        # Draw the counting line and info text
        cv2.line(annotated_frame, (0, LINE_Y_POSITION), (FRAME_WIDTH, LINE_Y_POSITION), (0, 0, 255), 2)
        info_text = f"In: {in_count} / Out: {out_count}"
        cv2.putText(annotated_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Display the resulting frame
        cv2.imshow("YOLO People Counter", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup on exit
    picam2.stop()
    cv2.destroyAllWindows()
    print(f"Final Count -> In: {in_count}, Out: {out_count}")

if __name__ == "__main__":
    main()
