import cv2
from picamera2 import Picamera2
import time
from ultralytics import YOLO
from collections import defaultdict

# --- Initial Settings for Optimization (★★★★★ ADJUST THESE VALUES ★★★★★) ---
# 1. Video Resolution
FRAME_WIDTH = 480
FRAME_HEIGHT = 360

# 2. Counting Line
LINE_Y_POSITION = 180

# 3. Model and Confidence Settings
MODEL_NAME = 'best.pt' 

# Adjust the threshold based on your custom model's performance. 
# Starting between 0.3 and 0.5 is recommended.
CONFIDENCE_THRESHOLD = 0.3

# 4. YOLO Processing Size
YOLO_IMG_SIZE = 320
# --------------------------------------------------------------------

def main():
    # 1. Initialize Picamera2
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "RGB888"})
    picam2.configure(config)
    picam2.start()
    time.sleep(1) # Wait for the camera to stabilize

    # 2. Load the custom YOLO model
    model = YOLO(MODEL_NAME)
    print(f"YOLO Custom model '{MODEL_NAME}' loaded. Starting person counting in optimized mode.")

    # 3. Initialize variables
    track_history = defaultdict(lambda: [])
    in_count = 0
    out_count = 0

    while True:
        # 4. Capture a frame
        frame = picam2.capture_array()

        # 5. Perform object tracking
        # 'classes=0' refers to the first class in your custom model's dataset.
        # If you only trained one class (e.g., 'head'), this is correct.
        results = model.track(frame, persist=True, verbose=False, classes=0, conf=CONFIDENCE_THRESHOLD, imgsz=YOLO_IMG_SIZE)

        # Draw bounding boxes and tracking IDs
        annotated_frame = results[0].plot()

        if results[0].boxes.id is not None:
            # 6. Get tracking data
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            # 7. Iterate over each tracked object
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                center_y = int(y)

                track = track_history[track_id]
                # Storing only the Y-coordinate is sufficient for line crossing and saves memory.
                track.append(center_y)
                if len(track) > 30:
                    track.pop(0)

                # 8. Check for line crossing
                if len(track) > 1:
                    prev_y = track[-2] # Get the Y-coordinate from the previous frame
                    
                    if prev_y < LINE_Y_POSITION and center_y >= LINE_Y_POSITION:
                        in_count += 1
                        # Remove the track history after counting to prevent double counts
                        track_history.pop(track_id, None) 
                    
                    elif prev_y > LINE_Y_POSITION and center_y <= LINE_Y_POSITION:
                        out_count += 1
                        # Remove the track history after counting to prevent double counts
                        track_history.pop(track_id, None)
        
        # Draw counting line and info text
        cv2.line(annotated_frame, (0, LINE_Y_POSITION), (FRAME_WIDTH, LINE_Y_POSITION), (0, 0, 255), 2)
        info_text = f"In: {in_count} / Out: {out_count}"
        cv2.putText(annotated_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

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
