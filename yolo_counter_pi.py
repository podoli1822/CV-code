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
# Set this line somewhere in the middle of the frame to detect crossing.
LINE_Y_POSITION = 240
# --------------------------------------------------------------------

def main():
    # 1. Initialize Picamera2
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "RGB888"})
    picam2.configure(config)
    picam2.start()
    time.sleep(1) # Wait for the camera to stabilize

    # 2. Load the YOLO model
    model = YOLO('yolov8n.pt')
    print("YOLO model loaded. Starting people counting.")

    # 3. Initialize variables
    # A dictionary to store the tracking history of objects
    track_history = defaultdict(lambda: [])
    
    # Counter variables
    in_count = 0
    out_count = 0

    while True:
        # 4. Capture a frame from the camera
        frame = picam2.capture_array()

        # 5. Perform object tracking with the YOLO model
        # persist=True keeps track information between frames.
        # verbose=False stops printing detailed logs for every frame.
        results = model.track(frame, persist=True, verbose=False)

        # Draw the bounding boxes and tracking IDs on the frame
        annotated_frame = results[0].plot()

        # Check if any objects are being tracked
        if results[0].boxes.id is not None:
            
            # 6. Iterate over each tracked object
            for box in results[0].boxes:
                # --- THIS IS THE KEY CHANGE: Only process objects that are people (class ID 0) ---
                if box.cls == 0:
                    track_id = int(box.id)
                    
                    # Get the center point of the bounding box
                    x1, y1, x2, y2 = box.xyxy[0]
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)

                    # Append the object's center point to its tracking history
                    track = track_history[track_id]
                    track.append((center_x, center_y))
                    # Keep the track history to a manageable length
                    if len(track) > 30:
                        track.pop(0)

                    # 7. Check if the person has crossed the counting line
                    # We need at least two points in the track to determine direction.
                    if len(track) > 1:
                        prev_y = track[-2][1]  # The y-coordinate from the previous frame
                        
                        # Crossing from top to bottom (counts as 'In')
                        if prev_y < LINE_Y_POSITION and center_y >= LINE_Y_POSITION:
                            in_count += 1
                            # The line below is commented out to keep the terminal clean.
                            # Uncomment for debugging.
                            # print(f"Person In (ID: {track_id}). Total In: {in_count}")
                            track_history[track_id] = [] # Reset track to prevent double counting
                        
                        # Crossing from bottom to top (counts as 'Out')
                        elif prev_y > LINE_Y_POSITION and center_y <= LINE_Y_POSITION:
                            out_count += 1
                            # The line below is commented out to keep the terminal clean.
                            # Uncomment for debugging.
                            # print(f"Person Out (ID: {track_id}). Total Out: {out_count}")
                            track_history[track_id] = [] # Reset track to prevent double counting
        
        # Draw the counting line on the frame
        cv2.line(annotated_frame, (0, LINE_Y_POSITION), (FRAME_WIDTH, LINE_Y_POSITION), (0, 0, 255), 2)
        
        # Display the count information on the frame
        info_text = f"In: {in_count} / Out: {out_count}"
        cv2.putText(annotated_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Display the resulting frame
        cv2.imshow("YOLO People Counter", annotated_frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup on exit
    picam2.stop()
    cv2.destroyAllWindows()
    print(f"Final Count -> In: {in_count}, Out: {out_count}")

if __name__ == "__main__":
    main()
