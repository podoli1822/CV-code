import cv2
from picamera2 import Picamera2
import time
import os
from ultralytics import YOLO
from collections import defaultdict
import numpy as np

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

# --- Initial Settings for Optimization ---
# 1. Video Resolution
FRAME_WIDTH = 480
FRAME_HEIGHT = 400

# 2. Counting Line
LINE_Y_POSITION = 180

# 3. Model and Confidence Settings
MODEL_NAME = './tflite/best_full_integer_quant.tflite' 
CONFIDENCE_THRESHOLD = 0.2

# 4. YOLO Processing Size
YOLO_IMG_SIZE = 352

inf_interval = 2

# --- [NEW] Zoom Settings ---
ENABLE_ZOOM = True      # True: 줌 기능 켜기, False: 끄기
ZOOM_FACTOR = 1.5       # 배율 설정 (예: 1.5배 확대, 2.0배 확대)

# --------------------------------------------------------------------

def apply_digital_zoom(frame, factor):
    """
    프레임의 중앙을 기준으로 잘라내어(Crop) 원래 크기로 리사이즈합니다.
    """
    if factor <= 1.0:
        return frame

    h, w = frame.shape[:2]
    
    # 1. 확대할 새로운 너비와 높이 계산 (중심 기준)
    new_h, new_w = int(h / factor), int(w / factor)

    # 2. 잘라낼 시작 좌표 계산
    y1 = (h - new_h) // 2
    x1 = (w - new_w) // 2
    y2 = y1 + new_h
    x2 = x1 + new_w

    # 3. 이미지 Crop (자르기)
    cropped_frame = frame[y1:y2, x1:x2]

    # 4. 원래 해상도로 Resize (늘리기) - 선형 보간법 사용
    # 리사이즈를 해야 LINE_Y_POSITION 좌표계가 유지되고 화면에 꽉 차게 보입니다.
    resized_frame = cv2.resize(cropped_frame, (w, h), interpolation=cv2.INTER_LINEAR)
    
    return resized_frame

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

    # (주의) 원본 코드 로직 유지: frame_counter가 while문 안에서 초기화되어 실제로는 매 프레임 동작 중임
    # 추후 프레임 스킵이 필요하면 frame_counter 변수 선언을 while문 밖으로 빼야 합니다.
    
    while True:
        frame_counter = True 
        
        # 4. Capture a frame
        frame = picam2.capture_array()
        
        # Rotate logic
        frame = np.ascontiguousarray(np.rot90(frame))

        # --- [NEW] Apply Zoom Logic ---
        if ENABLE_ZOOM:
            frame = apply_digital_zoom(frame, ZOOM_FACTOR)
        # ------------------------------

        # 5. Perform object tracking
        if (frame_counter):
            # frame이 이미 확대되었으므로, 모델은 확대된 이미지를 받아 처리합니다.
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
                track.append(center_y)
                if len(track) > 30:
                    track.pop(0)

                # 8. Check for line crossing
                if len(track) > 1:
                    prev_y = track[-2]
                    
                    if prev_y < LINE_Y_POSITION and center_y >= LINE_Y_POSITION:
                        in_count += 1
                        track_history.pop(track_id, None) 
                    
                    elif prev_y > LINE_Y_POSITION and center_y <= LINE_Y_POSITION:
                        out_count += 1
                        track_history.pop(track_id, None)
            
        
        # Draw counting line and info text
        # (주의) 화면이 확대되었으므로 LINE_Y_POSITION의 '물리적 위치'는 달라지지만
        # 화면상 픽셀 위치는 그대로 유지됩니다.
        cv2.line(annotated_frame, (0, LINE_Y_POSITION), (FRAME_WIDTH, LINE_Y_POSITION), (0, 0, 255), 2)
        
        zoom_text = f"Zoom: x{ZOOM_FACTOR}" if ENABLE_ZOOM else "Zoom: Off"
        info_text = f"In: {in_count} / Out: {out_count} | {zoom_text}"
        
        cv2.putText(annotated_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Display the resulting frame
        cv2.imshow("YOLO People Counter", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        if(frame_counter):
            frame_counter = False
        else:
            frame_counter = True

    # Cleanup on exit
    picam2.stop()
    cv2.destroyAllWindows()
    print(f"Final Count -> In: {in_count}, Out: {out_count}")

if __name__ == "__main__":
    main()
