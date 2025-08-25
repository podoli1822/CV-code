import cv2
from picamera2 import Picamera2
import time
from ultralytics import YOLO
from collections import defaultdict

# --- 초기 설정 값 (★★★★★ 사용 환경에 맞게 이 부분을 수정해야 합니다 ★★★★★) ---
# 1. 영상 해상도 설정
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# 2. 출입 감지 기준선 (Y 좌표)
#    카메라 영상의 중앙보다 약간 아래 또는 위로 설정합니다.
LINE_Y_POSITION = 240
# --------------------------------------------------------------------

def main():
    # 1. Picamera2 초기화
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "RGB888"})
    picam2.configure(config)
    picam2.start()
    time.sleep(1)

    # 2. YOLO 모델 로드
    model = YOLO('yolov8n.pt')
    print("YOLO 모델 로드 완료. 인원 계수를 시작합니다.")

    # 3. 변수 초기화
    # 객체의 이동 경로를 저장하기 위한 딕셔너리
    track_history = defaultdict(lambda: [])
    
    # 카운트 변수
    in_count = 0
    out_count = 0

    while True:
        # 4. 카메라에서 프레임 캡처
        frame = picam2.capture_array()

        # 5. YOLO 모델로 객체 "추적" 수행
        # model.track()을 사용하면 각 객체에 고유 ID가 부여됩니다.
        # persist=True는 프레임 간 추적 정보를 유지하라는 의미입니다.
        results = model.track(frame, persist=True)

        # 탐지된 객체가 있을 경우에만 boxes와 track_ids를 가져옵니다.
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu() # x, y, 너비, 높이
            track_ids = results[0].boxes.id.int().cpu().tolist()

            # 6. 추적된 각 객체에 대해 처리
            for box, track_id in zip(boxes, track_ids):
                # 사람(클래스 0)이 아닌 경우 무시 (필요 시 주석 해제)
                # obj_cls = int(results[0].boxes[track_ids.index(track_id)].cls)
                # if obj_cls != 0:
                #     continue

                x, y, w, h = box
                center_x = int(x)
                center_y = int(y)

                # 객체의 중심점 좌표를 추적 기록에 추가
                track = track_history[track_id]
                track.append((center_x, center_y))
                # 이동 경로가 너무 길어지지 않도록 일정 길이 유지
                if len(track) > 30:
                    track.pop(0)

                # 7. 기준선 통과 여부 확인
                # 이동 경로에 2개 이상의 점이 기록되어 있어야 방향을 알 수 있습니다.
                if len(track) > 1:
                    prev_y = track[-2][1] # 바로 이전 프레임의 y좌표
                    
                    # 기준선 위 -> 아래로 통과 (입장)
                    if prev_y < LINE_Y_POSITION and center_y >= LINE_Y_POSITION:
                        # 이미 카운트된 객체가 다시 카운트되는 것을 방지하기 위해,
                        # 한 번 통과하면 해당 객체는 기록을 초기화합니다.
                        in_count += 1
                        print(f"사람 들어옴 (ID: {track_id}). 총 입장: {in_count}")
                        track_history[track_id] = [] # 기록 초기화
                    
                    # 기준선 아래 -> 위로 통과 (퇴장)
                    elif prev_y > LINE_Y_POSITION and center_y <= LINE_Y_POSITION:
                        out_count += 1
                        print(f"사람 나감 (ID: {track_id}). 총 퇴장: {out_count}")
                        track_history[track_id] = [] # 기록 초기화

                # 8. 화면에 시각적 요소 그리기
                # 객체 바운딩 박스 (YOLOv8의 plot() 함수를 사용하면 더 편함)
                annotated_frame = results[0].plot()

        else: # 탐지된 객체가 없을 경우 원본 프레임 사용
            annotated_frame = frame

        # 기준선 그리기
        cv2.line(annotated_frame, (0, LINE_Y_POSITION), (FRAME_WIDTH, LINE_Y_POSITION), (0, 0, 255), 2)
        
        # 카운트 정보 표시
        info_text = f"In: {in_count} / Out: {out_count}"
        cv2.putText(annotated_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # 결과 화면 출력
        cv2.imshow("YOLO People Counter", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    picam2.stop()
    cv2.destroyAllWindows()
    print(f"최종 집계 -> 입장: {in_count}, 퇴장: {out_count}")

if __name__ == "__main__":
    main()
