import cv2
import numpy as np
from picamera2 import Picamera2
import time

# --- 초기 설정 값 (★★★★★ 사용 환경에 맞게 이 부분을 수정해야 합니다 ★★★★★) ---

# 1. 영상 해상도 설정
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# 2. 출입 감지 기준선 (Y 좌표)
#    카메라 영상의 중앙보다 약간 아래 또는 위로 설정합니다.
#    사람(객체)의 중심점이 이 선을 통과할 때 카운트됩니다.
LINE_Y_POSITION = 240 

# 3. 객체 감지 최소 크기 (픽셀 단위 면적)
#    너무 작은 움직임(노이즈, 그림자 등)을 무시하기 위한 값입니다.
#    실제 사람 머리가 감지될 때의 크기를 보고 조정해야 합니다.
MIN_CONTOUR_AREA = 300

# 4. 객체 추적 관련 설정
#    객체의 중심점이 이 거리(픽셀) 내에 있으면 같은 객체로 판단합니다.
MAX_DISTANCE_THRESHOLD = 50 
#    객체가 이 프레임 수만큼 보이지 않으면 추적 목록에서 삭제합니다.
MAX_FRAMES_TO_DISAPPEAR = 10
# --------------------------------------------------------------------

# 추적 중인 사람(객체)을 관리하는 클래스
class PersonTracker:
    def __init__(self, person_id, center_point):
        self.id = person_id
        self.center_points = [center_point]
        self.disappeared_frames = 0
        self.counted = False

    def get_last_center(self):
        return self.center_points[-1]

# 메인 함수
def main():
    # Picamera2 초기화
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "RGB888"})
    picam2.configure(config)
    picam2.start()
    time.sleep(1) # 카메라가 안정될 때까지 잠시 대기

    # 배경 제거 객체 생성 (MOG2 방식)
    # detectShadows=False로 설정하면 그림자를 객체로 인식하지 않아 더 좋습니다.
    background_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)

    # 변수 초기화
    tracked_persons = {}
    next_person_id = 0
    in_count = 0
    out_count = 0

    print("카메라 스트림 시작. 인원 계수를 시작합니다...")
    print(f"기준선 Y좌표: {LINE_Y_POSITION}, 최소 객체 크기: {MIN_CONTOUR_AREA}")

    while True:
        # 카메라에서 프레임 캡처
        frame = picam2.capture_array()
        
        # 1. 배경 제거
        fg_mask = background_subtractor.apply(frame)

        # 2. 노이즈 제거 (Morphology)
        # 마스크의 작은 구멍을 메우고, 작은 점들을 제거합니다.
        kernel = np.ones((5, 5), np.uint8)
        fg_mask = cv2.erode(fg_mask, kernel, iterations=1)
        fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)
        
        # 3. 객체(Contour) 찾기
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        current_detected_centers = []
        for contour in contours:
            # 설정한 최소 크기보다 작은 객체는 무시
            if cv2.contourArea(contour) < MIN_CONTOUR_AREA:
                continue

            # 객체의 경계 상자를 그리고 중심점 계산
            (x, y, w, h) = cv2.boundingRect(contour)
            center_x = int(x + w / 2)
            center_y = int(y + h / 2)
            current_detected_centers.append((center_x, center_y))
            
            # 영상에 경계 상자와 중심점 그리기
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(frame, (center_x, center_y), 4, (0, 0, 255), -1)

        # 4. 객체 추적 및 카운팅
        
        # 현재 프레임에서 감지된 객체들과 이전에 추적하던 객체들을 매칭
        unmatched_persons = list(tracked_persons.keys())
        
        for center in current_detected_centers:
            matched_person_id = None
            min_dist = MAX_DISTANCE_THRESHOLD
            
            for person_id, person in tracked_persons.items():
                dist = np.linalg.norm(np.array(center) - np.array(person.get_last_center()))
                if dist < min_dist:
                    min_dist = dist
                    matched_person_id = person_id
            
            if matched_person_id is not None:
                # 기존 객체와 매칭됨 -> 정보 업데이트
                tracked_persons[matched_person_id].center_points.append(center)
                tracked_persons[matched_person_id].disappeared_frames = 0
                if matched_person_id in unmatched_persons:
                    unmatched_persons.remove(matched_person_id)
            else:
                # 새로운 객체 발견 -> 새로 등록
                new_person = PersonTracker(next_person_id, center)
                tracked_persons[next_person_id] = new_person
                next_person_id += 1

        # 일정 시간 보이지 않은 객체는 목록에서 삭제
        for person_id in unmatched_persons:
            tracked_persons[person_id].disappeared_frames += 1
            if tracked_persons[person_id].disappeared_frames > MAX_FRAMES_TO_DISAPPEAR:
                del tracked_persons[person_id]
        
        # 기준선 통과 여부 확인하여 카운트
        for person_id, person in tracked_persons.items():
            # 객체의 이전 위치와 현재 위치
            if len(person.center_points) >= 2:
                prev_y = person.center_points[-2][1]
                current_y = person.center_points[-1][1]

                # 기준선을 통과했는지, 그리고 아직 카운트되지 않았는지 확인
                if (prev_y < LINE_Y_POSITION and current_y >= LINE_Y_POSITION) and not person.counted:
                    out_count += 1
                    person.counted = True
                    print(f"사람 나감. 총 나간 인원: {out_count}")
                elif (prev_y > LINE_Y_POSITION and current_y <= LINE_Y_POSITION) and not person.counted:
                    in_count += 1
                    person.counted = True
                    print(f"사람 들어옴. 총 들어온 인원: {in_count}")
        
        # 5. 화면에 정보 표시
        # 기준선 그리기
        cv2.line(frame, (0, LINE_Y_POSITION), (FRAME_WIDTH, LINE_Y_POSITION), (255, 0, 0), 2)
        
        # 카운트 정보 표시
        info_text = f"In: {in_count} / Out: {out_count}"
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # 화면에 결과 영상 출력
        # 라즈베리 파이 데스크탑 환경에서 실행 시 주석 해제
        # cv2.imshow("People Counter", frame)
        
        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 종료 처리
    # cv2.destroyAllWindows()
    picam2.stop()
    print("프로그램을 종료합니다.")
    print(f"최종 집계 -> 들어온 인원: {in_count}, 나간 인원: {out_count}")

if __name__ == "__main__":
    main()
