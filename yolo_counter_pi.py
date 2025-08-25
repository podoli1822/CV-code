# detecting people that enter and exiting from librarys in the Technion.

import os
import ast
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# Picamera2와 time 라이브러리를 import 합니다.
from picamera2 import Picamera2
import time

def run():
    '''
    Initialize counter class and run counting loop.
    '''
    import sys
    import cv2

    from util.logger import get_logger
    from FrameProcessor import FrameProcessor
    from util.debugger import mouse_callback, take_screenshot
    from keras import backend as K
    logger = get_logger()

    # IGNORE WARNINGS:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # ### Picamera2 초기화 부분 (수정됨) ###
    FRAME_WIDTH = 1280 # 카메라가 지원하는 해상도로 설정 필요
    FRAME_HEIGHT = 720 # 카메라가 지원하는 해상도로 설정 필요
    
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "RGB888"})
    picam2.configure(config)
    picam2.start()
    time.sleep(1) # 카메라 안정화 대기
    print("Picamera2 started successfully.")
    
    # 첫 프레임을 가져와서 너비와 높이 설정
    frame = picam2.capture_array()
    f_height, f_width, _ = frame.shape
    
    # ####load configuration from the env file#####
    # ##detection configuratio###
    detection_slowdown = ast.literal_eval(os.getenv('DETECTION_SLOWDOWN'))
    detection_interval = int(os.getenv('DI'))
    mcdf = int(os.getenv('MCDF'))
    detector = os.getenv('DETECTOR')

    # create detection region of interest polygon#
    use_droi = ast.literal_eval(os.getenv('USE_DROI'))
    droi = ast.literal_eval(os.getenv('DROI')) \
            if use_droi \
            else [(0, 0), (f_width, 0), (f_width, f_height), (0, f_height)]
    show_droi = ast.literal_eval(os.getenv('SHOW_DROI'))

    # confidence threshold of detection#
    confidence_threshold = float(os.getenv("CONFIDENCE_THRESHOLD"))
    
    # (이하 원본 코드와 동일 ... )
    sensitive_confidence_threshold = float(os.getenv("SENSITIVE_CONFIDENCE_THRESHOLD"))
    mctf = int(os.getenv('MCTF'))
    tracker = os.getenv('TRACKER')
    duplicate_object_threshold = float(os.getenv('OVERLAP_THRESHOLD'))
    use_counting_roi = ast.literal_eval(os.getenv('USE_COUNT_ROI'))
    counting_roi = ast.literal_eval(os.getenv('COUNTING_ROI')) if use_counting_roi else None
    show_roi_counting = ast.literal_eval(os.getenv('SHOW_COUNT_ROI'))
    counting_roi_outside = ast.literal_eval(os.getenv('COUNTING_ROI_OUTSIDE'))
    counting_line_orientation = os.getenv('COUNTING_LINE_ORIENTATION')
    counting_line_position = float(os.getenv('COUNTING_LINE_POSITION'))
    use_object_liveness = ast.literal_eval(os.getenv('ENABLE_OBJECT_LIVENESS'))
    roi_object_liveness = ast.literal_eval(os.getenv('OBJECT_LIVENESS_ROI')) if use_object_liveness else None
    show_object_liveness = ast.literal_eval(os.getenv('SHOW_OBJECT_LIVENESS'))
    frame_number_counting_color = int(os.getenv('COLOR_CHANGE_INTERVAL_FOR_COUNTING_LINE'))
    event_api_url = os.getenv('EVENT_API_URL')

    # ##output configuration###
    record = ast.literal_eval(os.getenv('RECORD'))
    UI = ast.literal_eval(os.getenv('UI'))
    debug = ast.literal_eval(os.getenv('DEBUG'))
    # ####### create people counter obejct ########
    people_counter = FrameProcessor(frame, tracker, droi, show_droi, mcdf,
                                     mctf, detection_interval, counting_line_orientation, counting_line_position,
                                   show_roi_counting, counting_roi, counting_roi_outside, frame_number_counting_color,
                                   detection_slowdown, roi_object_liveness, show_object_liveness, confidence_threshold, sensitive_confidence_threshold,
                                   duplicate_object_threshold, event_api_url)

    # ( ... 원본 코드와 동일 ...)
    if record:
        output_name ="output.avi"
        output_video = cv2.VideoWriter(os.getenv('OUTPUT_VIDEO_PATH') + output_name,
                                       cv2.VideoWriter_fourcc(*'MJPG'),
                                        30, (f_width, f_height))

    logger.info('Processing started.')
    
    cv2.namedWindow('Debug')
    cv2.setMouseCallback('Debug', mouse_callback, {'frame_width': f_width, 'frame_height': f_height})

    is_paused = False
    output_frame = None
    start_time = time.time()

    # ##########main loop ###############
    # while 루프 조건은 항상 참(True)으로 변경하여 계속 실행되도록 함
    while True:
        if debug:
            k = cv2.waitKey(1) & 0xFF
            if k == ord('p'):
                is_paused = not is_paused
            if k == ord('s') and output_frame is not None:
                take_screenshot(output_frame)
            if k == ord('q'):
                logger.info('Loop stopped.', extra={'meta': {'cat': 'COUNT_PROCESS'}})
                break

        if is_paused:
            time.sleep(0.5)
            continue
        
        # ### Picamera2로부터 프레임 읽기 (수정됨) ###
        frame = picam2.capture_array()
        
        # count people and show it in the video
        people_counter.track_and_detect(frame)
        output_frame = people_counter.visualize()

        if record:
             output_video.write(output_frame)

        if UI:
             debug_window_size = ast.literal_eval(os.getenv('DEBUG_WINDOW_SIZE'))
             resized_frame = cv2.resize(output_frame, debug_window_size)
             cv2.imshow('Debug', resized_frame)

    end_time = time.time()
    total_time = str(end_time-start_time)
    print("total time = " + total_time)
    print("total in : {0} \n total out {1}\n".format(str(people_counter.person_count_in), str(people_counter.person_count_out)))
    
    # ### 종료 처리 (수정됨) ###
    picam2.stop() # 카메라 정지
    if UI:
        cv2.destroyAllWindows()
    if record:
        output_video.release()
    K.clear_session()
    return people_counter.person_count_in, people_counter.person_count_out, total_time, people_counter.count_order

# (이하 원본 코드와 동일)
if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv(dotenv_path="./env.env")
    from util.logger import init_logger
    init_logger()
    run()
