# detecting people that enter and exiting from librarys in the Technion.

import os
import ast
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# Import the Picamera2 and time libraries
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

    # ### Picamera2 Initialization (Modified Section) ###
    FRAME_WIDTH = 1280 # You may need to adjust this to a resolution your camera supports
    FRAME_HEIGHT = 720 # You may need to adjust this to a resolution your camera supports
    
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "RGB888"})
    picam2.configure(config)
    picam2.start()
    time.sleep(1) # Wait for the camera to stabilize
    print("Picamera2 started successfully.")
    
    # Get the first frame to set width and height variables
    frame = picam2.capture_array()
    f_height, f_width, _ = frame.shape
    
    # ####load configuration from the env file#####
    # ##detection configuration###
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
    
    # ( ... rest of the configuration is the same as the original code ... )
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
    # ####### create people counter object ########
    people_counter = FrameProcessor(frame, tracker, droi, show_droi, mcdf,
                                     mctf, detection_interval, counting_line_orientation, counting_line_position,
                                   show_roi_counting, counting_roi, counting_roi_outside, frame_number_counting_color,
                                   detection_slowdown, roi_object_liveness, show_object_liveness, confidence_threshold, sensitive_confidence_threshold,
                                   duplicate_object_threshold, event_api_url)

    # ( ... video writer setup is the same as original ... )
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
    # The while loop condition is changed to True for a continuous camera stream
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
        
        # ### Read frame from Picamera2 (Modified Section) ###
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
    
    # ### Cleanup process (Modified Section) ###
    picam2.stop() # Stop the camera
    if UI:
        cv2.destroyAllWindows()
    if record:
        output_video.release()
    K.clear_session()
    return people_counter.person_count_in, people_counter.person_count_out, total_time, people_counter.count_order

# (The rest of the file is the same as the original)
if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv(dotenv_path="./env.env")
    from util.logger import init_logger
    init_logger()
    run()
