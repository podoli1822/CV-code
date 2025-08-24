from picamera2 import Picamera2
import cv2

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": "BGR888", "size": (640, 480)}))
picam2.start()

frame = picam2.capture_array()
cv2.imwrite("frame.jpg", frame)
print("Frame Saved")
