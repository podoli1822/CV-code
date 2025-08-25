from picamera2 import Picamera2
import time

picam2 = Picamera2()

camera_config = picam2.create_preview_configuration()
picam2.configure(camera_config)

picam2.start()

time.sleep(2)

picam2.capture_file("test_image.jpg")

print("사진 촬영 완료! 'test_image.jpg' 파일로 저장되었습니다.")

picam2.stop()
