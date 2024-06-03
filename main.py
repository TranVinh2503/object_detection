import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox

# Mở webcam
webcam = cv2.VideoCapture(0)

if not webcam.isOpened():
    print("Không thể mở webcam")
    exit()
    
while webcam.isOpened():
    # Đọc frame từ webcam
    status, frame = webcam.read()
    if not status:
        break

    bbox, label, conf = cv.detect_common_objects(frame,model="yolov3-tiny")
    output_image = draw_bbox(frame, bbox, label, conf)
    
    # Hiển thị kết quả
    cv2.imshow("Real-time object detection", output_image)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Đóng webcam và cửa sổ
webcam.release()
cv2.destroyAllWindows()