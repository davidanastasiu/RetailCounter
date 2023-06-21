from ultralytics import YOLO
import cv2
model = YOLO('/home/arpita/runs/detect/yolov8m_custom2/weights/best.pt')
img_path = "000000000000.jpg"
img = cv2.imread(img_path)
model.predict(img, save=True)