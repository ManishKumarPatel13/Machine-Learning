from ultralytics import YOLO
import cv2

model = YOLO("yolov10n.pt")
results = model.track(1, save=True, show=True,stream=True, conf=0.2)