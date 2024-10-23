from ultralytics import YOLO

model = YOLO('yolov8n-seg.pt')

model.train(data='config.yaml', epochs=64, imgsz=640)
