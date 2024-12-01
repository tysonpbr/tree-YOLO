from ultralytics import YOLO

model = YOLO('yolov8n-seg.pt')

batch_size = 4
num_epochs = 20
learning_rate = 0.001

model.train(
    data='config.yaml',
    epochs=num_epochs,
    imgsz=640,
    batch=batch_size,
    lr0=learning_rate
)
