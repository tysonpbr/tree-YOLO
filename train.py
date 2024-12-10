from ultralytics import YOLO
import os
import torch

if __name__ == '__main__':
    batch_sizes = [4, 8, 16, 32]
    learning_rates = [0.001, 0.005, 0.01, 0.02]

    num_epochs = 50
    momentum = 0.9
    img_size = 512

    output_dir = "trained_models"
    os.makedirs(output_dir, exist_ok=True)

    for batch_size in batch_sizes:
        for learning_rate in learning_rates:
            model = YOLO('yolov8n-seg.pt')
            
            print(f"Training with batch size: {batch_size}, learning rate: {learning_rate}")
            
            model.train(
                data='config.yaml',
                optimizer="SGD",
                device=0,
                momentum=momentum,
                epochs=num_epochs,
                imgsz=img_size,
                batch=batch_size,
                lr0=learning_rate
            )
            
            model_path = os.path.join(
                output_dir, 
                f"model_bs{batch_size}_lr{learning_rate}.pt"
            )
            model.export(format="torchscript", path=model_path)
            print(f"Model saved as {model_path}")
