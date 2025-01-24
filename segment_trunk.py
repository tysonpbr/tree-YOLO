import os
import torch
from sklearn.metrics import jaccard_score, accuracy_score
from ultralytics import YOLO
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torchvision.models.segmentation as models
from model import U2NET, U2NETP
import cv2

test_images_path = "data/test/real/"

results_path = "results_real/"

results_path_yolov8 = "results_real/yolov8"
results_path_u2net = "results_real/u2net"
results_path_deeplabv3 = "results_real/deeplabv3"

# Ensure results directories exist
os.makedirs(results_path_yolov8, exist_ok=True)
os.makedirs(results_path_u2net, exist_ok=True)
os.makedirs(results_path_deeplabv3, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_paths = {
    "YOLOv8": "models/yolov8_model.pt",
    "U2Net": "models/u2net_model.pth",
    "DeepLabv3": "models/deeplabv3_model.pth"
}

def resize_and_pad(image, target_size=512):
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized_image = cv2.resize(image, (new_w, new_h))
    
    delta_w = target_size - new_w
    delta_h = target_size - new_h
    top, bottom = delta_h // 2, delta_h - delta_h // 2
    left, right = delta_w // 2, delta_w - delta_w // 2
    
    padded_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return padded_image, (top, bottom, left, right, (h, w))

def resize_mask_to_original(mask, padding_info):
    top, bottom, left, right, original_size = padding_info
    mask = mask[top:512-bottom, left:512-right]  # Remove padding
    mask_resized = cv2.resize(mask, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)
    return mask_resized

def getResultsYolov8(path):
    model = YOLO(path)
    test_images = sorted(os.listdir(test_images_path))

    for image_name in test_images:
        image_path = os.path.join(test_images_path, image_name)
        image = cv2.imread(image_path)
        image, padding_info = resize_and_pad(image)

        results = model(image)
        
        for result in results:
            if result.masks:
                for j, mask in enumerate(result.masks.data):
                    if j == 0:
                        mask = mask.numpy() * 255
                        mask = resize_mask_to_original(mask.astype(np.uint8), padding_info)
                        output_path = os.path.join(results_path_yolov8, f"{os.path.splitext(image_name)[0]}_output.png")
                        cv2.imwrite(output_path, mask)

def getResultsU2net(path):
    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    model = U2NET(3, 1)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    test_images = sorted(os.listdir(test_images_path))

    for image_name in test_images:
        image_path = os.path.join(test_images_path, image_name)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image, padding_info = resize_and_pad(image)
        image_tensor = image_transform(Image.fromarray(image)).unsqueeze(0).to(device)
        
        with torch.no_grad():
            d0, _, _, _, _, _, _ = model(image_tensor)
            predicted_mask = d0.squeeze(0).squeeze(0).cpu().numpy()

        threshold = 0.5
        predicted_mask = (predicted_mask > threshold).astype(np.uint8)
        predicted_mask = resize_mask_to_original(predicted_mask, padding_info)

        output_path = os.path.join(results_path_u2net, f"{os.path.splitext(image_name)[0]}_output.png")
        plt.imsave(output_path, predicted_mask, cmap="gray")

def getResultsDeeplabv3(path):
    image_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    num_classes = 2
    model = models.deeplabv3_resnet50(weights=None)
    model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=(1, 1))

    state_dict = torch.load(path, map_location=device, weights_only=True)
    filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith("aux_classifier")}
    model.load_state_dict(filtered_state_dict, strict=False)

    model.to(device)
    model.eval()

    test_images = sorted(os.listdir(test_images_path))


    for image_name in test_images:
        image_path = os.path.join(test_images_path, image_name)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image, padding_info = resize_and_pad(image)
        image_tensor = image_transform(Image.fromarray(image)).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(image_tensor)["out"]
            predicted_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

        predicted_mask = resize_mask_to_original(predicted_mask, padding_info)

        output_path = os.path.join(results_path_deeplabv3, f"{os.path.splitext(image_name)[0]}_output.png")
        plt.imsave(output_path, predicted_mask, cmap="gray")

if __name__ == "__main__":
    print(f"Using device: {device}")

    for name, path in model_paths.items():
        print(f"Starting Testing for {name}".ljust(50), end="")
        print('')

        if name == "YOLOv8":
            getResultsYolov8(path)
        if name == "U2Net":
            getResultsU2net(path)
        if name == "DeepLabv3":
            getResultsDeeplabv3(path)

        print(f"\rCompleted Testing for {name}".ljust(50), end="")
        print('')

    print("Testing complete.")
