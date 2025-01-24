import os
import torch
from ultralytics import YOLO
import numpy as np
import cv2

models_folder_path = 'trained_models/'
model_file_path = 'weights/best.pt'

test_images_path = "data/images/test/"
test_masks_path = "data/masks/test/"

results_path = "results/test/"
results_path_file= "results/test/results.txt"
iou_results_path_file= "results/test/iou_results.txt"

if not os.path.exists(results_path):
    os.makedirs(results_path)

with open(results_path_file, "w") as file:
    pass

with open(iou_results_path_file, "w") as file:
    pass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calculate_iou(output, mask):
    output_binary = (output > 127).astype(np.uint8)
    mask_binary = (mask > 127).astype(np.uint8)
    
    intersection = np.logical_and(output_binary, mask_binary).sum()
    union = np.logical_or(output_binary, mask_binary).sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    iou = intersection / union
    return iou

def per_pixel_accuracy(output, mask):
    output_binary = (output > 127).astype(int)
    mask_binary = (mask > 127).astype(int)

    correct_pixels = np.sum(output_binary == mask_binary)
    total_pixels = output.size

    accuracy = correct_pixels / total_pixels
    return accuracy

def compare_with_ground_truth(generated_mask_path, ground_truth_mask_path):
    generated_mask = cv2.imread(generated_mask_path, cv2.IMREAD_GRAYSCALE)
    ground_truth_mask = cv2.imread(ground_truth_mask_path, cv2.IMREAD_GRAYSCALE)
    
    if generated_mask is None or ground_truth_mask is None:
        return 0

    accuracy = per_pixel_accuracy(generated_mask, ground_truth_mask)
    iou = calculate_iou(generated_mask, ground_truth_mask)
    return accuracy, iou

def getResultsYolov8(name):
    path = os.path.join(models_folder_path, name, model_file_path)
    model_results_path = os.path.join(results_path, name)
    model_results_path_file = os.path.join(model_results_path, 'results.txt')
    model_iou_results_path_file = os.path.join(model_results_path, 'iou_results.txt')

    model = YOLO(path)
    test_images = sorted(os.listdir(test_images_path))

    if not test_images:
        print(f"No test images found in {test_images_path} for {name}")
        return

    if not os.path.exists(model_results_path):
        os.makedirs(model_results_path)

    with open(model_results_path_file, "w") as file:
        pass

    with open(model_iou_results_path_file, "w") as file:
        pass

    model_accuracy = 0
    model_iou = 0

    for image_name in test_images:
        ground_truth_mask_path = os.path.join(test_masks_path, f"{os.path.splitext(image_name)[0]}.png")
        image_path = os.path.join(test_images_path, image_name)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Error reading image: {image_path}")
            continue
        
        results = model(image)

        accuracy = 0
        iou = 0
        
        for result in results:
            if not result.masks:
                print(f"No masks generated for {image_name}")
                continue
            
            for j, mask in enumerate(result.masks.data):
                if j == 0:
                    mask = mask.cpu().numpy() * 255
                    
                    output_path = os.path.join(model_results_path, f"{os.path.splitext(image_name)[0]}_mask_{j}.png")
                    cv2.imwrite(output_path, mask.astype(np.uint8))

                    accuracy, iou = compare_with_ground_truth(output_path, ground_truth_mask_path)
                    model_accuracy += accuracy
                    model_iou += iou

        with open(model_results_path_file, "a") as file:
            file.write(f"File: {image_name}, Accuracy: {accuracy}\n")

        with open(model_iou_results_path_file, "a") as file:
            file.write(f"File: {image_name}, IOU: {iou}\n")
          
        print(f"\rProcessed and saved output for {image_name}".ljust(50), end="")
    
    model_accuracy = model_accuracy / len(test_images)
    model_iou = model_iou / len(test_images)

    with open(results_path_file, "a") as file:
        file.write(f"{name} Accuracy: {model_accuracy}\n")
        
    with open(iou_results_path_file, "a") as file:
        file.write(f"{name} IOU: {model_iou}\n")

if __name__ == "__main__":
    print(f"Using device: {device}")

    if not os.listdir(models_folder_path):
        print(f"No models found in {models_folder_path}")
        exit(1)

    for folder_name in os.listdir(models_folder_path):
        print(f"Starting Testing for {folder_name}".ljust(50), end="")
        print('')

        getResultsYolov8(folder_name)

        print(f"\rCompleted Testing for {folder_name}".ljust(50), end="")
        print('')

    print("Testing complete.")
