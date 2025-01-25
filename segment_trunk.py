import os
import torch
from ultralytics import YOLO
import numpy as np
import cv2

current_model_path = 'trained_models/model_bs32_lr0.01/weights/best.pt'

test_images_path = "data/images/real/"

results_path_masks = "results/real/masks/"
results_path_combined = "results/real/combined/"
results_path_cropped = "results/real/cropped/"

os.makedirs(results_path_masks, exist_ok=True)
os.makedirs(results_path_combined, exist_ok=True)
os.makedirs(results_path_cropped, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = YOLO(current_model_path)

def find_largest_white_rectangle(mask):
    rows, cols = mask.shape

    height = np.zeros((rows, cols), dtype=int)
    height[0, :] = mask[0, :]
    for r in range(1, rows):
        height[r, :] = np.where(mask[r, :] != 0, height[r - 1, :] + 1, 0)

    max_area = 0
    best_coords = (0, 0, 0, 0)

    for r in range(rows):
        if np.max(mask[r, :]) == 0:
            continue
        stack = []
        for c in range(cols + 1):
            h = height[r, c] if c < cols else 0 
            while stack and h < height[r, stack[-1]]:
                H = height[r, stack.pop()]
                W = c if not stack else c - stack[-1] - 1
                area = H * W
                if area > max_area:
                    max_area = area
                    x = stack[-1] + 1 if stack else 0
                    best_coords = (x, r - H + 1, W, H)

                    if area >= (rows*cols / 6):
                       return best_coords
                    
            stack.append(c)

    return best_coords

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
    mask = mask[top:512-bottom, left:512-right]
    mask_resized = cv2.resize(mask, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)
    return mask_resized

def getResultsYolov8(image):
    image_resize, padding_info = resize_and_pad(image)

    results = model(image_resize)
    
    for result in results:
        if result.masks:
            for j, mask in enumerate(result.masks.data):
                if j == 0:
                    mask = mask.cpu().numpy() * 255
                    mask = resize_mask_to_original(mask.astype(np.uint8), padding_info)

                    binary_mask = (mask > 128).astype(np.uint8)
                    combined = cv2.bitwise_and(image, image, mask=binary_mask)

                    largest_rect = find_largest_white_rectangle(binary_mask)
                    x, y, w, h = largest_rect

                    cropped = image[y:y+h, x:x+w]

                    return mask, combined, cropped
                
    return None, None, None

if __name__ == "__main__":
    print(f"Using device: {device}")

    test_images = sorted(os.listdir(test_images_path))

    for image_name in test_images:
        image_path = os.path.join(test_images_path, image_name)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Failed to read image: {image_name}. Skipping...")
            continue

        mask, combined, cropped = getResultsYolov8(image)

        if mask is not None:
            output_path = os.path.join(results_path_masks, f"{os.path.splitext(image_name)[0]}.png")
            cv2.imwrite(output_path, mask)
            
            output_path = os.path.join(results_path_combined, f"{os.path.splitext(image_name)[0]}.png")
            cv2.imwrite(output_path, combined)
            
            output_path = os.path.join(results_path_cropped, f"{os.path.splitext(image_name)[0]}.png")
            cv2.imwrite(output_path, cropped)

        else:
            print(f"No mask detected for {image_name}.")

    print("Testing complete.")
