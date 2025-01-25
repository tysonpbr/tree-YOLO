from ultralytics import YOLO

import cv2
import os
import numpy as np


model_path = "C:/Users/tyson/Desktop/tree-segmentation/runs/segment/train/weights/last.pt"
images_dir = "C:/Users/tyson/Desktop/tree-segmentation/test/images"
masks_output_dir = "./test/output/masks/"
combined_output_dir = "./test/output/combined/"
cropped_output_dir = "./test/output/cropped/"

model = YOLO(model_path)

def find_largest_white_rectangle(mask):
    rows, cols = mask.shape

    height = np.zeros((rows, cols), dtype=int)

    for r in range(rows):
        for c in range(cols):
            if mask[r, c] != 0:
                height[r, c] = height[r - 1, c] + 1 if r > 0 else 1

    max_area = 0
    best_coords = (0, 0, 0, 0)

    for r in range(rows):
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
                    
            stack.append(c)

    return best_coords

for filename in os.listdir(images_dir):    
  if filename.endswith((".jpg", ".jpeg")):
    
    file_path = os.path.join(images_dir, filename)
    img = cv2.imread(file_path)
    
    if img is None:
      print(f"Failed to load {filename}")
      continue

    H, W, _ = img.shape

    results = model(img)

    for result in results:
      if result.masks:
        for j, mask in enumerate(result.masks.data):

          mask = mask.numpy() * 255
          mask = cv2.resize(mask, (W, H))
          cv2.imwrite(masks_output_dir + os.path.splitext(filename)[0] + '.png', mask)

          binary_mask = (mask > 128).astype(np.uint8)
          masked_image = cv2.bitwise_and(img, img, mask=binary_mask)
          cv2.imwrite(combined_output_dir + filename, masked_image)

          largest_rect = find_largest_white_rectangle(binary_mask)
          x, y, w, h = largest_rect
          cropped_image = img[y:y+h, x:x+w]
          cv2.imwrite(cropped_output_dir + filename, cropped_image)




