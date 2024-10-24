from ultralytics import YOLO

import cv2
import os
import numpy as np


model_path = "C:/Users/tyson/Desktop/tree-segmentation/runs/segment/train/weights/last.pt"
images_dir = "C:/Users/tyson/Desktop/tree-segmentation/test/images"
masks_output_dir = "./test/output/masks/"
combined_output_dir = "./test/output/combined/"

model = YOLO(model_path)

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




