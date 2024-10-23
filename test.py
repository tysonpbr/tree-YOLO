from ultralytics import YOLO

import cv2


model_path = "C:/Users/tyson/Desktop/tree-segmentation/runs/segment/train/weights/last.pt"

# image_path = "C:/Users/tyson/Desktop/tree-segmentation/test/red_maple3.jpg"
image_path = "C:/Users/tyson/Desktop/tree-segmentation/data/images/val/0.jpg"

img = cv2.imread(image_path)
H, W, _ = img.shape

model = YOLO(model_path)

results = model(img)

for result in results:
  if result.masks:
    for j, mask in enumerate(result.masks.data):

      mask = mask.numpy() * 255

      mask = cv2.resize(mask, (W, H))

      cv2.imwrite('./test/output.png', mask)
  else:
    print("No data")

