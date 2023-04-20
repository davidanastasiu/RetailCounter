import cv2
import numpy as np
from tqdm import tqdm
import sys
import os

path = sys.argv[1]
store_path = sys.argv[2]
cap = cv2.VideoCapture(path)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frames_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
bckg_sub = cv2.createBackgroundSubtractorMOG2()
os.makedirs(store_path, exist_ok=True) # create directory to store background images
for i in tqdm(range(frames_cnt)):
    ret, frame = cap.read()
    if not ret:
        break

    fg_mask = bckg_sub.apply(frame)
    bg_image = bckg_sub.getBackgroundImage()
    cv2.imwrite(os.path.join(store_path, f"{i:04d}.png"), bg_image)

print('Images saved at:', store_path)
cap.release()