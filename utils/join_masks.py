import os
import cv2
import numpy as np
from tqdm import tqdm
import sys

variants = ['YOLACT']
out_dir = 'all'

scenes = os.listdir(os.path.join(sys.argv[1], variants[0]))

for scene in scenes:
    act_path = os.path.join(sys.argv[1], out_dir, scene)
    os.makedirs(act_path, exist_ok=True)
    imgs = os.listdir(os.path.join(sys.argv[1], variants[0], scene))

    for i in tqdm(imgs):
        masks = []
        for v in variants:
            img = cv2.imread(os.path.join(sys.argv[1], v, scene, i))
            masks.append(img)

        out_mask = np.zeros(masks[0].shape, dtype=np.uint8)
        for m in masks:
            out_mask = np.where(m.astype(np.bool_), (255, 255, 255), out_mask)
        cv2.imwrite(os.path.join(act_path, i), out_mask)
