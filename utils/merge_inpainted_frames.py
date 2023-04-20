import os
import cv2
from tqdm import tqdm
import sys

base_path = sys.argv[1]

scenes = os.listdir(base_path)

for s in scenes:
    print('Scene: {:s}'.format(s))
    act_path = os.path.join(base_path, s)
    imgs = sorted(os.listdir(act_path))

    h, w, _ = cv2.imread(os.path.join(act_path, imgs[0])).shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vw = cv2.VideoWriter(
        os.path.join(act_path, s+'.mp4'), fourcc, 60,
        (w, h))

    for i in tqdm(imgs):
        img = cv2.imread(os.path.join(act_path, i))
        vw.write(img)

    vw.release()
