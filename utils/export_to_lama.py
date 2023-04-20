import cv2
import os
from tqdm import tqdm
import sys

frames_path = sys.argv[1]
masks_path = sys.argv[2]
output_path = sys.argv[3]
scene = sys.argv[4]

erosion_size = 9
element = cv2.getStructuringElement(cv2.MORPH_CROSS, (erosion_size, erosion_size))

print('Going to proceede scene {:s}'.format(scene))
os.makedirs(os.path.join(output_path, scene), exist_ok=True)
all_imgs = sorted(os.listdir(os.path.join(frames_path, scene)))
for img in tqdm(all_imgs):
    i = cv2.imread(os.path.join(frames_path, scene, img))
    base_name = img.split('.')[0]
    m = cv2.imread(os.path.join(masks_path, scene, base_name+'.png'))

    m = cv2.dilate(m, element, iterations=3)

    cv2.imwrite(os.path.join(output_path, scene, img), i)
    cv2.imwrite(os.path.join(output_path, scene, base_name+'_mask001.png'), m)
