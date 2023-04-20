import os
import cv2
import numpy as np
import json
import argparse


parser = argparse.ArgumentParser('ROI detection process!')
parser.add_argument('--store_path', type=str, required=True, help='save ROI')
parser.add_argument('--extracted_image', type=str, required=True, help='path of extarcted background images')

args = parser.parse_args()
extracted_path = args.extracted_image
store_path = args.store_path


# Get a list of all image file names in the mean scene folder
image_files = [file_name for file_name in os.listdir(extracted_path) if file_name.endswith('.png')]

# Iterate over the image files and read each one using cv2.imread()
for file_name in image_files:
    image_path = os.path.join(extracted_path, file_name)
    img = cv2.imread(image_path)
    img = cv2.resize(img, (432, 240))
    h, w, _ = img.shape
    scale = 1
    delta = 0
    grayscale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ddepth = cv2.CV_16S
    grad_x = cv2.Scharr(grayscale, ddepth, 1, 0, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Scharr(grayscale, ddepth, 0, 1, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    lg_x = 122
    lg_y = 70
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    seed = (int(w/2), int(h/2)+20)
    ret, out_img, mask, rect = cv2.floodFill(grad, np.zeros((h+2,w+2), dtype=np.uint8), seed, 255, 7, 7)
    rect = list(rect)
    if rect[2] < 100 or rect[3] < 80:
        rect[0] = lg_x
        rect[1] = lg_y
    last_good_x = rect[0]
    last_good_y = rect[1]

    # print('ROI',rect[0], rect[1], rect[0]+155, rect[1]+120, file_name)
    mean_img = np.median(np.array(img), axis=0).astype(np.uint8)
    json_path = os.path.join(store_path, os.path.splitext(file_name)[0] + ".json")
    with open(json_path, 'w') as f:
        json.dump([rect[0], rect[1], rect[0] + 155, rect[1] + 120], f)
