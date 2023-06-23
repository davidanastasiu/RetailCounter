import os
import cv2
import numpy as np
import json
import argparse


# Get a list of all image file names in the mean scene folder
parser = argparse.ArgumentParser('ROI Median Area')
parser.add_argument('--extracted_image', type=str, required=True, help='path of extarcted background images')
parser.add_argument('--store_path', type=str, required=True, help='save ROI')
parser.add_argument('--final_store_path', type=str, required=True, help='save Median ROI')

args = parser.parse_args()
extracted_path = args.extracted_image
store_path = args.store_path
image_files = sorted([file_name for file_name in os.listdir(extracted_path) if file_name.endswith('.png')], key=lambda x: int(x.split('.')[0]))
# Get a list of the ROI areas for the first 10 and last 10 frames
roi_areas_first_10 = []
roi_areas_last_10 = []
for i, file_name in enumerate(image_files):
    if i < 10:
        json_path = os.path.join(store_path, os.path.splitext(file_name)[0] + ".json")
        with open(json_path, 'r') as f:
            roi_rect = json.load(f)
        roi_area = (roi_rect[2] - roi_rect[0]) * (roi_rect[3] - roi_rect[1])
        roi_areas_first_10.append((roi_area, roi_rect))
    elif i >= len(image_files) - 10:
        json_path = os.path.join(store_path, os.path.splitext(file_name)[0] + ".json")
        with open(json_path, 'r') as f:
            roi_rect = json.load(f)
        roi_area = (roi_rect[2] - roi_rect[0]) * (roi_rect[3] - roi_rect[1])
        roi_areas_last_10.append((roi_area, roi_rect))

# Calculate the median of the ROI areas for the first 10 and last 10 frames
median_roi_area_first_10 = np.median([x[0] for x in roi_areas_first_10])
median_roi_area_last_10 = np.median([x[0] for x in roi_areas_last_10])

# Find the ROI rectangle that has the closest area to the median value
closest_roi_rect = None
closest_roi_area_diff = float('inf')
for roi_area, roi_rect in roi_areas_first_10 + roi_areas_last_10:
    area_diff = abs(roi_area - median_roi_area_first_10)
    if area_diff < closest_roi_area_diff:
        closest_roi_rect = roi_rect
        closest_roi_area_diff = area_diff

# Extract x1, y1, x2, y2 coordinates from the closest ROI rectangle
x1, y1, x2, y2 = closest_roi_rect
roi_dict = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}

# Save dictionary as JSON file
with open(args.final_store_path, 'w') as f:
    json.dump(roi_dict, f)

