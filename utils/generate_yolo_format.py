import json
import os


def coco_to_yolo(box, img_width, img_height):
    x, y, w, h = box
    x_center = x + w / 2
    y_center = y + h / 2
    x_center /= img_width
    y_center /= img_height
    w /= img_width
    h /= img_height
    return x_center, y_center, w, h

# Load the COCO annotation file

with open('/data/aicity/aic23/track4/PersonGONE/data/dataset/annotations/validation.json', 'r') as f:
    annotations = json.load(f)

# Create a dictionary to hold the annotations for each image
image_annotations = {}
for annotation in annotations['annotations']:
    image_id = annotation['image_id']
    if image_id not in image_annotations:
        image_annotations[image_id] = []
    image_annotations[image_id].append(annotation)

# Convert COCO annotations to YOLO format and save them in separate text files
output_folder = '/data/aicity/aic23/track4/PersonGONE/data/dataset_yolov8/valid/labels/'
for image in annotations['images']:
    image_id = image['id']
    filename = image['file_name']
    img_width = image['width']
    img_height = image['height']
    annotations_for_image = image_annotations.get(image_id, [])
    with open(os.path.join(output_folder, os.path.splitext(filename)[0] + '.txt'), 'w') as f:
        for annotation in annotations_for_image:
            category_id = annotation['category_id']
            box = annotation['bbox']
            x_center, y_center, w, h = coco_to_yolo(box, img_width, img_height)
            f.write('{} {} {} {} {}\n'.format(category_id, x_center, y_center, w, h))
