#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import cv2
import numpy as np

__all__ = ["vis"]


def vis(img, boxes, scores, cls_ids, conf=0.5, class_names=None):

    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        print(cls_id)
        score = scores[i]
        if score < conf:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        # text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
        text = 'ID: {} {:.1f}%'.format(cls_id+1, score * 100)
        # text = 'ID: {} {:.1f}%'.format(cls_id, score * 100)
        txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        # font = cv2.FONT_HERSHEY_COMPLEX_SMALL

        # txt_size = cv2.getTextSize(text, font, 1.2, 1)[0]
        txt_size = cv2.getTextSize(text, font, 0.8, 1)[0]

        cv2.rectangle(img, (x0, y0), (x1, y1), color, 1)
        # print(cls_id)
        txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 + 1),
            # (x0 + txt_size[0] + 10, y0 + int(1.5*txt_size[1])+10),
            (x0 + txt_size[0] + 1, y0 + int(1.5*txt_size[1])),
            txt_bk_color,
            -1
        )
        # cv2.putText(img, text, (x0+5, y0 + txt_size[1]+5), font, 1.2, txt_color, thickness=1)
        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.8, txt_color, thickness=1)

        # cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 1.2, txt_color, thickness=2)

    return img


# _COLORS = np.array(
#     [0.836, 0.91, 0.254, 0.094, 0.883, 0.219, 0.762, 0.855, 0.781, 0.07, 0.148, 0.484, 0.988, 0.938, 0.648, 0.559, 0.484, 0.504, 0.812, 0.531, 0.531, 0.758, 0.508, 0.902, 0.141, 0.398, 0.625, 0.383, 0.43, 0.477, 0.133, 0.441, 0.25, 0.297, 0.168, 0.863, 0.512, 0.625, 0.82, 0.891, 0.652, 0.594, 0.277, 0.418, 0.852, 0.344, 0.594, 0.027, 0.305, 0.812, 0.727, 0.785, 0.07, 0.875, 0.543, 0.188, 0.133, 0.957, 0.711, 0.34, 0.957, 0.863, 0.324, 0.859, 0.344, 0.16, 0.789, 0.191, 0.898, 0.512, 0.301, 0.523, 0.039, 0.258, 0.664, 0.078, 0.246, 0.227, 0.504, 0.902, 0.223, 0.012, 0.348, 0.711, 0.559, 0.277, 0.191, 0.004, 0.613, 0.398, 0.262, 0.121, 0.535, 0.066, 0.031, 0.785, 0.508, 0.215, 0.582, 0.863, 0.887, 0.652, 0.871, 0.391, 0.277, 0.855, 0.645, 0.145, 0.527, 0.055, 0.223, 0.914, 0.109, 0.199, 0.105, 0.461, 0.828, 0.938, 0.934, 0.684, 0.0, 0.492, 0.844, 0.918, 0.48, 0.09, 0.816, 0.445, 0.219, 0.488, 0.027, 0.398, 0.438, 0.957, 0.734, 0.574, 0.836, 0.008, 0.277, 0.211, 0.133, 0.059, 0.652, 0.145, 0.594, 0.957, 0.211, 0.641, 0.23, 0.863, 0.883, 0.055, 0.422, 0.383, 0.961, 0.223, 0.145, 0.949, 0.035, 0.387, 0.062, 0.09, 0.641, 0.57, 0.074, 0.68, 0.836, 0.969, 0.812, 0.598, 0.145, 0.027, 0.473, 0.148, 0.191, 0.051, 0.16, 0.051, 0.828, 0.242, 0.656, 0.824, 0.977, 0.047, 0.316, 0.961, 0.164, 0.492, 0.082, 0.387, 0.316, 0.605, 0.254, 0.258, 0.617, 0.121, 0.52, 0.012, 0.934, 0.762, 0.059, 0.289, 0.32, 0.383, 0.766, 0.098, 0.793, 0.676, 0.328, 0.238, 0.391, 0.863, 0.941, 0.895, 0.195, 0.609, 0.078, 0.352, 0.363, 0.18, 0.605, 0.898, 0.855, 0.293, 0.578, 0.266, 0.441, 0.043, 0.375, 0.078, 0.945, 0.426, 0.414, 0.469, 0.203, 0.91, 0.879, 0.023, 0.473, 0.125, 0.742, 0.379, 0.129, 0.387, 0.156, 0.004, 0.488, 0.496, 0.293, 0.82, 0.797, 0.25, 0.148, 0.168, 0.758, 0.75, 0.965, 0.938, 0.992, 0.77, 0.184, 0.938, 0.332, 0.344, 0.844, 0.359, 0.762, 0.473, 0.348, 0.137, 0.461, 0.332, 0.047, 0.41, 0.902, 0.23, 0.672, 0.672, 0.738, 0.926, 0.035, 0.234, 0.156, 0.898, 0.066, 0.973, 0.445, 0.117, 0.27, 0.344, 0.113, 0.492, 0.055, 0.969, 0.27, 0.641, 0.062, 0.363, 0.148, 0.48, 0.668, 0.262, 0.695, 0.699, 0.109, 0.582, 0.227, 0.887, 0.199, 0.691, 0.996, 0.523, 0.938, 0.801, 0.184, 0.367, 0.605, 0.66, 0.832, 0.934, 0.133, 0.809, 0.797, 0.805, 0.504, 0.789, 0.84, 0.617, 0.762, 0.219, 0.23, 0.41, 0.273, 0.957, 0.297, 0.238, 0.309, 0.863, 0.543, 0.441, 0.621, 0.488, 0.617, 0.883, 0.551, 0.477, 0.445, 0.109]
# ).astype(np.float32).reshape(-1, 3)

# _COLORS = np.random.uniform(0.0, 1.0, size=(116*3)).astype(np.float32).reshape(-1, 3)
# for visualization to the conference paper
_COLORS = np.full((116,3), (0,0,1), dtype=np.float32)