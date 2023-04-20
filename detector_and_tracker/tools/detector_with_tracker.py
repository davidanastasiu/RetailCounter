#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
import time
from loguru import logger

import cv2
import numpy as np
import pickle
import json
import matplotlib

import torch
from sort.sort import Sort
from yolox.tracker.byte_tracker import BYTETracker
from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis
from deep_sort.deepsort import DeepSort
from PIL import Image
from deep_sort import feature_extractor
from deep_sort.utils import vis_track



def make_parser():
    parser = argparse.ArgumentParser("YOLOV8 detection + tracker!")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None, required=True)
    parser.add_argument("--path", default=None, help="path to video", required=True)
    parser.add_argument("--roi_path", default=None, help="path to file with ROI definition (JSON)")
    parser.add_argument("--roi_expand", default=0.1, type=float, help="ROI expansion in percents of are (0-1)")
    parser.add_argument("--save_video", action="store_false", help="whether to save the inference result of video")
    #################################################
    parser.add_argument("--use_crop", action="store_true", help="whether to save the inference result of video")
    #################################################
    parser.add_argument("--tracker", type=str, default='DEEPSORT', help="which tracker to use (SORT/DEEPSORT)")
    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="experiment description file",
        required=True
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt path", required=True)
    parser.add_argument(
        "--device",
        default="cuda:1",
        type=str,
        help="device to run (cpu/gpu)",
    )
    parser.add_argument("--conf", default=0.50, type=float, help="test conf")
    parser.add_argument("--nms", default=0.60, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=640, type=int, help="test img size")
    return parser


class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        cls_names=COCO_CLASSES,
        device=torch.device("cuda:1")
        # device="GPU"
    ):
        self.model = model
        self.cls_names = cls_names
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.use_crop = exp.use_crop
        self.device = device
        self.preproc = ValTransform(legacy=False)


        if args.tracker == 'SORT':
            self.tracker = Sort(max_age=args.track_buffer, min_hits=3, iou_threshold=0.15)
        elif args.tracker == 'DEEPSORT':
           self.tracker = DeepSort(self.model, max_age=1, n_init=3,max_iou_distance= 0.30, nn_budget=None)
        else:
            raise Exception('Unknown tracker')
        logger.info("Going to use {:s} tracker".format(args.tracker))
        self.out_data = {}


    def inference(self, img):

        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == 'cuda:1':
            img = img.cuda()

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred) - yolox/utils/boxes.py
            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )

        return outputs, img_info

    def visual(self, output, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img
        output = output.cpu()

        bboxes = output[:, 0:4] # from 0 element slice from 0 -4

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]
        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
        timestamp = int(time.time())
        image_name = f"output/output_{timestamp}.png"
        matplotlib.image.imsave(image_name, vis_res)


        return vis_res

    def proceede_dets_to_tracker(self, output, img_info, frame_pos):
        if output is None:
            return

        ratio = img_info["ratio"]
        if args.tracker == "DEEPSORT":
            output = output.cpu()
            bboxes = output[:, 0:4]

            bboxes /= ratio
            cls = output[:, 6]
            scores = output[:, 4] * output[:, 5]
            class_confs = output[:, 5]
            det_confs = output[:, 4]
            # detections = self.tracker.update(bboxes,scores.tolist(),img_info['raw_img'])
            # image = vis_track(img_info['raw_img'], detections)
            # matplotlib.image.imsave('track.png', image)
            dets = [{'det': d[0:4].tolist(),
                         'class' : cls.tolist()[-1],
                          'class_conf' : class_confs.tolist()[-1],
                         'det_conf' :det_confs.tolist()[-1] }
                          for pos, d in enumerate(output)]

            self.out_data[frame_pos] = dets


################################################
##########################################################


    def apply_roi(self, img,frame_pos):
        if args.roi_path is not None:
            roi_filename = args.roi_path + '/' + '{:04d}'.format(frame_pos) + '.json'
            with open(roi_filename,'r') as f:
                self.roi = json.load(f)


            if args.roi_expand is not None:
                roi_w, roi_h = self.roi[2] - self.roi[0], self.roi[3] - self.roi[1]
                self.roi = [int(self.roi[0]-roi_w*(args.roi_expand/2)),
                                int(self.roi[1]-roi_h*(args.roi_expand/2)),
                                int(self.roi[2]+roi_w*(args.roi_expand/2)),
                                int(self.roi[3]+roi_h*(args.roi_expand/2))]
            else:
                self.roi = None
        if self.roi is not None:
            if self.use_crop:
                cropped_img = img[self.roi[1]:self.roi[3], self.roi[0]:self.roi[2], :]
                return cropped_img

            blur_bckg = np.full(img.shape, (213,215,221), dtype=np.uint8)
            blur_bckg[self.roi[1]:self.roi[3], self.roi[0]:self.roi[2], :] = img[self.roi[1]:self.roi[3], self.roi[0]:self.roi[2], :]
            return blur_bckg
        else:
            return img


def proceede_video(predictor, vis_folder, current_time, args):
    cap = cv2.VideoCapture(args.path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float

    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float

    fps = cap.get(cv2.CAP_PROP_FPS)
    save_name = '{:s}'.format(os.path.basename(args.path).split('.')[0])
    if args.save_video:

        save_path = os.path.join(vis_folder, save_name+'.mp4')

        logger.info(f"video save_path is {save_path}")
        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (432, 240))
    frame_pos = 0
    while True:
        ret_val, frame = cap.read()

        if ret_val:
            original_frame = frame.copy()
            frame = predictor.apply_roi(frame, frame_pos)
            predictor.out_data[frame_pos] = []
            outputs, img_info = predictor.inference(frame)
            img_info["raw_img"] = original_frame
            result_frame = predictor.visual(outputs[0], img_info, predictor.confthre)
            predictor.proceede_dets_to_tracker(outputs[0], img_info, frame_pos)



        else:
            break
    with open(os.path.join(vis_folder, save_name+'.pkl'), 'wb') as f:
        pickle.dump(predictor.out_data, f)
    if args.save_video:
        vid_writer.release()


def main(exp, args):
    assert args.tracker in ('SORT','DEEPSORT')

    if not args.experiment_name:
        args.experiment_name = exp.exp_name
    output_dir = 'YOLOX_outputs_1'
    file_name = os.path.join(output_dir, args.experiment_name)
    os.makedirs(file_name, exist_ok=True)

    vis_folder = None

    vis_folder = file_name
    os.makedirs(vis_folder, exist_ok=True)

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    exp.use_crop = True if args.use_crop else False

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

    if args.device == "gpu":
        device = torch.device("cuda:1")
        model.to(device)
    model.eval()

    logger.info("loading checkpoint")
    ckpt = torch.load(args.ckpt, map_location="cpu")
    # load the model state dict
    model.load_state_dict(ckpt["model"])
    logger.info("loaded checkpoint done.")
    predictor = Predictor(model, exp, COCO_CLASSES, args.device)
    current_time = time.localtime()
    proceede_video(predictor, vis_folder, current_time, args)


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, None)
    main(exp, args)


