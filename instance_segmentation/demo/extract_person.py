# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import cv2
import mmcv
import torch
import os
import numpy as np

from mmdet.apis import inference_detector, init_detector

variants = {'DetectoRS' : {'config' : 'configs/detectors/detectors_htc_r101_20e_coco.py',
                          'ckpt' : 'checkpoints/detectors_htc_r101_20e_coco_20210419_203638-348d533b.pth'},
            'HTC' : {'config' : 'configs/htc/htc_x101_64x4d_fpn_dconv_c3-c5_mstrain_400_1400_16x1_20e_coco.py',
                     'ckpt' : 'checkpoints/htc_x101_64x4d_fpn_dconv_c3-c5_mstrain_400_1400_16x1_20e_coco_20200312-946fd751.pth'},
            'PointRend' : {'config' : 'configs/point_rend/point_rend_r50_caffe_fpn_mstrain_3x_coco.py',
                           'ckpt' : 'checkpoints/point_rend_r50_caffe_fpn_mstrain_3x_coco-e0ebb6b7.pth'},
            'YOLACT' : {'config' : 'configs/yolact/yolact_r101_1x8_coco.py',
                        'ckpt' : 'checkpoints/yolact_r101_1x8_coco_20200908-4cbe9101.pth'}
            }

def process_data(result, img_path, score_thr=0.3):
    if isinstance(result, tuple):
        bbox_result, segm_result = result
        if isinstance(segm_result, tuple):
            segm_result = segm_result[0]  # ms rcnn
    else:
        bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    # draw segmentation masks
    segms = None
    if segm_result is not None and len(labels) > 0:  # non empty
        segms = mmcv.concat_list(segm_result)
        if isinstance(segms[0], torch.Tensor):
            segms = torch.stack(segms, dim=0).detach().cpu().numpy()
        else:
            segms = np.stack(segms, axis=0)

    if score_thr > 0:
        assert bboxes is not None and bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]
        if segms is not None:
            segms = segms[inds, ...]

    seg_img = np.zeros(segms.shape[1:], dtype=np.uint8)

    if segms is not None:
        for label in labels:
            if label == 0:
                seg_img = np.where(segms[label], 255, seg_img)
        cv2.imwrite(img_path, seg_img)


def parse_args():
    parser = argparse.ArgumentParser(description='MMDetection video demo')
    parser.add_argument('video', help='Video file')
    parser.add_argument('variant', help='Detector variant')
    parser.add_argument('base_path', default=None, type=str, help='Path where to store extraced masks')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='Bbox score threshold')
    parser.add_argument(
        '--wait-time',
        type=float,
        default=1,
        help='The interval of show (s), 0 is block')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    assert args.variant in variants

    var = variants[args.variant]

    model = init_detector(var['config'], var['ckpt'], device=args.device)
    video_reader = mmcv.VideoReader(args.video)
    act_path = os.path.join(args.base_path, args.variant, os.path.basename(args.video).split('.')[0])
    os.makedirs(act_path, exist_ok=True)

    cnt = 0

    for frame in mmcv.track_iter_progress(video_reader):
        result = inference_detector(model, frame)
        process_data(result, os.path.join(act_path, '{:06d}.png'.format(cnt)))
        cnt += 1



if __name__ == '__main__':
    main()
