import argparse
import os
import time
from loguru import logger
from ultralytics import YOLO
import cv2
import numpy as np
import pickle
import json
from yolox.data.data_augment import ValTransform
import matplotlib
import torch
from yolox.data.datasets import COCO_CLASSES
from yolox.utils import fuse_model, get_model_info, postprocess, vis
from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.utils import DEFAULT_CFG, ROOT, ops
from yolox.utils import fuse_model, get_model_info, postprocess, vis
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box
# from detector_and_tracker.detection.predict import DetectionPredictor


def make_parser():
    # parser = argparse.ArgumentParser("YOLOX detection + tracker!")
    parser = argparse.ArgumentParser("YOLOV8 detection + tracker!")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None, required=True)
    parser.add_argument("--path", default=None, help="path to video", required=True)
    parser.add_argument("--roi_path", default=None, help="path to file with ROI definition (JSON)")
    parser.add_argument("--roi_expand", default=0.1, type=float, help="ROI expansion in percents of are (0-1)")
    parser.add_argument("--save_video", action="store_false", help="whether to save the inference result of video")
    #################################################
    # parser.add_argument("--use_crop", action="store_false", help="whether to save the inference result of video")
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
        default="GPU",
        type=str,
        help="device to run (cpu/gpu)",
    )
    parser.add_argument("--conf", default=0.65, type=float, help="test conf")
    parser.add_argument("--nms", default=0.60, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=640, type=int, help="test img size")
    return parser


class Predictor(object):
    def __init__(
        self,
        model,
        cls_names=COCO_CLASSES,

        device="GPU"
    ):
        self.model = model
        # self.cls_names = cls_names
        self.num_classes = 117
        self.confthre = 0.60
        self.nmsthre = 0.60
        self.test_size = (640, 640)
        self.use_crop = args.use_crop
        self.device = device
        self.preproc = ValTransform(legacy=False)
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
        if self.device == "gpu":

            print('device',self.device )
            img = img.cuda(self.device)

        with torch.no_grad():
            t0 = time.time()
            result = self.model.predict(img, save=True)
            outputs = []
            for i in result:
                bboxes = i.boxes
                for box in bboxes:
                    b = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format
                    c = box.cls
                    d =box.conf
                    outputs.append(b)
                    outputs.append(c)
                    outputs.append(d)



        return outputs, img_info


    def visual(self, output, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if not output:
            return img

        bboxes = [output[0].cpu()] # from 0 element slice from 0 -4

        # preprocessing: resize
        bboxes[0] /= ratio

        cls = output[1].cpu()
        scores = output[2].cpu()
        vis_res = vis(img, bboxes, scores, cls, cls_conf)

        return vis_res

    def proceede_dets_to_tracker(self, output, img_info, frame_pos):
        if output is None:
            return

        ratio = img_info["ratio"]
        if args.tracker == "DEEPSORT":
            output = output.cpu()
            bboxes = output[0:4]
            # preprocessing: resize
            bboxes /= ratio
            cls = output[5]
            scores = output[4] * output[5]
            class_confs = output[4]
            det_confs = output[:, 4]
            bboxes = torch.Tensor(bboxes)
            detections = self.tracker.update(bboxes,scores.tolist(),img_info['raw_img'])
            print(detections)
            dets = [{'det': d[0:4].tolist(),
                     'class': cls.tolist()[-1],
                     'class_conf': class_confs.tolist()[-1]}
                    for pos, d in enumerate(output)]
            print(dets)
            self.out_data[frame_pos] = dets
            print(args.tracker, dets, frame_pos)

        ################################################
        ##########################################################

    def apply_roi(self, img, frame_pos):
        if args.roi_path is not None:
            roi_filename = args.roi_path + '/' + '{:04d}'.format(frame_pos) + '.json'
            with open(roi_filename, 'r') as f:
                self.roi = json.load(f)

            if args.roi_expand is not None:
                roi_w, roi_h = self.roi[2] - self.roi[0], self.roi[3] - self.roi[1]
                self.roi = [int(self.roi[0] - roi_w * (args.roi_expand / 2)),
                                int(self.roi[1] - roi_h * (args.roi_expand / 2)),
                                int(self.roi[2] + roi_w * (args.roi_expand / 2)),
                                int(self.roi[3] + roi_h * (args.roi_expand / 2))]
            else:
                self.roi = None
        if self.roi is not None:
            if self.use_crop:
                img = img[self.roi[1]:self.roi[3], self.roi[0]:self.roi[2], :]
                image_1 = cv2.resize(img, (640, 640))
                return image_1
            blur_bckg = np.full(img.shape, (213, 215, 221), dtype=np.uint8)
            blur_bckg[self.roi[1]:self.roi[3], self.roi[0]:self.roi[2], :] = img[self.roi[1]:self.roi[3],self.roi[0]:self.roi[2], :]
            return blur_bckg
        else:
            return img


def proceede_video(predictor, vis_folder, args):

    cap = cv2.VideoCapture(args.path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    print('width', width)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    print('height', height)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print('CAP_PROP_FPS',fps)
    save_name = '{:s}'.format(os.path.basename(args.path).split('.')[0])
    if args.save_video:
        save_path = os.path.join(vis_folder, save_name+'.mp4')
        logger.info(f"video save_path is {save_path}")
        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (432, 240))
    frame_pos = 0
    while True:
        ret_val, frame = cap.read()

        if ret_val:
            frame = cv2.resize(frame, (432, 240))
            original_frame = frame.copy()
            frame = predictor.apply_roi(frame, frame_pos)
            predictor.out_data[frame_pos] = []
            outputs, img_info = predictor.inference(frame)
            img_info["raw_img"] = original_frame
            result_frame = predictor.visual(outputs, img_info, predictor.confthre)
            if args.save_video:
                vid_writer.write(result_frame)

        else:
            break
    with open(os.path.join(vis_folder, save_name+'.pkl'), 'wb') as f:
        pickle.dump(predictor.out_data, f)
    if args.save_video:
        vid_writer.release()



def main(args):
    logger.info("Args: {}".format(args))
    if args.device == "GPU":
        model = YOLO("/home/arpita/runs/detect/yolov8m_custom2/weights/best.pt")
        file_name = os.path.join(args.experiment_name)
        os.makedirs(file_name, exist_ok=True)
        vis_folder = None
        # vis_folder = os.path.join(file_name, os.path.basename(args.path).split('.')[0])
        vis_folder = file_name
        os.makedirs(vis_folder, exist_ok=True)
        logger.info("loading checkpoint")
        ckpt = torch.load(args.ckpt, map_location="cpu")
        # load the model state dict
        # model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")
        predictor = Predictor(model,device="GPU")
        current_time = time.localtime()
        print(args)

        proceede_video(predictor, vis_folder, current_time, args)


if __name__ == "__main__":
    args = make_parser().parse_args()
    # exp = get_exp(args.exp_file, None)
    main(args)