from info import *
import os
from subprocess import call
import argparse


parser = argparse.ArgumentParser('Detetion, tracking, and submission creation!')
parser.add_argument('--experiment_name', type=str, default='yolox_l', help='Optional name of experiment')
parser.add_argument('--video_id', type=str, default=None, required=True, help='Absolute path file with video id and paths')
parser.add_argument('--cuda_path', default='/usr/local/cuda-11.3/lib64', type=str, help='CUDA library path')
parser.add_argument('--submission_name', default='submission.txt', type=str, help='Output submission file')
# parser.add_argument('--tracker', default='SORT', type=str, choices=['BYTE', 'SORT','DEEPSORT'], help='Tracker used')
parser.add_argument('--tracker', default='DEEPSORT', choices=('SORT','DEEPSORT'), help='tracker type (default: DEEPSORT)')
parser.add_argument('--img_size', default=640, type=int, help='Detector input image size')
args = parser.parse_args()

vids = load_ids_and_paths(args.video_id)

# # # ################################################################################
os.chdir('detector_and_tracker')
os.environ['LD_LIBRARY_PATH'] = args.cuda_path
os.environ['PYTHONPATH'] = os.getcwd()
for vid in vids:
    call(['python', 'tools/detector_with_tracker.py',
           '-expn', args.experiment_name,
           '--path', os.path.join(inpainting_path, vid['name'], vid['name']+'.mp4'),
           '--roi_path', os.path.join(rois_path, vid['name']),
           '--tracker', args.tracker,
           '--tsize', str(args.img_size),
           '-f', 'exps/aic_yolox_l.py',
           '-c', '/data/aicity/aic23/track4/PersonGONE/detector_and_tracker/checkpoints_1/yolo8/best_ckpt.pth'

           ])
################################################################################

################################################################################
os.chdir('..')

call(['python', '/data/aicity/aic23/track4/PersonGONE/utils/tracks_roi_frame.py',
      os.path.abspath(os.path.join('detector_and_tracker', 'YOLOX_outputs_1', args.experiment_name)),
      args.submission_name,
      args.video_id
      ])
# # # ################################################################################
