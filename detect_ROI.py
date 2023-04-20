from info import *
import os
from subprocess import call
import argparse


parser = argparse.ArgumentParser('ROI detection process!')
parser.add_argument('--video_id', type=str, default=None, required=True, help='Absolute path file with video id and paths')
parser.add_argument('--roi_seed', type=int, nargs=2, default=None, help='Seed to found ROI')
parser.add_argument('--cuda_path', default='/usr/local/cuda-11.3/lib64', type=str, help='device for training')
parser.add_argument('--extracted_image', type=str, required=True, help='path of extarcted background images')
parser.add_argument('--store_path', type=str, required=True, help='save the ROI')
parser.add_argument('--final_store_path', type=str, required=True, help='save Median ROI')

args = parser.parse_args()

vids = load_ids_and_paths(args.video_id)

# ################################################################################
print('Going to extract mean image for single scenes!')
for vid in vids:
    print('Extracting mean background model for scene:', vid['name'])
    call(['python', 'utils/bckg_subtraction.py',
          os.path.join(inpainting_path, vid['name'], vid['name']+'.mp4'),
          os.path.join(mean_scenes_path, vid['name'])])

################################################################################


################################################################################
print('Going to detect ROI')
for vid in vids:
    print('Detecting ROI for scene:', vid['name'])
    call(['python', 'utils/detect_tray.py'
    ])
################################################################################

################################################################################
'''Optional'''
print('Going to detect ROI Median area')
for vid in vids:
    print('Detecting ROI Median for scene:', vid['name'])
    call(['python', 'utils/detect_ROI_Median.py'
    ])
################################################################################