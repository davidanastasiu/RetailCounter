from info import *
import os
import argparse
from subprocess import call

parser = argparse.ArgumentParser('Proceede inpainting process!')
parser.add_argument('--video_id', type=str, default=None, required=True, help='Absolute path file with video id and paths')
parser.add_argument('--cuda_path', default='/usr/local/cuda-11.3/lib64', type=str, help='device for training')
parser.add_argument('--video',type=str, help='video for inpaitning')
parser.add_argument('--mask', type=str, help='mask to be inpainted')
args = parser.parse_args()


vids = load_ids_and_paths(args.video_id)
variants = ['DetectoRS', 'HTC', 'PointRend', 'YOLACT']


################################################################################
# Proceede person instance segmentaion
os.chdir('instance_segmentation')
os.environ['LD_LIBRARY_PATH'] = args.cuda_path
os.environ['PYTHONPATH'] = os.getcwd()
print('Going to generate person masks!')
for v in variants:
    for vid in vids:
        print('Going to generate person masks for video: {:s} and detector variant {:s}'.format(vid['vid_path'], v))
        call(['python', 'demo/extract_person.py',
              vid['vid_path'],
              v,
              masks_path
              ])
################################################################################


################################################################################
# Merge all masks to one for each frame
print('Going to join person masks!')
os.chdir('..')
call(['python', 'utils/join_masks.py',
      masks_path])
################################################################################


################################################################################
print('Going to extract frames!')
for vid in vids:
    call(['python', 'utils/extract_frames.py',
          vid['vid_path'],
          os.path.join(frames_path, vid['name'])
          ])
################################################################################


################################################################################
print('Going to extract masks and frames to LaMa input!')
for vid in vids:
    call(['python', 'utils/export_to_lama.py',
          frames_path,
          os.path.join(masks_path, 'all'),
          lama_path,
          vid['name']
          ])
################################################################################


################################################################################
print('Going to proceede Video inpanting')
os.environ['PYTHONPATH'] = os.getcwd()
call(['python', 'Video_inpainting/video_inpainting.py'])
################################################################################

################################################################################
print('Going to extract masks and frames to LaMa input!')
for vid in vids:
    call(['python', 'utils/export_to_lama.py',
          frames_path,
          os.path.join(masks_path, 'all'),
          lama_path,
          vid['name']
          ])
################################################################################


################################################################################
print('Going to proceede image inpanting by LaMa method!')
os.chdir('image_inpainting')
os.environ['PYTHONPATH'] = os.getcwd()
for vid in vids:
    call(['python', 'bin/predict.py',
          'model.path='+os.getcwd()+'/big-lama',
          'indir='+os.path.join(lama_path, vid['name']),
          'outdir='+os.path.join(inpainting_path, vid['name'])
          ])
################################################################################


################################################################################
print('Going to merge inpaited images to video!')
os.chdir('..')
call(['python', 'utils/merge_inpainted_frames.py',
      inpainting_path
      ])
################################################################################
