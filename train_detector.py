from info import *
import os
from subprocess import call
import argparse
import shutil

parser = argparse.ArgumentParser('Train YOLOX-L detector on generated dataset!')
parser.add_argument('--experiment_name', type=str, default='yolov8', help='Optional name of experiment')
parser.add_argument('--batch_size', type=int, default=3, help='batch size')
parser.add_argument('--epochs', type=int, default=50, help='count of training epochs')
parser.add_argument('--device', default= 2, type=int, help='device for training')
parser.add_argument('--cuda_path', default='/usr/local/cuda-11.3/lib64', type=str, help='device for training')
parser.add_argument("--resume", default=True, action="store_false", help="resume training")
args = parser.parse_args()

os.environ['LD_LIBRARY_PATH'] = args.cuda_path
os.chdir('detector_and_tracker')
os.environ['PYTHONPATH'] = os.getcwd()

call(['python', 'tools/train.py',
      '-expn', args.experiment_name,
      '--dataset_path', dataset_path,
      '-b', str(args.batch_size),
      '--epochs', str(args.epochs),
      '-d', str(args.device),
      '-f', 'exps/aic_yolov_8.py',
      # '--cache'
      # '--resume'
      ])

final_path = os.path.join('checkpoints', args.experiment_name)
os.makedirs(final_path)
shutil.move(os.path.join('YOLOV8_outputs', args.experiment_name, 'best_ckpt.pth'), os.path.join(final_path, 'best_ckpt_pth'))
shutil.rmtree(os.path.join('YOLOV8_outputs', args.experiment_name))
