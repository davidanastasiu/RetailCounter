from info import *
import argparse
import os
from subprocess import call


parser = argparse.ArgumentParser('Create dataset for YOLOX training')
parser.add_argument("--t_4_path", type=str, default=None, help="Path to root of Track4 data", required=True)
args = parser.parse_args()


os.makedirs(dataset_path, exist_ok=True)

call(['python', 'utils/generate_dataset_coco_format.py',
      '--t_4_train_path', args.t_4_path,
      '--store_path', '/data/aicity/aic23/track4/PersonGONE/images/',
      '--classes_path', os.path.join(abs_data_path, 'classes.json'),
      '--annotation_path','/data/aicity/aic23/track4/PersonGONE/train.json',
      '--count', '130000'
      ])

print('Train dataset DONE! Stored in:', os.path.join(dataset_path, 'train'))



print('Creating validation dataset in COCO format')
os.makedirs(os.path.join(dataset_path, 'validation'), exist_ok=True)
call(['python', 'utils/generate_dataset_coco_format.py',
      '--t_4_train_path', args.t_4_path,
      '--store_path', os.path.join(dataset_path, 'validation'),
      '--classes_path', os.path.join(abs_data_path, 'classes.json'),
      '--annotation_path', os.path.join(dataset_path, 'annotations', 'validation.json'),
      '--count', '20000'])
print('Validation dataset DONE! Stored in:', os.path.join(dataset_path, 'validation'))
