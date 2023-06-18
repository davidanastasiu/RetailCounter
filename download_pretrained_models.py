import wget
import os
import argparse
import zipfile
import shutil
from subprocess import call


parser = argparse.ArgumentParser('Download pretrained models weights')
parser.add_argument("--detector", action='store_true')
args = parser.parse_args()

################################################################################
# Ensure correct paths
call(['python', 'info.py'])
################################################################################

################################################################################
print('Downloading LaMa (inpainting model) weights!')
lama_path = os.path.join('image_inpainting', 'big-lama')
if os.path.exists(lama_path):
    print('Already downloaded - skipping.')
else:
    os.makedirs(lama_path,exist_ok=True)
    wget.download('https://nextcloud.fit.vutbr.cz/s/55rG5fDDzmnCFPo/download?path=%2F&files=big-lama.zip')
    with zipfile.ZipFile('big-lama.zip', 'r') as zip_ref:
        zip_ref.extractall('image_inpainting')
    os.remove('big-lama.zip')
    print('\nLama downloaded!')
################################################################################

# ################################################################################
print('Downloading instance segmentation models weights!')
instance_path = os.path.join('instance_segmentation', 'checkpoints')
os.makedirs(instance_path,exist_ok=True)
seg_path = os.path.join(instance_path, 'detectors_htc_r101_20e_coco_20210419_203638-348d533b.pth')
if os.path.exists(seg_path):
    print('DetectoRS already downloaded - skipping.')
else:
    wget.download('https://nextcloud.fit.vutbr.cz/s/55rG5fDDzmnCFPo/download?path=%2F&files=detectors_htc_r101_20e_coco_20210419_203638-348d533b.pth')
    shutil.move('detectors_htc_r101_20e_coco_20210419_203638-348d533b.pth', seg_path)
    print('\nDetectoRS downloaded!')
################################################################################

################################################################################
seg_path = os.path.join(instance_path, 'htc_x101_64x4d_fpn_dconv_c3-c5_mstrain_400_1400_16x1_20e_coco_20200312-946fd751.pth')
if os.path.exists(seg_path):
    print('HTC already downloaded - skipping.')
else:
    wget.download('https://nextcloud.fit.vutbr.cz/s/55rG5fDDzmnCFPo/download?path=%2F&files=htc_x101_64x4d_fpn_dconv_c3-c5_mstrain_400_1400_16x1_20e_coco_20200312-946fd751.pth')
    shutil.move('htc_x101_64x4d_fpn_dconv_c3-c5_mstrain_400_1400_16x1_20e_coco_20200312-946fd751.pth', seg_path)
    print('\nHTC downloaded!')
################################################################################

################################################################################
seg_path = os.path.join(instance_path, 'point_rend_r50_caffe_fpn_mstrain_3x_coco-e0ebb6b7.pth')
if os.path.exists(seg_path):
    print('PointRend already downloaded - skipping.')
else:
    wget.download('https://nextcloud.fit.vutbr.cz/s/55rG5fDDzmnCFPo/download?path=%2F&files=point_rend_r50_caffe_fpn_mstrain_3x_coco-e0ebb6b7.pth')
    shutil.move('point_rend_r50_caffe_fpn_mstrain_3x_coco-e0ebb6b7.pth', seg_path)
    print('\nPointRend downloaded!')
################################################################################

################################################################################
seg_path = os.path.join(instance_path, 'yolact_r101_1x8_coco_20200908-4cbe9101.pth')
if os.path.exists(seg_path):
    print('YOLACT already downloaded - skipping.')
else:
    wget.download('https://nextcloud.fit.vutbr.cz/s/55rG5fDDzmnCFPo/download?path=%2F&files=yolact_r101_1x8_coco_20200908-4cbe9101.pth')
    shutil.move('yolact_r101_1x8_coco_20200908-4cbe9101.pth', seg_path)
    print('\nYOLACT downloaded!')
################################################################################

################################################################################
if args.detector:
    det_path = os.path.join('detector_and_tracker', 'checkpoints', 'yolox_l')
    os.makedirs(det_path,exist_ok=True)
    if os.path.exists(os.path.join(det_path, 'best_ckpt.pth')):
        print('Detector already exists - skipping.')
    else:
        os.makedirs(det_path, exist_ok=True)
        print('Downloading pretraind YOLOX-L detector weights!')
        wget.download('https://nextcloud.fit.vutbr.cz/s/55rG5fDDzmnCFPo/download?path=%2F&files=best_ckpt.pth')
        shutil.move('best_ckpt.pth', os.path.join(det_path, 'best_ckpt.pth'))
        print('\nYOLOX-L downloaded!')
################################################################################
