import os

def load_ids_and_paths(file_path):
    with open(file_path, 'r') as f:
        data = f.readlines()
    vids = [d.strip().split(' ') for d in data]
    pth = os.path.dirname(os.path.abspath(file_path))
    vids = [{'id' : v[0], 'vid_path' : os.path.join(pth, v[1]), 'name' : v[1].split('.')[0]} for v in vids]
    return vids

base_path = './data'
abs_data_path = os.path.abspath(base_path)
dataset_path = os.path.join(abs_data_path, 'dataset_new')
masks_path = os.path.join(os.path.abspath(base_path), 'person_masks')
frames_path = os.path.join(os.path.abspath(base_path), 'frames')
lama_path = os.path.join(os.path.abspath(base_path), 'lama_input')
inpainting_path = os.path.join(os.path.abspath(base_path), 'inpainting')
mean_scenes_path = os.path.join(os.path.abspath(base_path), 'mean_scenes')
rois_path = os.path.join(os.path.abspath(base_path), 'ROIs')
os.makedirs(masks_path, exist_ok=True)
os.makedirs(frames_path, exist_ok=True)
os.makedirs(lama_path, exist_ok=True)
os.makedirs(inpainting_path, exist_ok=True)
os.makedirs(mean_scenes_path, exist_ok=True)
os.makedirs(rois_path, exist_ok=True)
