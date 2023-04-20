# BEWARE: class_id is increased +1 to challenge report (not list index)
import pickle
import json
import numpy as np
import math
import os
import sys
import collections
import argparse
parser = argparse.ArgumentParser('ROI PATH')
parser.add_argument('--roi_path', type=str, required=True, help='roi_path')
args = parser.parse_args()

def load_ids_and_paths(file_path):
    with open(file_path, 'r') as f:
        data = f.readlines()
    vids = [d.strip().split(' ') for d in data]
    pth = os.path.dirname(os.path.abspath(file_path))
    vids = [{'id' : v[0], 'vid_path' : os.path.join(pth, v[1]), 'name' : v[1].split('.')[0]} for v in vids]
    return vids


vids = load_ids_and_paths(sys.argv[3])


def get_mid_point(detection):
    return (int((detection[0]+detection[2])/2), int((detection[1]+detection[3])/2))

def roi_midpoint(frame_pos):
    roi_path = args.roi_path
    roi_filename = roi_path + '/' + '{:04d}'.format(frame_pos) + '.json'
    with open(roi_filename,'r') as f:
        roi = json.load(f)
    mid_point_roi = get_mid_point(roi)
    return mid_point_roi

def merge_tracks_by_ID(detection_track_data):
    # 'merge' by track ID
    tracks = {}
    track_id=0
    prev_frame = 0
    new_track_data = {}
    for key in detection_track_data:
        if detection_track_data[key]:
            new_track_data[key] = detection_track_data[key]

    sorted_tracks = collections.OrderedDict(sorted(detection_track_data.items()))

    for frame_num in sorted_tracks:
        if track_id not in tracks:
            tracks[int(track_id)] = []

        for det in sorted_tracks[frame_num]:
            if frame_num - prev_frame < 50:
                tracks[int(track_id)].append({
                        'pos': get_mid_point(det['det']),
                        'frame': frame_num,
                        'class': int(det['class']),
                        'det_conf': det['det_conf'],
                        'class_conf': det['class_conf']

                })
            else:
                track_id = track_id+1
                tracks[int(track_id)] = []
                tracks[track_id].append({
                        'pos': get_mid_point(det['det']),
                        'frame': frame_num,
                        'class': int(det['class']),
                        'det_conf': det['det_conf'],
                        'class_conf': det['class_conf']
                })

            prev_frame = frame_num

    del tracks[0]
    print(tracks.keys())
    return tracks


def get_track_classes(tracks):
    tracks_classes = {}

    for t in tracks:
        tracks_classes[t] = {'class' : None, 'dets' : [], 'probs_of_classes' : {}}
        for det in tracks[t]:
            tracks_classes[t]['dets'].append({'class' : det['class'], 'class_conf' : det['class_conf']})

    for t in tracks_classes:
        # insert probability for eaech detection (by class)
        for det in tracks_classes[t]['dets']:
            if det['class'] not in tracks_classes[t]['probs_of_classes']:
                tracks_classes[t]['probs_of_classes'][det['class']] = {'confs' : []}
            tracks_classes[t]['probs_of_classes'][det['class']]['confs'].append(det['class_conf'])


        # compute count of detections for each class and mean confidence value
        for class_id in tracks_classes[t]['probs_of_classes']:
            tracks_classes[t]['probs_of_classes'][class_id]['dets_count'] = len(tracks_classes[t]['probs_of_classes'][class_id]['confs'])
            tracks_classes[t]['probs_of_classes'][class_id]['mean'] = np.mean(tracks_classes[t]['probs_of_classes'][class_id]['confs'])


        # count of all class detections for track
        all_track_dets_cnt = 0
        for class_id in tracks_classes[t]['probs_of_classes']:
            all_track_dets_cnt += tracks_classes[t]['probs_of_classes'][class_id]['dets_count']

        # weighted mean of single classes (mean confidence value weighted by count of detections)
        for class_id in tracks_classes[t]['probs_of_classes']:
            tracks_classes[t]['probs_of_classes'][class_id]['weighted_mean'] = \
                    tracks_classes[t]['probs_of_classes'][class_id]['mean'] * \
                    (tracks_classes[t]['probs_of_classes'][class_id]['dets_count']/all_track_dets_cnt)

        best_class = None
        best_class_value = 0.0
        for class_id in tracks_classes[t]['probs_of_classes']:
            if tracks_classes[t]['probs_of_classes'][class_id]['weighted_mean'] > best_class_value:
                best_class_value = tracks_classes[t]['probs_of_classes'][class_id]['weighted_mean']
                best_class = class_id
        tracks_classes[t]['class'] = best_class


    with open("sample_final.json", "w") as outfile:
        json.dump(tracks_classes, outfile)
    return tracks_classes


def merge_by_position_and_class(tracks, tracks_classes):
    # merge close detections with came class (in some spatial and time 'window')
    time_frame = 90
    position_frame = 100
    # print('tracks',tracks)
    # print('tracks_classes',tracks_classes)
    with open("track_class_before_merge.json", "w") as outfile:
        json.dump(tracks_classes, outfile)


    to_remove = []

    for pos_1, t_1 in enumerate(tracks):
        for pos_2, t_2 in enumerate(tracks):
            if pos_2 <= pos_1:
                continue
            dist = np.linalg.norm(np.array(tracks[t_1][-1]['pos']) - np.array(tracks[t_2][0]['pos']))
            if dist < position_frame and \
             tracks_classes[t_1]['class'] == tracks_classes[t_2]['class'] and \
             np.abs(tracks[t_1][-1]['frame'] - tracks[t_2][0]['frame']) < time_frame:
                tracks[t_1] += tracks[t_2]
                tracks_classes[t_1] = {**tracks_classes[t_1], **tracks_classes[t_2]}
                to_remove.append(t_2)


    for t in to_remove:
        try:
            del(tracks[t])
            del(tracks_classes[t])
        except:
            pass




def export_submission(tracks, tracks_classes, out_file, scene_num):
    '''
        BEWARE: class number + 1 (to evaluation)
    '''

    print(os.getcwd())
    tracks_time = {}
    # store frame numbers for all track detections
    for t in tracks:
        tracks_time[t] = {'class' : tracks_classes[t]['class'], 'frames_nums' : [], 'det_conf':[]}
        for det in tracks[t]:
            if det['class'] == tracks_classes[t]['class']:
                tracks_time[t]['frames_nums'].append(det['frame'])
                tracks_time[t]['det_conf'].append(det['det_conf'])
    with open("sample_10.json", "w") as outfile:
        json.dump(tracks_time, outfile)

    # compute time of track detection
    txt = []
    class_list = []
    for t in tracks:
        for frame in tracks_time[t]['frames_nums']:
            det_conf_array = np.array(tracks_time[t]['det_conf'])
            mean_conf = np.mean(det_conf_array)
            closest_frame = tracks_time[t]['frames_nums'][np.argmin(np.abs(det_conf_array - mean_conf))]
        if tracks_time[t]['class'] in class_list:
            continue
        txt.append('{:d} {:d} {:d}'.format(scene_num, tracks_time[t]['class'] + 1, closest_frame))
        class_list.append(tracks_time[t]['class'])

    final_txt = sorted(txt, key=lambda x: int(x.split(" ")[-1]))
    for i in final_txt:
        out_file.write(i + '\n')


roi_expand = 0.1
scenes = sorted(os.listdir(sys.argv[1]))
scenes = [s.split('.')[0] for s in scenes]

out_file = open(sys.argv[2], 'w')

for pos, s in enumerate(scenes):
    pth = os.path.join(sys.argv[1], s + '.pkl')

    with open(pth, 'rb') as f:
        detection_track_data = pickle.load(f)

    tracks = merge_tracks_by_ID(detection_track_data)
    print('Total tracks before filtering:', len(tracks))

    print('Tracks after position filtering:', len(tracks))
    tracks_classes = get_track_classes(tracks)

    merge_by_position_and_class(tracks, tracks_classes)

    print('Tracks after merging by position and class:', len(tracks))

    scene_num = 0
    for v in vids:
        if v['name'] == s:
            scene_num = v['id']
            break
    export_submission(tracks, tracks_classes, out_file, int(scene_num))

out_file.close()