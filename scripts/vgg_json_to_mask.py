#!/usr/bin/python
import os
import numpy as np 
import cv2
import matplotlib.pyplot as plt
import json
import argparse
from tqdm import tqdm

def points2mask(x_coord, y_coord, w, h):
    mask = np.zeros((w,h))
    pts = np.asarray(zip(x_coord, y_coord), np.int32)
    pts = pts.reshape((-1, 1, 2)) 
    mask = cv2.fillPoly(mask,[pts],255) 
    return mask

if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument('-r','--root_path', type=str, required=True, help='path to HandleSegmentation root folder')
    ap.add_argument('-j','--json_file', type=str, required=True, help='path to the VGG json file')
    ap.add_argument('-d','--dest', type=str, required=True, help='destination, either dev or train')
    
    args = vars(ap.parse_args())

    json_file = args['json_file']
    dest = args['dest']
    if dest not in ['dev', 'train']:
        print('Please ensure -d, --dest is either dev or train')
        exit()

    base_path = args['root_path']
    dest_path = os.path.join(base_path, dest+'/Masks')

    #Read configuration file
    with open(os.path.join(base_path,'config.json')) as f:
        config = json.load(f)

    w = config['input_w']
    h = config['input_h']

    f = open(json_file)
    json_data = f.readlines()
    data = []
    for line in json_data:
        contents = json.loads(line)
        data.append(contents)

    data = data[0]
    for key in tqdm(data.keys(), desc='Creating Masks'):
        filename = data[key]['filename']
        regions = data[key]['regions']['0']
        x_coord = regions['shape_attributes']['all_points_x']
        y_coord = regions['shape_attributes']['all_points_y']
        mask = points2mask(x_coord, y_coord, w, h)

        mask_name = filename
        mask_path = os.path.join(dest_path, mask_name)
        cv2.imwrite(mask_path, mask)

    print('Done')

