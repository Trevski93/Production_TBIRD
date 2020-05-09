import os
from PIL import Image

import argparse

import numpy as np
import pandas as pd
import imageio

import imageio

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from helper import convert_map_to_lane_map, convert_map_to_road_map
from process_labels import *

def get_args():
    parser = argparse.ArgumentParser(description="Paths for getting data")
    parser.add_argument("--image_folder", type=str, default=None, required=True,
                         help="Path to folder with images")
    parser.add_argument("--annotation_csv", type=str, default=None, required=True,
                         help="Path to file with target data")
    parser.add_argument("--start_scene", type=int, default=106,
                         help="scene to start script")
    parser.add_argument("--end_scene", type=int, default=134,
                         help="scene to end script")
    parser.add_argument("--start_sample", type=int, default=0,
                         help="sample script")

    return parser.parse_args()


def process_and_save_images(image_folder,annotation_csv,start_scene,end_scene,start_sample):
    NUM_SAMPLE_PER_SCENE = 126
    NUM_IMAGE_PER_SAMPLE = 6
    image_names = [
        'CAM_FRONT_LEFT.jpeg',
        'CAM_FRONT.jpeg',
        'CAM_FRONT_RIGHT.jpeg',
        'CAM_BACK_LEFT.jpeg',
        'CAM_BACK.jpeg',
        'CAM_BACK_RIGHT.jpeg',
        ]
    labeled_scene_index = np.arange(start_scene, end_scene)
    
    annotation_dataframe = pd.read_csv(annotation_csv)

    for scene_id in labeled_scene_index:
        print("starting scene",scene_id)
        for sample_id in range(start_sample,NUM_SAMPLE_PER_SCENE):
            sample_path = os.path.join(image_folder, f'scene_{scene_id}', f'sample_{sample_id}')
            data_entries = annotation_dataframe[(annotation_dataframe['scene'] == scene_id) & (annotation_dataframe['sample'] == sample_id)]
            corners = data_entries[['fl_x', 'fr_x', 'bl_x', 'br_x', 'fl_y', 'fr_y','bl_y', 'br_y']].to_numpy()
            ego_path = os.path.join(sample_path, 'ego.png')
            ego_image = Image.open(ego_path)
            ego_image = torchvision.transforms.functional.to_tensor(ego_image)
            bounding_box = bounding_box_to_image(torch.as_tensor(corners).view(-1, 2, 4))
            for n in range(6):
                road_image = split_with_index(convert_map_to_road_map(ego_image),n)
                r_path=os.path.join(sample_path, f'ego_{n}.png')
                imageio.imwrite(r_path,road_image.to(torch.uint8),format="png")
                target = split_with_index(bounding_box,n)
                t_path=os.path.join(sample_path, f'target_{n}.png')
                imageio.imwrite(t_path,target.to(torch.uint8),format="png")
            if sample_id % 20 == 19:
                print("finished sample",sample_id)
            
def main():
    args = get_args()
    process_and_save_images(image_folder=args.image_folder,annotation_csv=args.annotation_csv,start_scene=args.start_scene,end_scene=args.end_scene,start_sample=args.start_sample)

if __name__ == "__main__":
    main()
