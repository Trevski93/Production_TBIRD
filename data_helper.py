import os
from PIL import Image

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from helper import convert_map_to_lane_map, convert_map_to_road_map
from process_labels import *

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

# The dataset class for unlabeled data.
class UnlabeledDataset(torch.utils.data.Dataset):
    def __init__(self, image_folder, scene_index, first_dim, transform):
        """
        Args:
            image_folder (string): the location of the image folder
            scene_index (list): a list of scene indices for the unlabeled data 
            first_dim ({'sample', 'image'}):
                'sample' will return [batch_size, NUM_IMAGE_PER_SAMPLE, 3, H, W]
                'image' will return [batch_size, 3, H, W] and the index of the camera [0 - 5]
                    CAM_FRONT_LEFT: 0
                    CAM_FRONT: 1
                    CAM_FRONT_RIGHT: 2
                    CAM_BACK_LEFT: 3
                    CAM_BACK.jpeg: 4
                    CAM_BACK_RIGHT: 5
            transform (Transform): The function to process the image
        """

        self.image_folder = image_folder
        self.scene_index = scene_index
        self.transform = transform

        assert first_dim in ['sample', 'image']
        self.first_dim = first_dim

    def __len__(self):
        if self.first_dim == 'sample':
            return self.scene_index.size * NUM_SAMPLE_PER_SCENE
        elif self.first_dim == 'image':
            return self.scene_index.size * NUM_SAMPLE_PER_SCENE * NUM_IMAGE_PER_SAMPLE
    
    def __getitem__(self, index):
        if self.first_dim == 'sample':
            scene_id = self.scene_index[index // NUM_SAMPLE_PER_SCENE]
            sample_id = index % NUM_SAMPLE_PER_SCENE
            sample_path = os.path.join(self.image_folder, f'scene_{scene_id}', f'sample_{sample_id}') 

            images = []
            for image_name in image_names:
                image_path = os.path.join(sample_path, image_name)
                image = Image.open(image_path)
                images.append(self.transform(image))
            image_tensor = torch.stack(images)
            
            return image_tensor

        elif self.first_dim == 'image':
            scene_id = self.scene_index[index // (NUM_SAMPLE_PER_SCENE * NUM_IMAGE_PER_SAMPLE)]
            sample_id = (index % (NUM_SAMPLE_PER_SCENE * NUM_IMAGE_PER_SAMPLE)) // NUM_IMAGE_PER_SAMPLE
            image_name = image_names[index % NUM_IMAGE_PER_SAMPLE]

            image_path = os.path.join(self.image_folder, f'scene_{scene_id}', f'sample_{sample_id}', image_name) 
            
            image = Image.open(image_path)

            return self.transform(image), index % NUM_IMAGE_PER_SAMPLE

# The dataset class for labeled data.
class LabeledDataset(torch.utils.data.Dataset):    
    def __init__(self, image_folder, annotation_file, scene_index, transform, extra_info=True):
        """
        Args:
            image_folder (string): the location of the image folder
            annotation_file (string): the location of the annotations
            scene_index (list): a list of scene indices for the unlabeled data 
            transform (Transform): The function to process the image
            extra_info (Boolean): whether you want the extra information
        """
        
        self.image_folder = image_folder
        self.annotation_dataframe = pd.read_csv(annotation_file)
        self.scene_index = scene_index
        self.transform = transform
        self.extra_info = extra_info
    
    def __len__(self):
        return self.scene_index.size * NUM_SAMPLE_PER_SCENE

    def __getitem__(self, index):
        scene_id = self.scene_index[index // NUM_SAMPLE_PER_SCENE]
        sample_id = index % NUM_SAMPLE_PER_SCENE
        sample_path = os.path.join(self.image_folder, f'scene_{scene_id}', f'sample_{sample_id}') 

        images = []
        for image_name in image_names:
            image_path = os.path.join(sample_path, image_name)
            image = Image.open(image_path) 

            images.append(self.transform(image))
        image_tensor = torch.stack(images)
        
        
        data_entries = self.annotation_dataframe[(self.annotation_dataframe['scene'] == scene_id) & (self.annotation_dataframe['sample'] == sample_id)]
        corners = data_entries[['fl_x', 'fr_x', 'bl_x', 'br_x', 'fl_y', 'fr_y','bl_y', 'br_y']].to_numpy()
        categories = data_entries.category_id.to_numpy()
        
        ego_path = os.path.join(sample_path, 'ego.png')
        ego_image = Image.open(ego_path)
        ego_image = ego_image.resize((64,64)) #Trevor - Resize image so we can compute loss static input
        ego_image = torchvision.transforms.functional.to_tensor(ego_image)
        road_image = convert_map_to_road_map(ego_image)
        
        target = {}
        target['bounding_box'] = torch.as_tensor(corners).view(-1, 2, 4)
        target['category'] = torch.as_tensor(categories)

        if self.extra_info:
            actions = data_entries.action_id.to_numpy()
            # You can change the binary_lane to False to get a lane with 
            lane_image = convert_map_to_lane_map(ego_image, binary_lane=True)
            
            extra = {}
            extra['action'] = torch.as_tensor(actions)
            extra['ego_image'] = ego_image
            extra['lane_image'] = lane_image

            return image_tensor, target, road_image, extra
        
        else:
            return image_tensor, target, road_image

   
class HybridDataset(torch.utils.data.Dataset):    
    def __init__(self, image_folder, annotation_file, scene_index, transform, extra_info=False):
        """
        Args:
            image_folder (string): the location of the image folder
            annotation_file (string): the location of the annotations
            scene_index (list): a list of scene indices for the unlabeled data 
            transform (Transform): The function to process the image
            extra_info (Boolean): whether you want the extra information
        """
        
        self.image_folder = image_folder
        self.annotation_dataframe = pd.read_csv(annotation_file)
        self.scene_index = scene_index
        self.transform = transform
        self.extra_info = extra_info
    
    def __len__(self):
        return self.scene_index.size * NUM_SAMPLE_PER_SCENE * NUM_IMAGE_PER_SAMPLE

    def __getitem__(self, index):
        
        #get image
        scene_id = self.scene_index[index // (NUM_SAMPLE_PER_SCENE * NUM_IMAGE_PER_SAMPLE)]
        sample_id = (index % (NUM_SAMPLE_PER_SCENE * NUM_IMAGE_PER_SAMPLE)) // NUM_IMAGE_PER_SAMPLE
        image_name = image_names[index % NUM_IMAGE_PER_SAMPLE]
        
        scene_id_discr=np.random.choice(self.scene_index)
        sample_id_discr=np.random.choice(126)

        sample_path = os.path.join(self.image_folder, f'scene_{scene_id}', f'sample_{sample_id}')
        image_path = os.path.join(self.image_folder, f'scene_{scene_id}', f'sample_{sample_id}', image_name) 
        sample_path_discr = os.path.join(self.image_folder, f'scene_{scene_id_discr}', f'sample_{sample_id_discr}')
            
        image = Image.open(image_path)
        
        #get Labels
        data_entries = self.annotation_dataframe[(self.annotation_dataframe['scene'] == scene_id) & (self.annotation_dataframe['sample'] == sample_id)]
        data_entries_discr = self.annotation_dataframe[(self.annotation_dataframe['scene'] == scene_id_discr) & (self.annotation_dataframe['sample'] == sample_id_discr)]
        corners = data_entries[['fl_x', 'fr_x', 'bl_x', 'br_x', 'fl_y', 'fr_y','bl_y', 'br_y']].to_numpy()
        corners_discr = data_entries_discr[['fl_x', 'fr_x', 'bl_x', 'br_x', 'fl_y', 'fr_y','bl_y', 'br_y']].to_numpy()
        
        
        ego_path = os.path.join(sample_path, 'ego.png')
        ego_path_discr = os.path.join(sample_path_discr, 'ego.png')
        ego_image = Image.open(ego_path)
        ego_image_discr = Image.open(ego_path_discr)
        #ego_image = ego_image.resize((64,64)) #Trevor - Resize image so we can compute loss static input
        ego_image = torchvision.transforms.functional.to_tensor(ego_image)
        ego_image_discr = torchvision.transforms.functional.to_tensor(ego_image_discr)
        road_image = split_with_index(convert_map_to_road_map(ego_image),index % NUM_IMAGE_PER_SAMPLE)
        road_image_discr = split_with_index(convert_map_to_road_map(ego_image_discr),index % NUM_IMAGE_PER_SAMPLE)
        
        #target = {}
        bounding_box = bounding_box_to_image(torch.as_tensor(corners).view(-1, 2, 4))
        bounding_box_discr = bounding_box_to_image(torch.as_tensor(corners_discr).view(-1, 2, 4))
        target = split_with_index(bounding_box,index % NUM_IMAGE_PER_SAMPLE)
        target_discr = split_with_index(bounding_box_discr,index % NUM_IMAGE_PER_SAMPLE)


        if self.extra_info:
            actions = data_entries.action_id.to_numpy()
            # You can change the binary_lane to False to get a lane with 
            lane_image = convert_map_to_lane_map(ego_image, binary_lane=True)
            
            extra = {}
            
            categories = data_entries.category_id.to_numpy()
            extra['category'] = torch.as_tensor(categories)
            
            extra['action'] = torch.as_tensor(actions)
            extra['ego_image'] = ego_image
            extra['lane_image'] = lane_image

            return self.transform(image), target, road_image, target_discr, road_image_discr, extra
        
        else:
            return self.transform(image), target, road_image, target_discr, road_image_discr

        
#Dataset from Images
def process_discr(topview):
    topview_n = torch.zeros((2, 658, 658))
    topview_n[1, topview==1] = 1.
    topview_n[0, topview==0] = 1.
    return topview_n

class HybridDataset2(torch.utils.data.Dataset):    
    def __init__(self, image_folder, annotation_file, scene_index, transform, extra_info=False, discr_run=True):
        """
        Args:
            image_folder (string): the location of the image folder
            annotation_file (string): the location of the annotations
            scene_index (list): a list of scene indices for the unlabeled data 
            transform (Transform): The function to process the image
            extra_info (Boolean): whether you want the extra information
            discr_run (Boolean): are we using a discriminator for this run
        """
        
        self.image_folder = image_folder
        self.annotation_dataframe = pd.read_csv(annotation_file)
        self.scene_index = scene_index
        self.transform = transform
        self.extra_info = extra_info
        self.discr_run = discr_run
    
    def __len__(self):
        return self.scene_index.size * NUM_SAMPLE_PER_SCENE * NUM_IMAGE_PER_SAMPLE

    def __getitem__(self, index):
        
        #get image
        scene_id = self.scene_index[index // (NUM_SAMPLE_PER_SCENE * NUM_IMAGE_PER_SAMPLE)]
        sample_id = (index % (NUM_SAMPLE_PER_SCENE * NUM_IMAGE_PER_SAMPLE)) // NUM_IMAGE_PER_SAMPLE
        image_name = image_names[index % NUM_IMAGE_PER_SAMPLE]
        
        sample_path = os.path.join(self.image_folder, f'scene_{scene_id}', f'sample_{sample_id}')
        image_path = os.path.join(self.image_folder, f'scene_{scene_id}', f'sample_{sample_id}', image_name)
            
        image = Image.open(image_path)
        
        
        #get Labels
        ego_path = os.path.join(sample_path, f'ego_{index % NUM_IMAGE_PER_SAMPLE}.png')
        ego_image = Image.open(ego_path)
        ego_image = torchvision.transforms.functional.to_tensor(ego_image)
        ego_image=ego_image.squeeze().long()
        
        target_path = os.path.join(sample_path, f'target_{index % NUM_IMAGE_PER_SAMPLE}.png')
        target_image = Image.open(target_path)
        target_image = torchvision.transforms.functional.to_tensor(target_image)
        target_image=target_image.squeeze().long()
        
        if self.discr_run:
            scene_id_discr=np.random.choice(self.scene_index)
            sample_id_discr=np.random.choice(126)
            sample_path_discr = os.path.join(self.image_folder, f'scene_{scene_id_discr}', f'sample_{sample_id_discr}')
            ego_path_discr = os.path.join(sample_path_discr, f'ego_{index % NUM_IMAGE_PER_SAMPLE}.png')
            ego_image_discr = Image.open(ego_path_discr)
            ego_image_discr = torchvision.transforms.functional.to_tensor(ego_image_discr)
            ego_image_discr = process_discr(ego_image_discr.squeeze())
            target_path_discr = os.path.join(sample_path_discr, f'target_{index % NUM_IMAGE_PER_SAMPLE}.png')
            target_image_discr = Image.open(target_path_discr)
            target_image_discr = torchvision.transforms.functional.to_tensor(target_image_discr)
            target_image_discr = process_discr(target_image_discr.squeeze())

            return self.transform(image), target_image, ego_image, target_image_discr, ego_image_discr        

        
        elif self.extra_info:
            data_entries = self.annotation_dataframe[(self.annotation_dataframe['scene'] == scene_id) & (self.annotation_dataframe['sample'] == sample_id)]
            categories = data_entries.category_id.to_numpy()
            target['category'] = torch.as_tensor(categories)
        
            actions = data_entries.action_id.to_numpy()
            # You can change the binary_lane to False to get a lane with 
            lane_image = convert_map_to_lane_map(ego_image, binary_lane=True)
            
            extra = {}
            extra['action'] = torch.as_tensor(actions)
            extra['ego_image'] = ego_image
            extra['lane_image'] = lane_image

            return self.transform(image), target_image, ego_image, extra
        
        else:
            return self.transform(image), target_image, ego_image
