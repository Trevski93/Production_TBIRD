"""
You need to implement all four functions in this file and also put your team info as a variable
Then you should submit the python file with your model class, the state_dict, and this file
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np

import monolayout_model_2 as ModelClasses
# import your model class
# import ...
import process_labels

# Put your transform function here, we will use it for our dataloader
# For bounding boxes task
def get_transform_task1(): 
    return torchvision.transforms.ToTensor()
# For road map task
def get_transform_task2(): 
    return torchvision.transforms.ToTensor()

class ModelLoader():
    # Fill the information for your team
    team_name = 'TBIRD'
    team_number = 14
    round_number = 2
    team_member = ['djk519','tim225','iel230']
    contact_email = '@nyu.edu'

    def __init__(self, model_file='./monolayout/both/weights_0/'):
        
        #cuda 
        device = torch.device("cuda")
        
        #encoder
        self.models = {}
        
        #Encoder
        self.models['encoder'] = ModelClasses.Encoder(18, 256, 306) 
        self.models['encoder'].load_state_dict(torch.load(model_file + 'encoder.pth',map_location="cuda:0"))
        self.models['encoder'].eval()
        
        self.models['encoder'].to(device)
        
        #Decoder - Static
        self.models['static_decoder'] = ModelClasses.Decoder(np.array([64, 64, 128, 256, 512]))
        self.models['static_decoder'].load_state_dict(torch.load(model_file + 'static_decoder.pth',map_location="cuda:0"))
        self.models['static_decoder'].eval()
        
        self.models['static_decoder'].to(device)
        
        #Decoder - Dynamic
        self.models['dynamic_decoder'] = ModelClasses.Decoder(np.array([64, 64, 128, 256, 512]))
        self.models['dynamic_decoder'].load_state_dict(torch.load(model_file + 'dynamic_decoder.pth',map_location="cuda:0"))
        self.models['dynamic_decoder'].eval()
        
        self.models['dynamic_decoder'].to(device)
                
        
    def get_bounding_boxes(self, samples):
        # samples is a cuda tensor with size [batch_size, 6, 3, 256, 306]
        # You need to return a tuple with size 'batch_size' and each element is a cuda tensor [N, 2, 4]
        # where N is the number of object
        
        boxes=[]
        outputs=[]
        for n in range(6):
            outputs.append(torch.argmax(self.models["dynamic_decoder"](self.models["encoder"](samples[:,n])),1))
        output=process_labels.stitch(outputs)
        for btc in range(samples.shape[0]):
            boxes.append(process_labels.image_to_bounding_box(output[btc]))
        
        return tuple(boxes)
        #return torch.rand(1, 15, 2, 4) * 10

    def get_binary_road_map(self, samples):
        # samples is a cuda tensor with size [batch_size, 6, 3, 256, 306]
        # You need to return a cuda tensor with size [batch_size, 800, 800]
        
        outputs=[]
        for n in range(6):
            outputs.append(torch.argmax(self.models["static_decoder"](self.models["encoder"](samples[:,n])),1))
        output=process_labels.stitch(outputs)
        
        return output
        #return torch.rand(1, 800, 800) > 0.5

        



