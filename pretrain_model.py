import numpy as np
import time
# import eval_segm
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
#from tensorboardX import SummaryWriter
# from validate import *
import os
import tqdm
import argparse
#from monolayout_eval import *                              commented out
# from utils import *
# from kitti_utils import *
# from layers import *
# from metric.iou import IoU
# from fcn_iou import *
#from IPython import embed
#from torch.autograd import Variable
from monolayout_resnet_encoder import ResnetEncoder


import monolayout_model_2 as model

from monolayout_model_2 import Conv3x3

from monolayout_train import data_loader

###

import pandas as pd

import torchvision

#from data_helper import UnlabeledDataset, LabeledDataset, HybridDataset, HybridDataset2
#from helper import collate_fn, compute_ts_road_map

#from process_labels import *







class Encoder(nn.Module):
    def __init__(self, num_layers, img_ht, img_wt, pretrained=False, num_input_images=1):
        super(Encoder, self).__init__()
        
        self.resnet_encoder = ResnetEncoder(num_layers, pretrained, num_input_images)#opt.weights_init == "pretrained"))
        num_ch_enc = self.resnet_encoder.num_ch_enc
        #convolution to reduce depth and size of features before fc
        self.conv1 = Conv3x3(num_ch_enc[-1], 128)
        print(num_ch_enc[-1])
        self.conv2 = Conv3x3(128, 128)
        self.pool = nn.MaxPool2d(2, 2, return_indices=True)
        
        #fully connected
        curr_h = img_ht//(2**6)
        curr_w = img_wt//(2**6)
        features_in = curr_h*curr_w*128
        self.fc_mu = torch.nn.Linear(features_in, 2048)
        self.fc_sigma = torch.nn.Linear(features_in, 2048)
        self.fc = torch.nn.Linear(features_in, 2048)
        self.ffc = nn.Linear(128*2*2, 128*2*2*2)
    
    
    def forward(self, x, is_training= True):
        
        batch_size, c, h, w = x.shape
        x = self.resnet_encoder(x)[-1]
        x = self.conv1(x)
        x, idx1 = self.pool(x)
        x = self.conv2(x)
        x, idx2 = self.pool(x)
        x = x.view(-1, 128*2*2)
        x = self.ffc(x)
        return x, idx1, idx2


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        
        
        #self.conv1 = model.Conv3x3(128,128)
        #self.unpool1 = nn.MaxUnpool2d(2,2)
        #self.unpool2 = nn.MaxUnpool2d(2,2)
        #self.conv2 = model.Conv3x3(128, 512)
        
        self.conv2 = nn.ConvTranspose2d(128, 512, 3, 2)
        
        self.convt1 = nn.ConvTranspose2d(512, 512, 3, 2)
        self.convt2 = nn.ConvTranspose2d(512, 256, 3, 2)
        self.convt3 = nn.ConvTranspose2d(256, 256, 3, 2)
        self.convt4 = nn.ConvTranspose2d(256, 128, 3, 2)
        self.convt5 = nn.ConvTranspose2d(128, 128, 3, 2)
        self.convt6 = nn.ConvTranspose2d(128, 64, 3, 2)
        self.convt7 = nn.ConvTranspose2d(64, 64, 3, 2)
        self.convt8 = nn.ConvTranspose2d(64, 3, 7, 2)
        self.rel = nn.ReLU()
        self.sig = nn.Sigmoid()
    
    

    def forward(self, x):
        #x = self.unpool1(x, self.idx2)
        #x = self.conv1(x)
        #x = self.unpool2(x)
        x = self.conv2(x)
        #for i in range(3):
        x = self.convt1(x)
        x = self.rel(x)

        x = self.convt2(x)
        x = self.rel(x)
        #for i in range(3):
        x = self.convt3(x)
        x = self.rel(x)

        x = self.convt4(x)
        x = self.rel(x)
        #for i in range(2):
        x = self.convt5(x)
        x = self.rel(x)

        x = self.convt6(x)
        x = self.rel(x)
        #for i in range(2):
        x = self.convt7(x)
        x = self.rel(x)

        x = self.convt8(x)
        x = F.interpolate(x, size=(256,306))
        x = self.sig(x)
        return x



############################################################################################
############################################################################################
############################################################################################

class Pre_VAE(nn.Module):
    def __init__(self, num_layers, img_ht, img_wt, batch_size, pretrained=False, num_input_images=1):
        super(Pre_VAE, self).__init__()

        self.batch_size = batch_size
        self.resnet_encoder = ResnetEncoder(num_layers, pretrained, num_input_images)#opt.weights_init == "pretrained"))
        num_ch_enc = self.resnet_encoder.num_ch_enc
        #convolution to reduce depth and size of features before fc
        self.conv1 = Conv3x3(num_ch_enc[-1], 128)
        #print(num_ch_enc[-1])
        self.conv2 = Conv3x3(128, 128)
        self.pool = nn.MaxPool2d(2, 2, return_indices=True)
        
        #fully connected
        curr_h = img_ht//(2**6)
        curr_w = img_wt//(2**6)
        features_in = curr_h*curr_w*128
        self.fc_mu = torch.nn.Linear(features_in, 2048)
        self.fc_sigma = torch.nn.Linear(features_in, 2048)
        self.fc = torch.nn.Linear(features_in, 2048)
        self.ffc = nn.Linear(128*2*2, 128*2*2*2)
    
    
        self.convd2 = nn.ConvTranspose2d(128, 512, 3, 2)
        self.convt1 = nn.ConvTranspose2d(512, 512, 3, 2)
        self.convt2 = nn.ConvTranspose2d(512, 256, 3, 2)
        self.convt3 = nn.ConvTranspose2d(256, 256, 3, 2)
        self.convt4 = nn.ConvTranspose2d(256, 128, 3, 2)
        self.convt5 = nn.ConvTranspose2d(128, 128, 3, 2)
        self.convt6 = nn.ConvTranspose2d(128, 64, 3, 2)
        self.convt7 = nn.ConvTranspose2d(64, 64, 3, 2)
        self.convt8 = nn.ConvTranspose2d(64, 3, 7, 2)
        self.rel = nn.ReLU()
        self.sig = nn.Sigmoid()
    
    


    def reparameterise(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x):
        
        batch_size, c, h, w = x.shape
        x = self.resnet_encoder(x)[-1]
        x = self.conv1(x)
        x, idx1 = self.pool(x)
        x = self.conv2(x)
        x, idx2 = self.pool(x)
        #print(x.shape)
        x = x.view(-1, 128*2*2)
        x = self.ffc(x)
        
        
        
        mu_logvar = x.view(self.batch_size, 2, 128*2*2)
        mu = mu_logvar[:, 0, :]
        logvar = mu_logvar[:, 1, :]
        z = self.reparameterise(mu, logvar)
        z = z.view(self.batch_size, 128, 2, 2)
        
        
        #x = self.unpool1(x, self.idx2)
        #x = self.conv1(x)
        #x = self.unpool2(x)
        z = self.convd2(z)
        #for i in range(3):
        z = self.convt1(z)
        z = self.rel(z)
        
        z = self.convt2(z)
        z = self.rel(z)
        #for i in range(3):
        z = self.convt3(z)
        z = self.rel(z)
        
        z = self.convt4(z)
        z = self.rel(z)
        #for i in range(2):
        z = self.convt5(z)
        z = self.rel(z)
        
        z = self.convt6(z)
        z = self.rel(z)
        #for i in range(2):
        z = self.convt7(z)
        z = self.rel(z)
        
        z = self.convt8(z)
        z = F.interpolate(z, size=(256,306))
        z = self.sig(z)
        
        return z, mu, logvar





############################################################################################
############################################################################################
############################################################################################
############################################################################################






















class Pre_VAE2(nn.Module):
    def __init__(self):
        super(Pre_VAE2, self).__init__()

        self.encoder = Encoder(18, 306, 256)
        self.decoder = Decoder()

    def reparameterise(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x):
        x = self.encoder.forward(x)
        print(x)
        mu_logvar = x.view(-1, 2, 128*2*2)
        mu = mu_logvar[:, 0, :]
        logvar = mu_logvar[:, 1, :]
        z = self.reparameterise(mu, logvar)
        return self.decoder.forward(z), mu, logvar

                                     




