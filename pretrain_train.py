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
from monolayout_eval import *
# from utils import *
# from kitti_utils import *
# from layers import *
# from metric.iou import IoU
# from fcn_iou import *
from IPython import embed
from torch.autograd import Variable
from monolayout_resnet_encoder import ResnetEncoder


import monolayout_model_2 as model

from monolayout_model_2 import Conv3x3
from monolayout_train import data_loader
#from monolayout_train import data_loader
from pretrain_model import Pre_VAE

###

import pandas as pd

import torchvision

from data_helper import UnlabeledDataset, LabeledDataset, HybridDataset, HybridDataset2
from helper import collate_fn, compute_ts_road_map

from process_labels import *







class Pretrainer:
    
    def __init__(self, min_batch=1, epochs=6):
        self.image_folder = "/Users/ianleefmans/Desktop/Pretrain/Project-TBIRD_processed_data/data"
        self.annotation_csv = "/Users/ianleefmans/Desktop/Pretrain/Project-TBIRD_processed_data/data/annotation.csv"
        self.save_path = "/Users/ianleefmans/Desktop/Pretrain/"
        self.model_name = "Pretrain_Models"
        self.log_frequency = 2
        self.res_layers = 18
        self.height = 256
        self.width = 306
        self.min_batch = min_batch
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #self.criterion = nn.functional.binary_cross_entropy(reduction='sum')
        self.learning_rate = 1e-3
        self.epochs = epochs
        
    

        # Data Loader

        #self.train_loader, self.test_loader = data_loader(self.image_folder, self.annotation_csv, self.min_batch, False)
        #############################################################################################
        self.transform = torchvision.transforms.ToTensor()
        self.unlabeled_trainset = UnlabeledDataset(image_folder="/Users/ianleefmans/Desktop/Deep_Learning/Project/data",scene_index=np.arange(0,125), first_dim='image',transform=self.transform)
        self.train_loader = torch.utils.data.DataLoader(self.unlabeled_trainset,batch_size=1, shuffle=False,
                                                   num_workers=2)

        self.unlabeled_testset = UnlabeledDataset(image_folder="/Users/ianleefmans/Desktop/Deep_Learning/Project/data",scene_index=np.arange(125,131), first_dim='image',transform=self.transform)
        self.test_loader = torch.utils.data.DataLoader(self.unlabeled_testset,batch_size=1, shuffle=False,
                                                  num_workers=2)

        
        
        
        
        #############################################################################################
    
        print("There are {:d} training items and {:d} test items\n".format(len(self.train_loader),
                                                                                 len(self.test_loader)))

        # Initialize VAE
        self.model = Pre_VAE(self.res_layers, self.height, self.width, min_batch).to(self.device)

        # Optimimizer

        self.optim = torch.optim.Adam(self.model.parameters(), lr = self.learning_rate)
    
    
    #Loss Function
    def loss_fcn(self, x_hat, x, mu, logvar):
        BCE = nn.functional.binary_cross_entropy(x_hat.view(-1, 3*self.height*self.width), x.view(-1, 3*self.height*self.width),  reduction='sum')
        KLD = 0.5 * torch.sum(logvar.exp() - logvar - 1 + mu.pow(2))
        return BCE+KLD



    # Train

    def train(self):
        codes = dict(mulis=list(), logsig2=list(), y=list())
        for epoch in range(0, self.epochs + 1):
            if epoch > 0:
                self.model.train()
                train_loss = 0
                for x in self.train_loader:
                    #print(x)
                    #print(len(x))
                    x = x[0][0].view(1, 3, 256,306)
                    x = x.to(self.device)
                    # ===================forward=====================
                    x_hat, mu, logvar = self.model(x)
                    loss = self.loss_fcn(x_hat, x, mu, logvar)
                    train_loss += loss.item()
                    # ===================backward====================
                    self.optim.zero_grad()
                    loss.backward()
                    self.optim.step()
                    i=1
                    print(i)
                    i+=1
                # ===================log========================
                print(f'====> Epoch: {epoch} Average loss: {train_loss / len(train_loader.dataset):.4f}')
        
        
        
            means, logvars, labels = list(), list(), list()
            with torch.no_grad():
                self.model.eval()
                test_loss = 0
                for x in self.test_loader:
                
                    #print(x[0][0].shape)
                    #print(len(x))
                    x = x[0][0].view(1, 3, 256,306)
                    
                    x = x.to(self.device)
                    # ===================forward=====================
                    x_hat, mu, logvar = self.model(x)
                    test_loss += self.loss_fcn(x_hat, x, mu, logvar).item()
                    # =====================log=======================
                    means.append(mu.detach())
                    logvars.append(logvar.detach())
                    #labels.append(y.detach())
            # ===================log========================
            codes['mulis'].append(torch.cat(means))
            codes['logsig2'].append(torch.cat(logvars))
            #codes['y'].append(torch.cat(labels))
            test_loss /= len(self.test_loader.dataset)
            print(f'====> Test set loss: {test_loss:.4f}')

            best_test_loss = 99999999999999999999999999999999
            
            if epoch % self.log_frequency == 0:
                if test_loss < best_test_loss:
                    best_test_loss = test_loss
                    pass #save model as best model - should this be different from save model?
                    
                self.save_model()



    def save_model(self):
        save_path = os.path.join(self.save_path, self.model_name, "weights_{}".format(self.epoch))
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    
        for model_name, model in self.models.items():
            model_path = os.path.join(self.save_path, "{}.pth".format(model_name))
            state_dict = model.state_dict()
            if model_name == "encoder":
                state_dict["height"] = self.height
                state_dict["width"] = self.width
            
            torch.save(state_dict, model_path)
        optim_path = os.path.join(save_path, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), optim_path)






if __name__ == "__main__":
    pretrainer = Pretrainer()
    pretrainer.train()
