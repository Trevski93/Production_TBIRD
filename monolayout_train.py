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


import monolayout_model_2 as model

###

import pandas as pd

import torchvision

from data_helper import UnlabeledDataset, LabeledDataset, HybridDataset, HybridDataset2
from helper import collate_fn, compute_ts_road_map

from process_labels import *

def data_loader(image_folder,annotation_csv,batch_size, discr):
    #print("YES INDEED FILE REFERENCING WORKS",type(image_folder),type(annotation_csv))
    #labeled_scene_index = np.arange(106, 134)
    #labeled_scene_index = np.arange(107,108)
    #val_index=np.random.choice(labeled_scene_index, size=7, replace=False)
    
    val_index  = np.arange(124,125)
    #val_index  = np.arange(127,134)#labeled_scene_index #Comment out
    
    #print("Val index scene indexxxxxxxx:", val_index) 
    #train_index=np.setdiff1d(labeled_scene_index,val_index)
    
    train_index  = np.arange(123,124)#labeled_scene_index #comment out
    #train_index  = np.arange(106,127)#labeled_scene_index #comment out
    
    
    
    #print("Train index scene indexxxxxxxx:", train_index)
    transform = torchvision.transforms.ToTensor()
    train_set = HybridDataset2(image_folder=image_folder,
                                  annotation_file=annotation_csv,
                                  scene_index=train_index,
                                  transform=transform,
                                  extra_info=False,
                                  discr_run=discr
                                 )
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=collate_fn)
    #print("TRAINLOADER DIMENSIONSSSSSS:",type(trainloader))
    val_set = HybridDataset2(image_folder=image_folder,
                                  annotation_file=annotation_csv,
                                  scene_index=val_index,
                                  transform=transform,
                                  extra_info=False,
                                  discr_run=False
                                 )
    valloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=collate_fn)
    #print("VALLOADER DIMENSIONSSSSSS:",type(valloader))
    return trainloader, valloader

def get_args():
    parser = argparse.ArgumentParser(description="MonoLayout options")
    parser.add_argument("--save_path", type=str, default="./models/",
                         help="Path to save models")
    parser.add_argument("--model_name", type=str, default="monolayout",
                         help="Model Name with specifications")
    parser.add_argument("--type", type=str, choices=["both", "static", "dynamic"],
                         help="Type of model being trained")
    parser.add_argument("--batch_size", type=int, default=12, #originally was 16 but I changed it to be size of two samples -djk
                         help="Mini-Batch size")
    parser.add_argument("--image_folder", type=str, default='../../djk519/deeplearning/Project-TBIRD/data',
                         help="path for data folder")
    parser.add_argument("--annotation_csv", type=str, default='../../djk519/deeplearning/Project-TBIRD/data/annotation.csv',
                         help="path for csv with target info")
    parser.add_argument("--load_weights_folder", type=str, default=None,
                         help="path to load saved weights")
    parser.add_argument("--lr", type=float, default=1e-5,
                         help="learning rate")
    parser.add_argument("--lr_D", type=float, default=1e-5,
                         help="discriminator learning rate")
    parser.add_argument("--scheduler_step_size", type=int, default=5,
                         help="step size for the both schedulers")
    parser.add_argument("--static_weight", type=float, default=5.,
                         help="static weight for calculating loss")
    parser.add_argument("--dynamic_weight", type=float, default=15.,
                         help="dynamic weight for calculating loss")
    parser.add_argument("--occ_map_size", type=int, default=658, #originally was 128 but I changed it to match our data
                         help="size of topview occupancy map")
    parser.add_argument("--num_epochs", type=int, default=100,
                         help="Max number of training epochs")
    parser.add_argument("--log_frequency", type=int, default=5,
                         help="Log files every x epochs")
    parser.add_argument("--lambda_D", type=float, default=0.01,
                         help="tradeoff weight for discriminator loss")
    parser.add_argument("--discr_train_epoch", type=int, default=5,
                         help="epoch to start training discriminator")
    
    return parser.parse_args()


class Trainer:
    def __init__(self):
        self.opt = get_args()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.models = {}
        self.weight = {}
        self.weight["static"] = self.opt.static_weight
        self.weight["dynamic"] = self.opt.dynamic_weight
        self.criterion_d = nn.BCEWithLogitsLoss()
        self.parameters_to_train = []
        self.parameters_to_train_D = []
        
        ## Data Loaders
        self.discr_run = (self.opt.discr_train_epoch < self.opt.num_epochs)
        self.train_loader, self.val_loader = data_loader(self.opt.image_folder, self.opt.annotation_csv, self.opt.batch_size, self.discr_run)

        print("There are {:d} training items and {:d} validation items\n".format(
            len(self.train_loader), len(self.val_loader)))
        
        self.height = 256
        self.width = 306

        # Initializing models
        self.models["encoder"] = model.Encoder(18, self.height, self.width, pretrained=False) #.to(self.device) #test batch_size 
        if self.opt.type == "both":
            self.models["static_decoder"] = model.Decoder(
                self.models["encoder"].resnet_encoder.num_ch_enc) #.to(self.device)
            
            self.models["static_discr"] = model.Discriminator() #.to(self.device)
            self.models["dynamic_discr"] = model.Discriminator() #.to(self.device)
            
            self.models["dynamic_decoder"] = model.Decoder(
                self.models["encoder"].resnet_encoder.num_ch_enc) #.to(self.device)
        else:
            self.models["decoder"] = model.Decoder(self.models["encoder"].resnet_encoder.num_ch_enc) #.to(self.device)
            self.models["discriminator"] = model.Discriminator() #.to(self.device)
            
        print("************************TESTING**********************************")
        try:
            self.models['encoder'].get_device()
            print("encoder cuda yes")
              
        except AttributeError: 
            print("encoder model is not cuda")
              
        try:
            self.models['static_decoder'].get_device()
            print("static_decoder cuda yes")
              
        except AttributeError: 
            print("static_decoder model is not cuda")
              
        try:
            self.models['static_discr'].get_device()
            print("static_discr cuda yes")
              
        except AttributeError: 
            print("static_discr model is not cuda")
              
        try:
            self.models['dynamic_discr'].get_device()
            print("dynamic_discr cuda yes")
              
        except AttributeError: 
            print("dynamic_discr model is not cuda")  
              
        try:
            self.models['dynamic_decoder'].get_device()
            print("dynamic_decoder cuda yes")
              
        except AttributeError: 
            print("dynamic_decoder model is not cuda") 
              
              
        print("************************TESTING**********************************")
        
        for key in self.models.keys():
            self.models[key].to(self.device) ## NEW POTENTIALLY CRITICAL LINE OF CODE HERE FOR CUDA - TREVOR
            if "discr" in key:
                self.parameters_to_train_D += list(self.models[key].parameters().cuda())
            else:
                self.parameters_to_train += list(self.models[key].parameters().cuda())
                
#         for key in self.models.keys():
#             self.models[key].to(self.device)
#             if "discr" in key:
#                 self.parameters_to_train_D += list(self.models[key].parameters())
#             else:
#                 self.parameters_to_train += list(self.models[key].parameters())

        # Optimization 
        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.lr)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(self.model_optimizer, 
            self.opt.scheduler_step_size, 0.1)

        self.model_optimizer_D = optim.Adam(self.parameters_to_train_D, self.opt.lr_D)
        self.model_lr_scheduler_D = optim.lr_scheduler.StepLR(self.model_optimizer_D, 
            self.opt.scheduler_step_size, 0.1)

        if self.opt.load_weights_folder is not None: #check this code
            self.load_model()

        for key in self.models.keys():
            self.models[key].to(self.device)
        
        self.patch = (1, self.opt.occ_map_size // 2**4 + 1, self.opt.occ_map_size // 2**4 + 1)

        self.valid = Variable(torch.Tensor(np.ones((self.opt.batch_size, *self.patch))),
                                                     requires_grad=False).float().cuda()
        self.fake = Variable(torch.Tensor(np.zeros((self.opt.batch_size, *self.patch))),
                                                     requires_grad=False).float().cuda()

        print("initialization done")

    def train(self):
        best_val=-1.0 #this is for if it's supposed to be high,need to check   
        for self.epoch in range(self.opt.num_epochs):
            with torch.autograd.set_detect_anomaly(True):
                loss = self.run_epoch(best_val)
            print("Epoch: %d | Loss: %.4f | Discriminator Loss: %.4f"%(self.epoch, loss["loss"], 
                loss["loss_discr"]))

            #if self.epoch % self.opt.log_frequency == 0:
                #print("starting val")
                #our_val=self.validation()
                #if our_val > best_val:
                    #best_val=our_val
                    #pass #save model as best model - should this be different from save model?

                #self.save_model()

    def process_batch(self, inputs_samp, validation=False):
        sample, target, roadmap, = inputs_samp[0],inputs_samp[1],inputs_samp[2]
        inputs={}
        outputs = {}
        #print("SAMPLE SHAPEEEEEEEEEEE:", sample[0].size())
        inputs["color"] = torch.stack(sample).to(self.device)
        inputs["static"] = torch.stack(roadmap).to(self.device)
        inputs["dynamic"] = torch.stack(target).to(self.device)
        
        if (self.discr_run and validation==False):
            dynamic_discr, static_discr = inputs_samp[3], inputs_samp[4]
            outputs["static_discr"] = torch.stack(static_discr).to(self.device)
            outputs["dynamic_discr"] = torch.stack(dynamic_discr).to(self.device)

        #losses = {} #added losses - Trevor But it gets added in compute_losses() -Davida
        
        #print("PROCESS BATCH:, STACKED INPUT SIZESSSSSSSSSSSS and TYPESSS")
        #print("image",inputs["color"].size(),inputs["color"].dtype)
        #print("road",inputs["static"].size(),inputs["static"].dtype)
        #print("target",inputs["dynamic"].size(),inputs["dynamic"].dtype)
        #print("road osm",outputs["static_discr"].size(),inputs["static_discr"].dtype)
        #print("target osm",outputs["dynamic_discr"].size(),inputs["dynamic_discr"].dtype)
        
#         for key in self.models.keys():
#             self.models[key].to(self.device)
#             if "discr" in key:
#                 self.parameters_to_train_D += list(self.models[key].parameters())
#             else:
#                 self.parameters_to_train += list(self.models[key].parameters())
        
        
        #for key, inpt in inputs.items():
            #print("TYPE",inpt[0])
        #    inputs[key] = inpt.to(self.device) #removed [0] index
        #for key, inpt in outputs.items():
            #print("TYPE",inpt[0])
        #    outputs[key] = inpt.to(self.device) #removed [0] index
            
        #print("LETS DOUBLE CHECKKKKKKKKK:", inputs["color"].size())

        features = self.models["encoder"](inputs["color"]).cuda() #tensor
  
        #print("features variable in training script:",features.device(1))
        
        if self.opt.type == "both":
            #self.models["dynamic_decoder"].get_device()
            #self.models["dynamic_decoder"] = self.models["dynamic_decoder"].cuda()
            outputs["dynamic"] = self.models["dynamic_decoder"](features).cuda()
            outputs["static"] = self.models["static_decoder"](features).cuda()
            
            #print("PROCESS BATCHHHHHHHHHH, OUTPUTTSSS:", outputs["dynamic"].size(), outputs["static"].size()) 
            if validation:
                return inputs, outputs
            
            losses = self.compute_losses(inputs,outputs)
            losses["loss"] = losses["dynamic_loss"] + losses["static_loss"] 
            #losses["loss_discr"] = torch.zeros(1) 
        
        else:
            outputs["topview"] = self.models["decoder"](features)
            if validation:
                return inputs, outputs
            losses = self.compute_losses(inputs, outputs) #TREVOR added line
            #losses["loss_discr"] = torch.zeros(1) #TREVOR added line 
            #return outputs,losses #TREVOR  added line 

        return outputs,losses
        
        #losses = self.compute_losses(inputs, outputs) - original code
        #losses["loss_discr"] = torch.zeros(1) - orginal code 

        #return outputs, losses - original code


    def run_epoch1(self):
        self.model_optimizer.step()
        # self.model_optimizer_D.step()
        loss = {}
        loss["loss"], loss["loss_discr"] = 0., 0.
        for batch_idx, inputs in tqdm.tqdm(enumerate(self.train_loader)):
            outputs, losses = self.process_batch(inputs)
            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()

            loss["loss"] += losses["loss"].item()
            loss["loss_discr"] += losses["loss_discr"].item()
        loss["loss"] /= len(self.train_loader)
        loss["loss_discr"] /= len(self.train_loader)
        return loss


    def run_epoch(self, best_val):
        self.model_optimizer.step()
        self.model_optimizer_D.step()
        loss = {}
        loss["loss"], loss["loss_discr"] = 0.0, 0.0
        i=0                                                                 #Ian Added
        for batch_idx, ipts in tqdm.tqdm(enumerate(self.train_loader)):
            #print(len(inputs)) #5 elements in tuple? each tuple within of lenth 1 - Trevor
            #print("RUN EPOCH:INPUTSSSSSSSSSSSSSSS",batch_idx)
            #print(inputs[0][0].size())
            #print(inputs[1][0].size())
            #print(inputs[2][0].size())
            #print(inputs[3][0].size())
            #print(inputs[4][0].size())
            outputs, losses = self.process_batch(ipts)
            #print("processing done")
            self.model_optimizer.zero_grad()
            
            # Train Discriminator
            if self.epoch >= self.opt.discr_train_epoch:
                print("Hi Discriminator!!!!!!!!!!")
                if self.opt.type == "both":
                    fake_pred_static = self.models["static_discr"](outputs["static"])
                    fake_pred_dynamic = self.models["dynamic_discr"](outputs["dynamic"])
                    real_pred_static = self.models["static_discr"](outputs["static_discr"])
                    real_pred_dynamic = self.models["dynamic_discr"](outputs["dynamic_discr"])
                    #print("valid size:",self.valid.shape)
                    #print("output size:", fake_pred_static.shape)
                    loss_GAN = self.criterion_d(fake_pred_static, self.valid) + self.criterion_d(fake_pred_dynamic, self.valid)
                  #  loss_D1 = self.criterion_d(fake_pred_static, self.fake)
                    loss_D2 = self.criterion_d(real_pred_static, self.valid)
                  #  loss_D3 = self.criterion_d(fake_pred_dynamic, self.fake)
                  #  loss_D4 = self.criterion_d(real_pred_dynamic, self.valid)
                  #  loss_D = self.criterion_d(fake_pred_static, self.fake)+ self.criterion_d(real_pred_static, self.valid) + self.criterion_d(fake_pred_dynamic, self.fake)+ self.criterion_d(real_pred_dynamic, self.valid)
                    loss_G = self.opt.lambda_D * loss_GAN + losses["loss"]
                else:
                    fake_pred = self.models["discriminator"](outputs["topview"])
                    real_pred = self.models["discriminator"](outputs[self.opt.type+"_discr"])
                    loss_GAN = self.criterion_d(fake_pred, self.valid)
                    loss_D = self.criterion_d(fake_pred, self.fake)+ self.criterion_d(real_pred, self.valid)
                    loss_G = self.opt.lambda_D * loss_GAN + losses["loss"]
                
                #print("STARTING BACKPROPPPPPPP")
                loss_G.backward(retain_graph=True)
                self.model_optimizer.step()
                #print("BACKPROP loss_G DONE")
                self.model_optimizer_D.zero_grad()
                #print("Using D2")
                loss_D2.backward()
               # loss_D.backward()
                self.model_optimizer_D.step()
                #print("BACKPROP	loss_D DONE")
                loss["loss_discr"] += loss_D2.item()
                #print(loss["loss_discr"])
               # loss["loss_discr"] += loss_D.item()
                loss["loss"] += losses["loss"].item()
            else:
                losses["loss"].backward()
                self.model_optimizer.step()
                loss["loss"] += losses["loss"].item()


            if i % self.opt.log_frequency == 0:                     #Ian added
                print("starting val")                               #Ian added
                our_val=self.validation()                           #Ian added
                if our_val > best_val:                              #Ian added
                    best_val=our_val                                #Ian added
                    pass #save model as best model - should this be different from save model?
                
                self.save_model()                                   #Ian added
            i+=1                                                    #Ian added


        loss["loss"] /= len(self.train_loader)
        loss["loss_discr"] /= len(self.train_loader)
        
        return loss


    def validation(self):
        if self.opt.type == "both":
            iou_static, iou_dynamic, mAP_static,mAP_dynamic = np.array([0., 0.]), np.array([0., 0.]),np.array([0., 0.]), np.array([0., 0.])
            threat_static,threat_dynamic = 0,0
            for batch_idx, ipts in tqdm.tqdm(enumerate(self.val_loader)):
                with torch.no_grad():
                    inputs, outputs = self.process_batch(ipts, True)
                pred_static = torch.argmax(outputs["static"].detach(), 1).cuda().numpy()
                pred_dynamic = torch.argmax(outputs["dynamic"].detach(), 1).cuda().numpy()
                true_static = torch.squeeze(inputs["static"],1).detach().cuda().numpy()
                true_dynamic = torch.squeeze(inputs["dynamic"],1).detach().cuda().numpy()
                #print("pred shape",pred_static.shape, "true shape",true_static.shape)
                threat_static+= compute_ts_road_map(pred_static,true_static)
                threat_dynamic+= compute_ts_road_map(pred_dynamic,true_dynamic)
                for bb in range(pred_static.shape[0]):
                  #  iou_static += mean_IU(pred_static[bb], true_static[bb])
                  #  iou_dynamic += mean_IU(pred_dynamic[bb], true_dynamic[bb])
                    mAP_static += mean_precision(pred_static[bb], true_static[bb])
                    mAP_dynamic += mean_precision(pred_dynamic[bb], true_dynamic[bb])
            iou_static /= len(self.val_loader)
            mAP_static /= len(self.val_loader)
            threat_static /= len(self.val_loader)
            iou_dynamic /= len(self.val_loader)
            mAP_dynamic /= len(self.val_loader)
            threat_dynamic /= len(self.val_loader)
          #  print("Epoch: %d | Validation: Static: mIOU: %.8f mAP: %.4f Dynamic: mIOU: %.8f mAP: %.4f"%(self.epoch, iou_static[1], mAP_static[1], iou_dynamic[1], mAP_dynamic[1]))
            print("Epoch: %d | Validation: Static: mTS: %.8f mAP: %.4f Dynamic: mTS: %.8f mAP: %.4f"%(self.epoch, threat_static, mAP_static[1], threat_dynamic,mAP_dynamic[1]))
            return threat_static + threat_dynamic #may want to see about having two returns to save independently
        else:
            iou, mAP = np.array([0., 0.]), np.array([0., 0.])
            threat = 0
            for batch_idx, ipts in tqdm.tqdm(enumerate(self.val_loader)):
                with torch.no_grad():
                    inputs, outputs = self.process_batch(ipts, True)
                pred = torch.argmax(outputs["topview"].detach(), 1).cuda().numpy()
                true = torch.squeeze(inputs[self.opt.type],1).detach().cuda().numpy()
                #print(pred.shape, true.shape)
                threat+=compute_ts_road_map(pred,true)
                for bb in range(pred.shape[0]):
               #     iou += mean_IU(pred[bb], true[bb])
                    mAP += mean_precision(pred[bb], true[bb])
            iou /= len(self.val_loader)
            mAP /= len(self.val_loader)
            threat /= len(self.val_loader)
          #  print("Epoch: %d | Validation: mIOU: %.4f mAP: %.4f"%(self.epoch, iou[1], mAP[1]))
            print("Epoch: %d | Validation: mTS: %.4f mAP: %.4f"%(self.epoch, threat, mAP[1]))
            return threat


    def compute_losses(self, inputs, outputs):
        losses = {}
        if self.opt.type == "both":
            losses["static_loss"] = self.compute_topview_loss(outputs["static"], inputs["static"], 
                self.weight["static"]) #TREVOR - hardcoded type weight 
            
            losses["dynamic_loss"] = self.compute_topview_loss(outputs["dynamic"], inputs["dynamic"], 
                self.weight["dynamic"])  #Trevor - #old was outputs["dynamic_loss"] #hardcoded type weight
            
        else:
            losses["loss"] = self.compute_topview_loss(outputs["topview"], inputs[self.opt.type], 
                self.weight[self.opt.type])

        return losses

    def compute_topview_loss(self, outputs, true_top_view, weight):
            
        generated_top_view = outputs #TREVOR - should be temporary so dimensions match up 
        #print("COMPUTE TOPVIEW LOSS: GENERATED OUTPUTSSSSSS SIZEEEEEE",generated_top_view.size())
        #print("COMPUTE TOPVIEW LOSS: TRUEE OUTPUTSSSS SIZEEEEEE",true_top_view.size())
        #true_top_view = torch.ones(generated_top_view.size()).cuda()
        #loss = self.weighted_binary_cross_entropy(generated_top_view, true_top_view, torch.Tensor([1, 25]))
        #true_top_view = true_top_view - Davida commented out
        #loss = CrossEntropyLoss2d() #TREVOR added , Davida commented out
        
        loss = nn.CrossEntropyLoss(weight = torch.Tensor([1., weight]).cuda())# - TREVOR commented out, Davida uncommented
        
        #GENERATED TOP VIEW: torch.Size([6, 2, 64, 64])
        #TRUE TOP VIEW: torch.Size([800, 800])
       
        
        output = loss(generated_top_view, true_top_view)
        return output.mean()

    def save_model(self):
        save_path = os.path.join(self.opt.save_path, self.opt.model_name, self.opt.type, "weights_{}".format(self.epoch))

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for model_name, model in self.models.items():
            model_path = os.path.join(save_path, "{}.pth".format(model_name))
            state_dict = model.state_dict()
            #if model_name == "encoder":
             #   state_dict["height"] = self.height
              #  state_dict["width"] = self.width

            torch.save(state_dict, model_path)
        optim_path = os.path.join(save_path, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), optim_path)

    def load_model(self):
         """Load model(s) from disk
         """
         self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

         assert os.path.isdir(self.opt.load_weights_folder), \
             "Cannot find folder {}".format(self.opt.load_weights_folder)
         print("loading model from folder {}".format(self.opt.load_weights_folder))

         for key in self.models.keys():
             print("Loading {} weights...".format(n))
             path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(key))
             model_dict = self.models[key].state_dict()
             pretrained_dict = torch.load(path)
             pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
             model_dict.update(pretrained_dict)
             self.models[key].load_state_dict(model_dict)

         # loading adam state
         optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
         if os.path.isfile(optimizer_load_path):
             print("Loading Adam weights")
             optimizer_dict = torch.load(optimizer_load_path)
             self.model_optimizer.load_state_dict(optimizer_dict)
             #self.optimizer_G.load_state_dict(optimizer_dict)
         else:
             print("Cannot find Adam weights so Adam is randomly initialized")




if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()
