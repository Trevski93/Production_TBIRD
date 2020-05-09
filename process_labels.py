import numpy as np
import cv2
import torch
from scipy import ndimage


# Note: may need to adjust datatypes/use of np vs torch throughout
# This code is likely NOT optimized

def get_y(pointa,pointb,our_point):
    slope=(pointa[1]-pointb[1])/(pointa[0]-pointb[0])
    return((our_point-pointa[0])*slope+pointa[1])

def get_x(pointa,pointb,our_point):
    slope=(pointa[1]-pointb[1])/(pointa[0]-pointb[0])
    return((our_point-pointa[1])/slope+pointa[0])
    
def bounding_box_to_image(bounding_box_list):
    fake_road=torch.zeros(800,800)
    for i, bb in enumerate(bounding_box_list):
        point_squence = torch.stack([bb[:, 0], bb[:, 1], bb[:, 3], bb[:, 2]])
        point_squence_adj=torch.ones(point_squence.shape)
        point_squence_adj.T[0] = point_squence.T[0] * 10 + 400
        point_squence_adj.T[1] = -point_squence.T[1] * 10 + 400
        for j in range(int(np.floor(min(point_squence_adj.T[0]))),int(np.ceil(max(point_squence_adj.T[0]))+1)):
            for k in range(int(np.floor(min(point_squence_adj.T[1]))),int(np.ceil(max(point_squence_adj.T[1]))+1)):
                y1=get_y(point_squence_adj[0],point_squence_adj[1],j)
                y2=get_y(point_squence_adj[2],point_squence_adj[3],j)
                x1=get_x(point_squence_adj[0],point_squence_adj[3],k)
                x2=get_x(point_squence_adj[1],point_squence_adj[2],k)
                if ((min(y1,y2)<=k<=max(y1,y2)) and (min(x1,x2)<=j<=max(x1,x2))):
                    fake_road[k,j]=1
    return fake_road
    
def image_to_bounding_box(binary_image):
    contours,_ =cv2.findContours(    cv2.convertScaleAbs(binary_image.numpy())  , cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    boxes=np.zeros((len(contours),2,4))
    for i in range(len(contours)):
        boxes[i,:,:]=(cv2.boxPoints(cv2.minAreaRect(contours[i])).transpose())
        boxes[i,0]=(boxes[i,0]-400)/10
        boxes[i,1]=-(boxes[i,1]-400)/10
        temp=boxes[i,:,0]
        boxes[i,:,0]=boxes[i,:,2]
        boxes[i,:,2]=boxes[i,:,1]
        boxes[i,:,1]=boxes[i,:,3]
        boxes[i,:,3]=temp
    return torch.from_numpy(boxes)


def split_into_six(original_image): #could use a lot of simplifictation but should work
    #split image
    maps=[torch.zeros((1132,1132)) for _ in range(6)]
    for i in range(400,800):
        for j in range(800):
            if i==400:
                if j<400:
                    maps[0][j+166,i+166]=original_image[j,i]
                elif j>400:
                    maps[3][965-j,965-i]=original_image[799-j,799-i]
            else:
                angle=np.arctan((j-400)/(i-400))
                if -np.radians(30) <= angle <= np.radians(30):
                    maps[1][j+166,i+166]=original_image[j,i]
                    maps[4][965-j,965-i]=original_image[799-j,799-i]
                elif angle > np.radians(30):
                    maps[2][j+166,i+166]=original_image[j,i]
                    maps[3][965-j,965-i]=original_image[799-j,799-i]
                elif angle < -np.radians(30):
                    maps[0][j+166,i+166]=original_image[j,i]
                    maps[5][965-j,965-i]=original_image[799-j,799-i]
    
    #rotate image
    maps_rot = [torch.zeros((1132,1132)) for i in range(len(maps))]
    for n in range(3):
        alpha = 30.0 + 60.0 * n # in degrees
        sin=np.sin(np.radians(alpha))
        cos=np.cos(np.radians(alpha))
        for i in range(566,966):
            for j in range(166,966):
                maps_rot[n][int(-sin*(i-566)+cos*(j-566)+566),int(cos*(i-566)+sin*(j-566)+566)]=maps[n][j,i]
                maps_rot[n+3][int(-sin*(i-566)+cos*(j-566)+566),int(-cos*(i-566)-sin*(j-566)-566)]=maps[n+3][j,1131-i]
        maps_rot[n]=torch.from_numpy(ndimage.binary_fill_holes(maps_rot[n]).astype(int))
        maps_rot[n+3]=torch.from_numpy(ndimage.binary_fill_holes(maps_rot[n+3]).astype(int))
    
    # trim for output
    output_image = [torch.zeros((556,556)) for _ in range(6)]
    for n in range(6):
        output_image[n]=maps_rot[n][:658,237:895]
    
    return output_image


def split_with_index(original_image,index):
    
    maps_rot = torch.zeros((1132,1132))
    
    if index ==0:
        alpha = 30.0 + 60.0 * index # in degrees
        sin=np.sin(np.radians(alpha))
        cos=np.cos(np.radians(alpha))
        for j in range(166,566):
            maps_rot[int(-sin*(0)+cos*(j-566)+566),int(cos*(0)+sin*(j-566)+566)]=original_image[j-166,400]
            for i in range(567,966):
                angle=np.arctan((j-566)/(i-566))
                if angle < -np.radians(30):
                    maps_rot[int(-sin*(i-566)+cos*(j-566)+566),int(cos*(i-566)+sin*(j-566)+566)]=original_image[j-166,i-166]
    
    elif index ==1:
        alpha = 30.0 + 60.0 * index # in degrees
        sin=np.sin(np.radians(alpha))
        cos=np.cos(np.radians(alpha))
        for j in range(331,801):
            for i in range(567,966):
                angle=np.arctan((j-566)/(i-566))
                if -np.radians(30) <= angle <= np.radians(30):
                    maps_rot[int(-sin*(i-566)+cos*(j-566)+566),int(cos*(i-566)+sin*(j-566)+566)]=original_image[j-166,i-166]
        
    elif index ==2:
        alpha = 30.0 + 60.0 * index # in degrees
        sin=np.sin(np.radians(alpha))
        cos=np.cos(np.radians(alpha))
        for j in range(566,966):
            for i in range(567,966):
                angle=np.arctan((j-566)/(i-566))
                if angle > np.radians(30):
                    maps_rot[int(-sin*(i-566)+cos*(j-566)+566),int(cos*(i-566)+sin*(j-566)+566)]=original_image[j-166,i-166]
    
    elif index==3:
        alpha = 30.0 + 60.0 * (index-3) # in degrees
        sin=np.sin(np.radians(alpha))
        cos=np.cos(np.radians(alpha))
        for j in range(566,966):
            maps_rot[int(-sin*(0)+cos*(565-j)+566),int(-cos*(0)-sin*(565-j)-566)]=original_image[965-j,400]
            for i in range(567,966):
                angle=np.arctan((j-566)/(i-566))
                if angle > np.radians(30):
                    maps_rot[int(-sin*(i-566)+cos*(565-j)+566),int(-cos*(i-566)-sin*(565-j)-566)]=original_image[965-j,965-i]
                    
    elif index==4:
        alpha = 30.0 + 60.0 * (index-3) # in degrees
        sin=np.sin(np.radians(alpha))
        cos=np.cos(np.radians(alpha))
        for j in range(331,801):
            for i in range(567,966):
                angle=np.arctan((j-566)/(i-566))
                if -np.radians(30) <= angle <= np.radians(30):
                    maps_rot[int(-sin*(i-566)+cos*(565-j)+566),int(-cos*(i-566)-sin*(565-j)-566)]=original_image[965-j,965-i]
    
    elif index==5:
        alpha = 30.0 + 60.0 * (index-3) # in degrees
        sin=np.sin(np.radians(alpha))
        cos=np.cos(np.radians(alpha))
        for j in range(166,566):
            for i in range(567,966):
                angle=np.arctan((j-566)/(i-566))
                if angle < -np.radians(30):
                    maps_rot[int(-sin*(i-566)+cos*(565-j)+566),int(-cos*(i-566)-sin*(565-j)-566)]=original_image[965-j,965-i]    
    
    # trim for output
    maps_rot=torch.from_numpy(ndimage.binary_fill_holes(maps_rot).astype(int))
    output_image=maps_rot[:658,237:895]
    
    return output_image


def stitch(rotated_outputs):
    output_pad=[torch.zeros(rotated_outputs[0].shape[0],1132,1132) for _ in range(6)]
    for n in range(6):
        output_pad[n][:,:658,237:895]=rotated_outputs[n]
    out_rot = [torch.zeros(rotated_outputs[0].shape[0],1132,1132) for i in range(6)]
    out_ret = torch.zeros(rotated_outputs[0].shape[0],800,800)
    for n in range(3):
        alpha = 30.0 + 60.0 * n # in degrees
        sin=np.sin(np.radians(alpha))
        cos=np.cos(np.radians(alpha))
        for i in range(566,966):
            for j in range(166,966):
                out_rot[n][:,j,i]=output_pad[n][:,int(-sin*(i-566)+cos*(j-566)+566),int(cos*(i-566)+sin*(j-566)+566)]
                out_rot[n+3][:,j,1132-i]=output_pad[n+3][:,int(-sin*(i-566)+cos*(j-566)+566),int(-cos*(i-566)-sin*(j-566)-566)]
        out_ret+=out_rot[n][:,166:966,166:966]
        out_ret+=out_rot[n+3][:,166:966,166:966]
    
    for btc in range(rotated_outputs[0].shape[0]):
        out_ret[btc]=torch.from_numpy(ndimage.binary_fill_holes(out_ret[btc]))
    return out_ret
