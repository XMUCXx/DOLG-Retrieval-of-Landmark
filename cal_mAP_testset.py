# -*- coding: utf-8 -*-
# Author: yongyuan.name
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning import Trainer

from model.dolg import DolgNet
from config import Config

from dataset.transform import image_transform
from PIL import Image
import numpy
import sys

import numpy as np
import h5py

#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
import argparse
import os
import torch


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_imlist(path):
    return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg')]


print("--------------------------------------------------")
print("               Loading DOLG Model")
print("--------------------------------------------------")
    
checkpoint="./lightning_logs/version_1/checkpoints/epoch=44-step=11250.ckpt"
model=DolgNet.load_from_checkpoint(checkpoint, input_dim=Config.input_dim, hidden_dim=Config.hidden_dim, output_dim=Config.output_dim, num_of_classes=Config.num_of_classes).to(device)
model.eval()


ap = argparse.ArgumentParser()

ap.add_argument("-index", required = True,
	help = "Path to index")
ap.add_argument("-testdir", required = True,
	help = "Path to testset")

args = vars(ap.parse_args())

db = args["testdir"]
img_list = get_imlist(db)


# read in indexed images' feature vectors and corresponding image names
h5f = h5py.File(args["index"],'r')
# feats = h5f['dataset_1'][:]
feats = h5f['dataset_1'][:]
#print(feats)
imgNames = h5f['dataset_2'][:]
#print(imgNames)
h5f.close()
        
print("--------------------------------------------------")
print("               test starts")
print("--------------------------------------------------")

print("Testset Size:"+str(len(img_list)))
with open("Testset_mAP_log.txt","w") as f:
  f.write("Testset Size:"+str(len(img_list))+"\n")
mAP=0
mAP10=0
for img_id, img_path in enumerate(img_list):    
  queryDir = img_path
  # extract query image's feature, compute simlarity score and sort
  path=str(queryDir)
  try:
    img = jpeg.JPEG(path).decode()
  except:
    img = Image.open(path)
    img=img.convert("RGB")
    img=numpy.array(img)  
  img = image_transform(image=img)['image'] 
  img=img[numpy.newaxis,:]
  
  queryVec = model.extract_feat(img,device)
  scores = np.dot(queryVec, feats.T)
  rank_ID = np.argsort(scores)[::-1]
  rank_score = scores[rank_ID]
  # number of top retrieved images to show
  maxres = rank_ID.shape[0]
  imlist = [imgNames[index].decode('utf-8') for i,index in enumerate(rank_ID[0:maxres])]
  AP=0
  AP10=0
  rel_rank=0
  for i in range(maxres):
    if imlist[i][0:2]==path[-9:-7]:
      rel_rank+=1
      AP+=rel_rank/(i+1)
    if i==9:
      AP10=AP/rel_rank
  AP/=rel_rank
  
  mAP+=AP
  mAP10+=AP10
  print("Test_id: "+str(img_id)+" Image Path:"+path+" AP:"+str(AP)+" AP@10:"+str(AP10))
  with open("Testset_mAP_log.txt","a") as f:
    f.write("Test_id: "+str(img_id)+" Image Path:"+path+" AP:"+str(AP)+" AP@10:"+str(AP10)+"\n")
mAP/=len(img_list)
mAP10/=len(img_list)
print("Testset mAP: "+str(mAP)+" mAP@10:"+str(mAP10))
with open("Testset_mAP_log.txt","a") as f:
  f.write("Testset mAP: "+str(mAP)+" mAP@10:"+str(mAP10))
