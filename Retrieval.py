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
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("--------------------------------------------------")
print("               Loading DOLG Model")
print("--------------------------------------------------")
    
checkpoint="./lightning_logs/version_1/checkpoints/epoch=44-step=11250.ckpt"
model=DolgNet.load_from_checkpoint(checkpoint, input_dim=Config.input_dim, hidden_dim=Config.hidden_dim, output_dim=Config.output_dim, num_of_classes=Config.num_of_classes).to(device)
model.eval()


ap = argparse.ArgumentParser()
ap.add_argument("-query", required = True,
	help = "Path to query which contains image to be queried")
ap.add_argument("-index", required = True,
	help = "Path to index")
ap.add_argument("-result", required = True,
	help = "Path for output retrieved images")
args = vars(ap.parse_args())


# read in indexed images' feature vectors and corresponding image names
h5f = h5py.File(args["index"],'r')
# feats = h5f['dataset_1'][:]
feats = h5f['dataset_1'][:]
#print(feats)
imgNames = h5f['dataset_2'][:]
#print(imgNames)
h5f.close()
        
print("--------------------------------------------------")
print("               searching starts")
print("--------------------------------------------------")
    
# read and show query image
queryDir = args["query"]
#queryImg = mpimg.imread(queryDir)
#plt.title("Query Image")
#plt.imshow(queryImg)
#plt.show()

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
#print rank_ID
#print rank_score


# number of top retrieved images to show
maxres = 10
imlist = [imgNames[index].decode('utf-8') for i,index in enumerate(rank_ID[0:maxres])]
print("top %d images in order are: " %maxres, imlist)

'''
# show top #maxres retrieved result one by one
for i,im in enumerate(imlist):
    image = mpimg.imread(args["result"]+"/"+str(im, 'utf-8'))
    plt.title("search output %d" %(i+1))
    plt.imshow(image)
    plt.show()
'''
with open ("Retrieval_Results.txt","w") as f:
  for i,im in enumerate(imlist):
    f.write(args["result"]+"/"+str(im)+"\n")
    