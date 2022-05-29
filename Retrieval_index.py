# -*- coding: utf-8 -*-
# Author: yongyuan.name
import os
import h5py
import numpy as np
import argparse

from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning import Trainer

from model.dolg import DolgNet
from config import Config

from dataset.transform import image_transform
from PIL import Image
import numpy
import sys
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ap = argparse.ArgumentParser()
ap.add_argument("-database", required = True,
	help = "Path to database which contains images to be indexed")
ap.add_argument("-index", required = True,
	help = "Name of index file")
args = vars(ap.parse_args())


'''
 Returns a list of filenames for all jpg images in a directory. 
'''
def get_imlist(path):
    return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg')]


'''
 Extract features and index the images
'''
if __name__ == "__main__":
    
    print("--------------------------------------------------")
    print("         Loading DOLG Model")
    print("--------------------------------------------------")
    
    checkpoint="./lightning_logs/version_1/checkpoints/epoch=44-step=11250.ckpt"
    model=DolgNet.load_from_checkpoint(checkpoint, input_dim=Config.input_dim, hidden_dim=Config.hidden_dim, output_dim=Config.output_dim, num_of_classes=Config.num_of_classes).to(device)
    model.eval()
    
    db = args["database"]
    img_list = get_imlist(db)
    
    print("--------------------------------------------------")
    print("         feature extraction starts")
    print("--------------------------------------------------")
    
    feats = []
    names = []

    
    for i, img_path in enumerate(img_list):
        path=str(img_path)
        try:
          img = jpeg.JPEG(path).decode()
        except:
          img = Image.open(path)
          img=img.convert("RGB")
          img=numpy.array(img)  
        img = image_transform(image=img)['image'] 
        img=img[numpy.newaxis,:]
    
        norm_feat = model.extract_feat(img,device)
        img_name = os.path.split(img_path)[1]
        feats.append(norm_feat)
        names.append(img_name)
        print("extracting feature from image No. %d , %d images in total" %((i+1), len(img_list)))

    feats = np.array(feats)
    # print(feats)
    # directory for storing extracted features
    output = args["index"]
    
    print("--------------------------------------------------")
    print("      writing feature extraction results ...")
    print("--------------------------------------------------")


    h5f = h5py.File(output, 'w')
    h5f.create_dataset('dataset_1', data = feats)
    h5f.create_dataset('dataset_2', data = names)
    #h5f.create_dataset('dataset_2', data = np.string_(names))
    h5f.close()
