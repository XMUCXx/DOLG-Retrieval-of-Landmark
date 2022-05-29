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

print("Loading Model...")
checkpoint="./lightning_logs/version_1/checkpoints/epoch=44-step=11250.ckpt"
model=DolgNet.load_from_checkpoint(checkpoint, input_dim=Config.input_dim, hidden_dim=Config.hidden_dim, output_dim=Config.output_dim, num_of_classes=Config.num_of_classes).to(device)
model.eval()
print("Calculating...")
#print(model)

path=str(sys.argv[1])
try:
  img = jpeg.JPEG(path).decode()
except:
  img = Image.open(path)
  img=img.convert("RGB")
  img=numpy.array(img)  
img = image_transform(image=img)['image'] 

img=img[numpy.newaxis,:]

print("Most Possible label_id:"+str(model.predict(img,device)))
