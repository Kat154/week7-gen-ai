import torch 
import clip 
from PIL import Image 
import glob 
import numpy as np 

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device = device)
print(f'all models:{clip.available_models()}')
#Function for embedding images 
