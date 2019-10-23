import cls_for_reading_tif as clsrt
import create_data_bunch as dataFuncs
import fastai
from fastai.vision import *
from fastai.callbacks import *
from fastai.utils.mem import *
from pathlib import Path
import torch
import torchvision
from torch import nn
import models as mymodels
import create_learner as CL
from PIL import Image
import matplotlib.pyplot as plt
import fastai.vision as faiv
import os

def predict(imageFile):
    im1 = clsrt.open_tiff(imageFile)
    im1.show(cmap='gray',figsize=(5,5))
    location_pretrained = "/Users/vishwanathsomashekar/Documents/Insight/mriDenoise/data/processed/train"
    learn = load_learner(location_pretrained)
    res = learn.predict(im1)
    to_save_data = Image.fromarray(im1.data - res[2])
    fname= os.path.join(imageFile.split(".")[0], f"_denoised.tif")
    to_save_data.save(fname)