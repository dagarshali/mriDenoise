import fastai.vision as faiv
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch import nn
import torchvision
import cls_for_reading_tif as clsrt
import create_data_bunch as dataFuncs
import fastai
from fastai.vision import *
from fastai.callbacks import *
from fastai.utils.mem import *
from pathlib import Path
import models as mymodels
from skimage.measure import compare_psnr, compare_ssim

def calculate_psnr_ssim(im_target,im_to_test):
    fn_np = np.asarray(Image.open(im_to_test),dtype=np.float32)
    target_psnr = np.asarray(Image.open(im_target),dtype=np.float32)
    val_range = im_to_test.max() - im_to_test.min()
    psnr = compare_psnr(im_target,im_to_test,data_range=val_range)
    ssim = compare_ssim(im_target,im_to_test,data_range=val_range)
    return psnr,ssim
