
"""
Necesssary Files and Libraries
"""

import fastai
from fastai.vision import *
from fastai.callbacks import *
from fastai.utils.mem import *
from torchvision.models import vgg16_bn
from glob import glob
import os
import sys
from PIL import Image
import nibabel as nib
import cls_for_reading_tif as clsRT

def get_y_fn(x):
    """
    this function finds the files for the target images to create the dataset, which
    is then used to create the dataloaders
    """
    parent = 'train' if 'train' in str(x) else 'valid'
    fn = path_to_groundtruth/x.name
    #print(fn)
    #fn = data_dir/'flair'/parent/f'{str(x.stem)[:10]}-FLAIR_reg_zscore.nii.gz'
    return fn

def generate_data_bunch(path_to_train_data,path_to_target_data,rand_split=0.1,im_sz,bs):
    """
    This function creates a databunch that is used by Fastai library to
    iterate through the dataset.
    path_to_train: string with absolute path to the folder that contains
    training images

    path_to_target: string with absolute path tothe folder that contains
    target images

    rand_split: Flaot that varies between 0 to 1.0 to split the dataset
    it training/validation. The default is 0.1

    im_sz: Int that defines the size of the image. I many situations, we
    don't have enough training data to work with as is the case in
    project. In order to overcome this issue, I used progressive reszing,
    wherein I resize the images in train dataset to various sizes. As 
    far as the model is concerned it is seeing new data every time we
    resize

    bs: Int that describes the batch_size. this is function of size of
    images and how much memory we have on the GPU/CPU to work with

    """
    path_train = Path(path_to_train_data)
    path_target = Path(path_to_target_data)
    src = clsRT.TiffTiffList.from_folder(path_train).split_by_rand_pct(rand_split,seed=42)
    tfms = tfms = get_transforms(max_rotate=20, max_zoom=1.3, max_lighting=None, max_warp=0.4,
                    p_affine=1., p_lighting=1.)
    data = (src.label_from_func(get_y_fn)
    .transform(new_tfms,size=im_sz, resize_method=ResizeMethod.SQUISH,tfm_y=True)
    .databunch(bs=bs)).normalize(do_y=True)
    data.c = 1
    return data
    
    

