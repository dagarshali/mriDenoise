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
import predictor


# path_to_train = "/Users/vishwanathsomashekar/Documents/Insight/mriDenoise/data/processed/train"
# path_to_target = "/Users/vishwanathsomashekar/Documents/Insight/mriDenoise/data/processed/target"



# bs = 4
# im_size = 45

# data= dataFuncs.generate_data_bunch(path_to_train,path_to_target,im_size,bs)
# # # print(data)
# learner = CL.createLearner(data,model_choice=1)
# # # print(learn.summary())
# num_epochs = 1
# cont_from_model = "best"
# lr = 1e-3
# CL.train_model(learner,num_epochs,lr,cont_from_model)
# CL.optimal_lrFinder(learner)

imfile = "/Users/vishwanathsomashekar/Documents/Insight/mriDenoise/data/processed/train/t1_icbm_normal_1mm_pn1_rf0_48.tif"
predictor.predict(imfile)