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


def createLearner(data,model_choice=2):
    if model_choice == 1:
        model = mymodels.DnCNN()
    else:
        model = mymodels.myDnCNN()
    learn = Learner(data,model,metrics=mse,loss_func=MSELossFlat())

    return learn
def train_model(learn,num_epochs,lr,cont_from_model=None):
    if cont_from_model is not None:
        learn.load(cont_from_model)
        learn.fit_one_cycle(num_epochs,max_lr=lr,wd=1e-3,callbacks = [callbacks.SaveModelCallback(learn, mode='min',every='improvement', monitor='mean_squared_error', name='best')])
    else:
        learn.fit_one_cycle(num_epochs,max_lr=lr,wd=1e-3,callbacks = [callbacks.SaveModelCallback(learn, mode='min',every='improvement', monitor='mean_squared_error', name='best')])
    learn.export()
    return

def optimal_lrFinder(learn):
    learn.lr_find()
    learn.recorder.plot()
    # fig1.savefig("lr_find_result.jpg",dpi=1000,bbox_inches='tight')
    return
    