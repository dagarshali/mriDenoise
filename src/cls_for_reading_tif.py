import fastai.vision as faiv
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch import nn
import torchvision
"""
This was created for handling the operations like open the tiff file and
returning that as Fastai Image object
"""

def open_tiff(fn:str) -> faiv.Image:
    """ Return fastai `Image` object created from Tiff image in file `fn`."""
    
    
    return faiv.Image(torch.Tensor(np.asarray(Image.open(fn),dtype=np.float32)[None,...]))
    

    
class TiffItemList(faiv.ImageList):
    """ custom item list for Tiff files """
    def open(self, fn:faiv.PathOrStr)->faiv.Image: return open_tiff(fn)

    
class TiffTiffList(TiffItemList):
    """ item list suitable for synthesis tasks """
    _label_cls = TiffItemList