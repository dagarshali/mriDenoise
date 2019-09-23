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


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)



def conv_niftti_2_tif(source,dest,axis,startVol,endVolume):
  source = source
  startVolume = startVol
  endVolume = endVolume
  axis = axis #options are 0, 1 or 2
  fnames = glob(os.path.join(source, '*.nii*')) #get the list of niftii files in the source
  
  ensure_dir(dest)
  
  for fn in fnames:
    im = nib.load(fn).get_data()
    startIdx = int(startVolume * im.shape[axis])
    endIdx = int(endVolume * im.shape[axis]) + 1
    for i in range(startIdx, endIdx):
      if axis == 0:
        imdata_for_tiff = Image.fromarray(im[i,:,:])
      elif axis == 1:
        imdata_for_tiff = Image.fromarray(im[:,i,:])
      else:
        imdata_for_tiff = Image.fromarray(im[:,:,i])
      fname = os.path.join(dest , f"_{i}.tif")
      print(fname)
      imdata_for_tiff.save(fname)
