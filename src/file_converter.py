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
  """ if the directory doesn't exist in the path given, 
  this function creates a directory"""
  print("inside ensure_dir")
  directory = os.path.dirname(file_path)
  print(file_path)
  if not os.path.exists(directory):
      os.makedirs(directory)



def conv_niftti_2_tif(source,dest,axis,startVol,endVolume):
  """
  This function takes in the location of the 3D nifti images and converts each image
  to 2D Tiff images based on the selection of the parameters (axis, startVol, and endVolume)
  source - the path to the location of niftti files
  dest - the path to the location where we want to put the Tiff files
  axis - MRI images are 3-dimensional, choose a value of 0, 1 or 2 to take the 
         slices along those directions
         0 - Sagittal View
         1 - Axial View
         2 - Coronal View
  """
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


def conv_niftti_2_tif_noise_truth(source,zero_noise,dest_train,dest_target,axis,startVol,endVolume):
  """
  Based on the parameters:
  source: location of the 3D nifti images
  dest_truth: location where the ground truth 2D tiff files are stored
  dest_target: location where the training images (2D tiff) are stored
  axis: choice of 0 (Sagittal), 1 (Axial), or 2 (Coronal)
  startVol and endVolume: the ranage of slices to extract in float (0 to 1)

  The function takes in the 3D MRI (nifti format) with some noise and the corresponding 3d MRI (nifti format)
  with zero noise and calculates the noise in each slice by substracting actual
  from zero noise. It then stores the images in dest_truth and dest_target locations
  """
  fnames = glob(os.path.join(source, '*.nii*')) #get the list of niftii files in the source
  fn1 = os.path.join(zero_noise, 't1_icbm_normal_1mm_pn0_rf0.mnc.gz.nii.gz')
  #print(fn1)
  #print(dest_truth)
  #print(dest_noise)
  ensure_dir(dest_target)
  ensure_dir(dest_train)
  
  for fn in fnames:
    fname_base = os.path.basename(fn).split(".")[0]
    
    print(fname_base)
    im_noise = nib.load(fn).get_data()
    im_truth = nib.load(fn1).get_data()
    im_pure_noise = im_noise - im_truth
    
    startIdx = int(startVol * im_noise.shape[axis])
    endIdx = int(endVolume * im_noise.shape[axis]) + 1
    for i in range(startIdx, endIdx):
      if axis == 0:
        imdata_for_target_tiff = Image.fromarray(im_pure_noise[i,:,:])
        imdata_for_train_tiff = Image.fromarray(im_noise[i,:,:])
      elif axis == 1:
        imdata_for_target_tiff = Image.fromarray(im_pure_noise[:,i,:])
        imdata_for_train_tiff = Image.fromarray(im_noise[:,i,:])
      else:
        imdata_for_target_tiff = Image.fromarray(im_pure_noise[:,:,i])
        imdata_for_train_tiff = Image.fromarray(im_noise[:,:,i])
      #pdb.set_trace()
      fname_target = os.path.join(dest_target , f"{fname_base}_{i}_target.tif")
      #print(fname_noise)
      fname_train = os.path.join(dest_train, f"{fname_base}_{i}.tif")
      imdata_for_target_tiff.save(fname_target)
      imdata_for_train_tiff.save(fname_train)


def conv_niftti_2_tif_no_noise_truth(source,zero_noise,dest_train,dest_target,axis,startVol,endVolume):
  """
  Based on the parameters:
  source: location of the 3D nifti images
  dest_truth: location where the ground truth 2D tiff files are stored
  dest_target: location where the training images (2D tiff) are stored
  axis: choice of 0 (Sagittal), 1 (Axial), or 2 (Coronal)
  startVol and endVolume: the ranage of slices to extract in float (0 to 1)

  The function takes in the 3D MRI (nifti format) with some noise and the corresponding 3d MRI (nifti format)
  with zero noise and stores in dest_truth and dest_target locations
  """
  fnames_with_noise = glob(os.path.join(source, '*.nii*')) #get the list of niftii files in the source
  fname_zero_noise = os.path.join(zero_noise, 't1_icbm_normal_1mm_pn0_rf0.mnc.gz.nii.gz')
  
  #print(fn1)
  #print(dest_truth)
  #print(dest_noise)
  ensure_dir(dest_target)
  ensure_dir(dest_train)
  
