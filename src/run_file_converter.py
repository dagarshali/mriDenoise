# -*- coding: utf-8 -*-
"""
You need to provide :
- source: the path to the location of niftii files
- dest : the path to the location where you want th tiff files to be saved
- axis : options are 0, 1 or 2. The axis along which to take the slices
- startVol: where you want to start taking the slice
- endVol: where you want to end taking the slice4


"""
import file_converter


zero_noise = "/Users/vishwanathsomashekar/Documents/Insight/mriDenoise/data/raw"
source = "/Users/vishwanathsomashekar/Documents/Insight/mriDenoise/data/preprocessed" #Enter the source directory here
dest_target = "/Users/vishwanathsomashekar/Documents/Insight/mriDenoise/data/processed/newtarget" # Enter the destination directory here
dest_train = "/Users/vishwanathsomashekar/Documents/Insight/mriDenoise/data/processed/newtrain"
axis = 1 # the direction along which we want to make the slices
startVol = 0.2 # all mri scans of the brain might also have some part of the neck. this start and end vol will make sure we have mostly the brain
endVolume = 0.8

# Call the file converter to convert the niftii images to tiff
# file_converter.conv_niftti_2_tif(source,dest,axis,startVol,endVol)
file_converter.conv_niftti_2_tif_no_noise_truth(source,zero_noise,dest_train,dest_target,axis,startVol,endVolume)



