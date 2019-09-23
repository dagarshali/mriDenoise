# -*- coding: utf-8 -*-
"""
fucke this 
"""
import file_converter



source = "/Users/vishwanathsomashekar/Documents/Insight/mriDenoise/data/raw" #Enter the source directory here
dest = "/Users/vishwanathsomashekar/Documents/Insight/mriDenoise/data/processed/" # Enter the destination directory here
axis = 0 # the direction along which we want to make the slices
startVol = 0.2 # all mri scans of the brain might also have some part of the neck. this start and end vol will make sure we have mostly the brain
endVol = 0.3
file_converter.conv_niftti_2_tif(source,dest,axis,startVol,endVol)



