# Idea
All images that are obtained using **M**agnetic **R**esonance **I**maging have noise in them. A typical way to deal with the noise is to find the parameters that govern the noise for each MRI image and then apply a filter to denoise the image. This package will take in a brain MRI image which has varying degrees of noise and will denoise it.

You can find the presentation about this project [here](https://docs.google.com/presentation/d/13zEXSVC88StifwMSQzszNhbBF1AJsuvRK_QVdIYP24k/edit?usp=sharing)

# Datasets
The datasets used to train and test the models were obtained from [Brainweb](https://brainweb.bic.mni.mcgill.ca)

# With this package

 - You can create CNN based models with varying number of layers in the model
 - You can train the network with your MR images
 - You can use it as to make an inference on your MR image/s

# Required packages
pip install -r requirements

# How it works

 - Input an 3D MR image in Niftii format and convert it to 2D Slices in tif format using the script convert_to_2D_tiff_images.py 

# Tutorial
 - I have included a tutorial on jupyter notebook called mriDenoise_tutorial to get you started
