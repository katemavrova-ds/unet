# Semantic segmentation on Data Science Bowl 2018 data 

dataset: https://www.kaggle.com/c/data-science-bowl-2018/overview

EDA and full train and evaluating process are in google colab files(links provided below). 

- https://colab.research.google.com/drive/1pLOoGFnLx2BDbJm3vS60yONjwsRW_1Ei (full train and evaluating process)

- https://colab.research.google.com/drive/1TbZG5uQQsPME1FXbinbcLPNh50e46peM (EDA)

## Structure
That repo contains: 
<ul>
    <li> data folder: folder with images and masks(train and test were inzipped manually)</li> 
  <li>result folder: contains saved model, serialized objects, resulting masks and other</li>
   <li>metrics.py - file, where dice metric and loss defined</li>
  <li>extract_image_data.py - file with tool for reading train and test data(ImageExtractor class)</li>
  <li>unet.py - Unet model</li>
  <li>train.py - training of a model</li>
  <li>evaluation.py - model evaluation</li> 
  <li>predict_mask.py - inference on test data</li>
  <li>requirements.txt</li>
</ul>

Environment: PyCharm 

## Instructions
Dataset should be in './data'(unziped stage1_train and stage1_test), 
  1) use requirements.txt to set up environment
  1) Run train.py to train model; it generates model-dsbowl2018-1.h5 file with model that will be in project folder
  2) run predict_mask.py; output will be in results folder
  
  - evaluation.py and predict_mask.py load trained model from .h5 file 
  - To download dataset in colab with KaggleAPI, use kaggle.json from that repo(upload it from PC in one of cells). 
  


## Solution

To train neural network, images should have one size. 256x256 is minimum size that was found in training set, so all images were resized to 256x256.
 
One image can have a few masks that should be combined. For each mask, resize the image to 256x256, then add a color channel resulting in a (256,265,1) matrix. Then, add the resized matrix image to the final combined training mask by taking the maximum value of the two matrices or in other words, where the mask , resulting in the expected combined mask.


NN is a UNet model with 5 convolutional layers(downsamplimg) and 5 Conv2DTranspose layers(upsampling) with 3x3 filter size(Dropout used with each layer). Number of filters in each layer is 2^n(the n-th deep channel's exponent i.e. 2^n 16,32,64,128,256).   

Data augmentation technique was applied to prevent overfitting of model. Keras provides the ImageDataGenerator class that defines the configuration for image data preparation and augmentation. Horizontal and vertical flip, shifts, rotation, shear angle were used as parameters for data augmentation. 