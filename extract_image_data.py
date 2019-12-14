import os
import numpy as np
from skimage.io import imread
from skimage.transform import resize


class ImageExtractor:

    def __init__(self,img_width, img_height,  img_channels, train_path,
                 test_path):
        self.img_width = img_width
        self.img_height = img_height
        self.img_channels = img_channels
        self.train_path = train_path
        self.test_path = test_path
        self.X_train = None
        self.Y_train = None
        self.X_sub = None
        self.sizes_test = []
        self.test_ids = next(os.walk(self.test_path))[1]
        self.train_ids = next(os.walk(self.train_path))[1]

    def extract_data(self):


        self.X_train = np.zeros((len(self.train_ids), self.img_height, self.img_width, self.img_channels), dtype=np.uint8)
        self.Y_train = np.zeros((len(self.train_ids), self.img_height, self.img_width, 1), dtype=np.bool)

        # Load up the training images and masks
        for n, id_ in enumerate(self.train_ids):
            img = imread(self.train_path + id_ + '/images/' + id_ + '.png')[:, :, :self.img_channels]
            # Make sure the image is 256x256, we can change this value above by the constant
            img = resize(img, (self.img_height, self.img_width), mode='constant', preserve_range=True)
            self.X_train[n] = img
            # Prepare the corresponding maskss
            # shape 256x256x1 (only black and white so one channel)
            mask = np.zeros((self.img_height, self.img_width, 1), dtype=np.bool)
            # For each mask, resize the image to 256x256, then add a color channel resulting in a (256,265,1) matrix.
            # Then, add the resized matrix image to the final combined training mask by taking the maximum value of
            # the two matrices or in other words, where the mask , resulting in the expected combined mask.
            for mask_file in next(os.walk(self.train_path + id_ + '/masks/'))[2]:
                mask_ = imread(self.train_path + id_ + '/masks/' + mask_file)
                mask_ = resize(mask_, (self.img_height, self.img_width), mode='constant', preserve_range=True)
                mask_ = np.expand_dims(mask_, axis=-1)
                mask = np.maximum(mask, mask_)
            self.Y_train[n] = mask

        # Load the test data sets
        # Get and resize test images
        self.X_sub = np.zeros((len(self.test_ids), self.img_height, self.img_width, self.img_channels), dtype=np.uint8)

        print('Getting and resizing test images ... ')
        for n, id_ in enumerate(self.test_ids):
            path = self.test_path + id_
            img = imread(path + '/images/' + id_ + '.png')[:, :, :self.img_channels]
            self.sizes_test.append([img.shape[0], img.shape[1]])
            img = resize(img, (self.img_height, self.img_width), mode='constant', preserve_range=True)
            self.X_sub[n] = img


    def get_train_data(self):
        return self.X_train, self.Y_train

    def get_submition_data(self):
        return self.X_sub

    def get_sizes_test(self):
        return self.sizes_test

    def get_test_ids(self):
        return self.test_ids

    def get_train_ids(self):
        return self.train_ids




