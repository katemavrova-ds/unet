import pickle
from keras.models import load_model
from metrics import bce_dice_loss
from metrics import dice_coef
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from extract_image_data import ImageExtractor
import numpy as np
import random
from sklearn.model_selection import train_test_split


if __name__ == "__main__":

    seed = 42
    random.seed = seed
    np.random.seed(seed=seed)

    IMG_WIDTH = 256
    IMG_HEIGHT = 256
    filepath = "./results/"

    ig = ImageExtractor(img_width=IMG_WIDTH, img_height=IMG_HEIGHT, img_channels=3, train_path='./data/stage1_train/',
                        test_path='./data/stage1_test/')

    ig.extract_data()

    X_train, Y_train = ig.get_train_data()

    X_tr, X_test, y_tr, y_test = train_test_split(
        X_train, Y_train, test_size=0.05, random_state=seed)



    model_path = 'model-dsbowl2018-1.h5'
    model1 = load_model(model_path,
                        custom_objects={'bce_dice_loss': bce_dice_loss, 'dice_coef': dice_coef})



    dice = []
    for img, msk in zip(X_test, y_test):
         img = np.reshape(img, (1, 256, 256, 3))
         msk = np.reshape(msk, (1, 256, 256, 1))

         pred = model1.predict(img, batch_size=None)
         err = model1.evaluate(img, msk, batch_size=None, verbose=0)

         dice.append(err[1])

         # pred = (pred > 0.5) * 1.0
         # plt.figure(figsize=(20, 60))
         # plt.subplot(131)
         # plt.imshow(img[0, :, :, :])
         # plt.subplot(132)
         # plt.imshow(msk[0, :, :, 0])
         # plt.subplot(133)
         # plt.imshow(pred[0, :, :, 0])
         # plt.show()
    print(np.mean(dice))

