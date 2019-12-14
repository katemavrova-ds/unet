from skimage.morphology import label
import numpy as np
from keras.models import load_model
from metrics import bce_dice_loss
from metrics import dice_coef
import pickle
import skimage.transform
import pandas as pd
import datetime
from skimage.io import imsave
import warnings
warnings.filterwarnings("ignore")
from extract_image_data import ImageExtractor
import random
#encoding submition

def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def prob_to_rles(x, cutoff=0.5):
    lab_img = label(x > cutoff)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)



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

    X_sub = ig.get_submition_data()

    sizes_test = ig.get_sizes_test()
    test_ids = ig.get_test_ids()

    model_path = 'model-dsbowl2018-1.h5'

    model = load_model(model_path,
                        custom_objects={'bce_dice_loss': bce_dice_loss, 'dice_coef': dice_coef})

    Y_hat = model.predict(X_sub, verbose=1)

    upsampled_images = []

    for i, shape in enumerate(sizes_test):
        img_ = skimage.transform.resize(Y_hat[i], (shape[0], shape[1]), mode='constant', preserve_range=True)
        upsampled_images.append(img_)

    new_test_ids = []
    rles = []

    for n, id_ in enumerate(test_ids):

        rle = list(prob_to_rles(upsampled_images[n]))
        rles.extend(rle)
        new_test_ids.extend([id_] * len(rle))
        imsave(filepath+"sub_img/"+id_+".png", upsampled_images[n])


    # Create submission DataFrame
    sub = pd.DataFrame()
    sub['ImageId'] = new_test_ids
    sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    print('Submission output to: sub-{}.csv'.format(timestamp))
    sub.to_csv(filepath+"sub-{}.csv".format(timestamp), index=False)



