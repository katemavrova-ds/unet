from extract_image_data import ImageExtractor
from unet import unet_model
from metrics import bce_dice_loss
from metrics import dice_coef
import random
import numpy as np
import pickle

from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint


# Runtime data augmentation
def get_train_test_augmented(X_data, Y_data, validation_split=0.25, batch_size=32, seed=42):
    X_train, X_test, Y_train, Y_test = train_test_split(X_data,
                                                        Y_data,
                                                        train_size=1 - validation_split,
                                                        test_size=validation_split,
                                                        random_state=seed)
    # Image data generator distortion options
    data_gen_args = dict(rotation_range=45.,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         shear_range=0.2,
                         zoom_range=0.2,
                         horizontal_flip=True,
                         vertical_flip=True,
                         fill_mode='reflect')

    # Train data, provide the same seed and keyword arguments to the fit and flow methods
    X_datagen = ImageDataGenerator(**data_gen_args)
    Y_datagen = ImageDataGenerator(**data_gen_args)
    X_datagen.fit(X_train, augment=True, seed=seed)
    Y_datagen.fit(Y_train, augment=True, seed=seed)
    X_train_augmented = X_datagen.flow(X_train, batch_size=batch_size, shuffle=True, seed=seed)
    Y_train_augmented = Y_datagen.flow(Y_train, batch_size=batch_size, shuffle=True, seed=seed)

    # Test data, no data augmentation, but we create a generator anyway
    X_datagen_val = ImageDataGenerator()
    Y_datagen_val = ImageDataGenerator()
    X_datagen_val.fit(X_test, augment=True, seed=seed)
    Y_datagen_val.fit(Y_test, augment=True, seed=seed)
    X_test_augmented = X_datagen_val.flow(X_test, batch_size=batch_size, shuffle=True, seed=seed)
    Y_test_augmented = Y_datagen_val.flow(Y_test, batch_size=batch_size, shuffle=True, seed=seed)

    # combine generators into one which yields image and masks
    train_generator = zip(X_train_augmented, Y_train_augmented)
    test_generator = zip(X_test_augmented, Y_test_augmented)

    return train_generator, test_generator




if __name__ == "__main__":

  seed = 42
  random.seed = seed
  np.random.seed(seed=seed)

  IMG_WIDTH = 256
  IMG_HEIGHT = 256
  filepath = "./results/"

  ig = ImageExtractor(img_width=IMG_WIDTH, img_height=IMG_HEIGHT,  img_channels=3, train_path='./data/stage1_train/',
                  test_path='./data/stage1_test/')


  ig.extract_data()

  X_train, Y_train = ig.get_train_data()

  with open(filepath+'ig.pickle', 'wb') as f:
       pickle.dump(ig, f)

  model = unet_model(img_width=IMG_WIDTH, img_height=IMG_HEIGHT, n_ch_exps = [4, 5, 6, 7, 8, 9])
  model.summary()

  model.compile(optimizer='adam', loss=bce_dice_loss, metrics=[dice_coef])

  X_tr, X_test, y_tr, y_test = train_test_split(
      X_train, Y_train, test_size=0.05, random_state=seed)

  earlystopper = EarlyStopping(patience=5, verbose=1)
  checkpointer = ModelCheckpoint('model-dsbowl2018-1.h5', verbose=1, save_best_only=True)

  batch_size = 16

  train_generator, test_generator = get_train_test_augmented(X_data=X_tr, Y_data=y_tr, validation_split=0.15,
                                                             batch_size=batch_size, seed=seed
                                                            )
  model.fit_generator(train_generator, validation_data=test_generator, validation_steps=batch_size / 2,
                      steps_per_epoch=len(X_train) / (batch_size * 2), epochs=30,
                      callbacks=[earlystopper, checkpointer])

  results = model.evaluate(X_test, y_test, batch_size=batch_size)

  print("Dice coef:" + str(results[1]))

  model.save(filepath+'model-dsbowl2018-1.h5')

