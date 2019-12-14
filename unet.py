from keras.layers import Input,  Conv2D, Conv2DTranspose
from keras.layers import MaxPooling2D, Dropout
from keras.models import Model
from keras.layers.merge import concatenate

def unet_model(img_width, img_height, n_ch_exps, kernel_size=(3, 3), k_init='he_normal', pretrained_weights=None,
               dropout=0.1):
    # the n-th deep channel's exponent i.e. 2**n 16,32,64,128,256
    # n_ch_exps = [4, 5, 6, 7, 8, 9]

    # channels_last data format:
    ch_axis = 3
    input_shape = (img_width, img_height, 3)

    inp = Input(shape=input_shape)
    encodeds = []

    # encoder
    enc = inp
    print(n_ch_exps)
    for l_idx, n_ch in enumerate(n_ch_exps):
        enc = Conv2D(filters=2 ** n_ch, kernel_size=kernel_size, activation='relu', padding='same',
                     kernel_initializer=k_init)(enc)
        enc = Dropout(dropout * l_idx, )(enc)
        enc = Conv2D(filters=2 ** n_ch, kernel_size=kernel_size, activation='relu', padding='same',
                     kernel_initializer=k_init)(enc)
        encodeds.append(enc)
        # print(l_idx, enc)
        if n_ch < n_ch_exps[-1]:  # do not run max pooling on the last encoding/downsampling step
            enc = MaxPooling2D(pool_size=(2, 2))(enc)

    # decoder
    dec = enc
    print(n_ch_exps[::-1][1:])
    decoder_n_chs = n_ch_exps[::-1][1:]
    for l_idx, n_ch in enumerate(decoder_n_chs):
        l_idx_rev = len(n_ch_exps) - l_idx - 2  #
        dec = Conv2DTranspose(filters=2 ** n_ch, kernel_size=kernel_size, strides=(2, 2), activation='relu',
                              padding='same', kernel_initializer=k_init)(dec)
        dec = concatenate([dec, encodeds[l_idx_rev]], axis=ch_axis)
        dec = Conv2D(filters=2 ** n_ch, kernel_size=kernel_size, activation='relu', padding='same',
                     kernel_initializer=k_init)(dec)
        dec = Dropout(dropout * l_idx)(dec)
        dec = Conv2D(filters=2 ** n_ch, kernel_size=kernel_size, activation='relu', padding='same',
                     kernel_initializer=k_init)(dec)

    outp = Conv2DTranspose(filters=1, kernel_size=kernel_size, activation='sigmoid', padding='same',
                           kernel_initializer='glorot_normal')(dec)

    model = Model(inputs=[inp], outputs=[outp])

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model