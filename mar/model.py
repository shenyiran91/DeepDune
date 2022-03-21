import math
from tensorflow import keras
from tensorflow.keras import layers
from transformer import VitImgPatchLayer, VitPosEncodingLayer, SelfAttention

def unet(nb_filter=[16, 32, 64, 128, 256]):
    keras.backend.clear_session()

    # can take arbitrary input size
    image = keras.Input((None, None, 1), name='input')

    conv0 = layers.Conv2D(nb_filter[0], (3, 3), padding='same')(image)
    conv0 = layers.BatchNormalization()(conv0)
    conv0 = layers.Activation('relu')(conv0)
    conv0 = layers.Conv2D(nb_filter[0], (3, 3), padding='same')(conv0)
    conv0 = layers.BatchNormalization()(conv0)
    conv0 = layers.Activation('relu')(conv0)

    pool0 = layers.MaxPooling2D(pool_size=(2, 2))(conv0)

    conv1 = layers.Conv2D(nb_filter[1], (3, 3), padding='same')(pool0)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.Activation('relu')(conv1)
    conv1 = layers.Conv2D(nb_filter[1], (3, 3), padding='same')(conv1)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.Activation('relu')(conv1)

    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = layers.Conv2D(nb_filter[2], (3, 3), padding='same')(pool1)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.Activation('relu')(conv2)
    conv2 = layers.Conv2D(nb_filter[2], (3, 3), padding='same')(conv2)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.Activation('relu')(conv2)

    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = layers.Conv2D(nb_filter[3], (3, 3), padding='same')(pool2)
    conv3 = layers.BatchNormalization()(conv3)
    conv3 = layers.Activation('relu')(conv3)
    conv3 = layers.Conv2D(nb_filter[3], (3, 3), padding='same')(conv3)
    conv3 = layers.BatchNormalization()(conv3)
    conv3 = layers.Activation('relu')(conv3)

    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = layers.Conv2D(nb_filter[4], (3, 3), padding='same')(pool3)
    conv4 = layers.BatchNormalization()(conv4)
    conv4 = layers.Activation('relu')(conv4)
    conv4 = layers.Conv2D(nb_filter[4], (3, 3), padding='same')(conv4)
    conv4 = layers.BatchNormalization()(conv4)
    conv4 = layers.Activation('relu')(conv4)

    up5 = layers.concatenate(
        [layers.UpSampling2D(size=(2, 2))(conv4), conv3], axis=-1)

    conv5 = layers.Conv2D(nb_filter[3], (3, 3), padding='same')(up5)
    conv5 = layers.BatchNormalization()(conv5)
    conv5 = layers.Activation('relu')(conv5)
    conv5 = layers.Conv2D(nb_filter[3], (3, 3), padding='same')(conv5)
    conv5 = layers.BatchNormalization()(conv5)
    conv5 = layers.Activation('relu')(conv5)

    up6 = layers.concatenate(
        [layers.UpSampling2D(size=(2, 2))(conv5), conv2], axis=-1)

    conv6 = layers.Conv2D(nb_filter[2], (3, 3), padding='same')(up6)
    conv6 = layers.BatchNormalization()(conv6)
    conv6 = layers.Activation('relu')(conv6)
    conv6 = layers.Conv2D(nb_filter[2], (3, 3), padding='same')(conv6)
    conv6 = layers.BatchNormalization()(conv6)
    conv6 = layers.Activation('relu')(conv6)

    up7 = layers.concatenate(
        [layers.UpSampling2D(size=(2, 2))(conv6), conv1], axis=-1)

    conv7 = layers.Conv2D(nb_filter[1], (3, 3), padding='same')(up7)
    conv7 = layers.BatchNormalization()(conv7)
    conv7 = layers.Activation('relu')(conv7)
    conv7 = layers.Conv2D(nb_filter[1], (3, 3), padding='same')(conv7)
    conv7 = layers.BatchNormalization()(conv7)
    conv7 = layers.Activation('relu')(conv7)

    up8 = layers.concatenate(
        [layers.UpSampling2D(size=(2, 2))(conv7), conv0], axis=-1)

    conv8 = layers.Conv2D(nb_filter[0], (3, 3), padding='same')(up8)
    conv8 = layers.BatchNormalization()(conv8)
    conv8 = layers.Activation('relu')(conv8)
    conv8 = layers.Conv2D(nb_filter[0], (3, 3), padding='same')(conv8)
    conv8 = layers.BatchNormalization()(conv8)
    conv8 = layers.Activation('relu')(conv8)

    conv9 = layers.Conv2D(1, (1, 1), activation='sigmoid')(conv8)

    model = keras.Model(inputs=[image], outputs=[conv9])

    return model


def unetpp(nb_filter=[16, 32, 64, 128, 256], deep_supervision=False):
    keras.backend.clear_session()

    # can take arbitrary input size
    image = keras.Input((None, None, 1), name='input')

    conv0 = layers.Conv2D(nb_filter[0], (3, 3), padding='same')(image)
    conv0 = layers.BatchNormalization()(conv0)
    conv0 = layers.Activation('relu')(conv0)
    conv0 = layers.Conv2D(nb_filter[0], (3, 3), padding='same')(conv0)
    conv0 = layers.BatchNormalization()(conv0)
    conv0 = layers.Activation('relu')(conv0)

    pool0 = layers.MaxPooling2D(pool_size=(2, 2))(conv0)

    conv1 = layers.Conv2D(nb_filter[1], (3, 3), padding='same')(pool0)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.Activation('relu')(conv1)
    conv1 = layers.Conv2D(nb_filter[1], (3, 3), padding='same')(conv1)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.Activation('relu')(conv1)

    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    up1 = layers.concatenate(
        [layers.UpSampling2D(size=(2, 2))(conv1), conv0], axis=-1)

    conv2 = layers.Conv2D(nb_filter[0], (3, 3), padding='same')(up1)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.Activation('relu')(conv2)
    conv2 = layers.Conv2D(nb_filter[0], (3, 3), padding='same')(conv2)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.Activation('relu')(conv2)

    conv3 = layers.Conv2D(nb_filter[2], (3, 3), padding='same')(pool1)
    conv3 = layers.BatchNormalization()(conv3)
    conv3 = layers.Activation('relu')(conv3)
    conv3 = layers.Conv2D(nb_filter[2], (3, 3), padding='same')(conv3)
    conv3 = layers.BatchNormalization()(conv3)
    conv3 = layers.Activation('relu')(conv3)

    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    up2 = layers.concatenate(
        [layers.UpSampling2D(size=(2, 2))(conv3), conv1], axis=-1)

    conv4 = layers.Conv2D(nb_filter[1], (3, 3), padding='same')(up2)
    conv4 = layers.BatchNormalization()(conv4)
    conv4 = layers.Activation('relu')(conv4)
    conv4 = layers.Conv2D(nb_filter[1], (3, 3), padding='same')(conv4)
    conv4 = layers.BatchNormalization()(conv4)
    conv4 = layers.Activation('relu')(conv4)

    up3 = layers.concatenate([layers.concatenate(
        [layers.UpSampling2D(size=(2, 2))(conv4), conv0], axis=-1), conv2])

    conv5 = layers.Conv2D(nb_filter[0], (3, 3), padding='same')(up3)
    conv5 = layers.BatchNormalization()(conv5)
    conv5 = layers.Activation('relu')(conv5)
    conv5 = layers.Conv2D(nb_filter[0], (3, 3), padding='same')(conv5)
    conv5 = layers.BatchNormalization()(conv5)
    conv5 = layers.Activation('relu')(conv5)

    conv6 = layers.Conv2D(nb_filter[3], (3, 3), padding='same')(pool2)
    conv6 = layers.BatchNormalization()(conv6)
    conv6 = layers.Activation('relu')(conv6)
    conv6 = layers.Conv2D(nb_filter[3], (3, 3), padding='same')(conv6)
    conv6 = layers.BatchNormalization()(conv6)
    conv6 = layers.Activation('relu')(conv6)

    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv6)

    up4 = layers.concatenate(
        [layers.UpSampling2D(size=(2, 2))(conv6), conv3], axis=-1)

    conv7 = layers.Conv2D(nb_filter[2], (3, 3), padding='same')(up4)
    conv7 = layers.BatchNormalization()(conv7)
    conv7 = layers.Activation('relu')(conv7)
    conv7 = layers.Conv2D(nb_filter[2], (3, 3), padding='same')(conv7)
    conv7 = layers.BatchNormalization()(conv7)
    conv7 = layers.Activation('relu')(conv7)

    up5 = layers.concatenate([layers.concatenate(
        [layers.UpSampling2D(size=(2, 2))(conv7), conv1], axis=-1), conv4])

    conv8 = layers.Conv2D(nb_filter[1], (3, 3), padding='same')(up5)
    conv8 = layers.BatchNormalization()(conv8)
    conv8 = layers.Activation('relu')(conv8)
    conv8 = layers.Conv2D(nb_filter[1], (3, 3), padding='same')(conv8)
    conv8 = layers.BatchNormalization()(conv8)
    conv8 = layers.Activation('relu')(conv8)

    up6 = layers.concatenate([layers.concatenate(
        [layers.UpSampling2D(size=(2, 2))(conv8), conv0], axis=-1), conv2, conv5])

    conv9 = layers.Conv2D(nb_filter[0], (3, 3), padding='same')(up6)
    conv9 = layers.BatchNormalization()(conv9)
    conv9 = layers.Activation('relu')(conv9)
    conv9 = layers.Conv2D(nb_filter[0], (3, 3), padding='same')(conv9)
    conv9 = layers.BatchNormalization()(conv9)
    conv9 = layers.Activation('relu')(conv9)

    conv10 = layers.Conv2D(nb_filter[4], (3, 3), padding='same')(pool3)
    conv10 = layers.BatchNormalization()(conv10)
    conv10 = layers.Activation('relu')(conv10)
    conv10 = layers.Conv2D(nb_filter[4], (3, 3), padding='same')(conv10)
    conv10 = layers.BatchNormalization()(conv10)
    conv10 = layers.Activation('relu')(conv10)

    up7 = layers.concatenate(
        [layers.UpSampling2D(size=(2, 2))(conv10), conv6], axis=-1)

    conv11 = layers.Conv2D(nb_filter[3], (3, 3), padding='same')(up7)
    conv11 = layers.BatchNormalization()(conv11)
    conv11 = layers.Activation('relu')(conv11)
    conv11 = layers.Conv2D(nb_filter[3], (3, 3), padding='same')(conv11)
    conv11 = layers.BatchNormalization()(conv11)
    conv11 = layers.Activation('relu')(conv11)

    up8 = layers.concatenate([layers.concatenate(
        [layers.UpSampling2D(size=(2, 2))(conv11), conv3], axis=-1), conv7])

    conv12 = layers.Conv2D(nb_filter[2], (3, 3), padding='same')(up8)
    conv12 = layers.BatchNormalization()(conv12)
    conv12 = layers.Activation('relu')(conv12)
    conv12 = layers.Conv2D(nb_filter[2], (3, 3), padding='same')(conv12)
    conv12 = layers.BatchNormalization()(conv12)
    conv12 = layers.Activation('relu')(conv12)

    up9 = layers.concatenate([layers.concatenate(
        [layers.UpSampling2D(size=(2, 2))(conv12), conv1], axis=-1), conv4, conv8])

    conv13 = layers.Conv2D(nb_filter[1], (3, 3), padding='same')(up9)
    conv13 = layers.BatchNormalization()(conv13)
    conv13 = layers.Activation('relu')(conv13)
    conv13 = layers.Conv2D(nb_filter[1], (3, 3), padding='same')(conv13)
    conv13 = layers.BatchNormalization()(conv13)
    conv13 = layers.Activation('relu')(conv13)

    up10 = layers.concatenate([layers.concatenate([layers.UpSampling2D(
        size=(2, 2))(conv13), conv0], axis=-1), conv2, conv5, conv9])

    conv14 = layers.Conv2D(nb_filter[0], (3, 3), padding='same')(up10)
    conv14 = layers.BatchNormalization()(conv14)
    conv14 = layers.Activation('relu')(conv14)
    conv14 = layers.Conv2D(nb_filter[0], (3, 3), padding='same')(conv14)
    conv14 = layers.BatchNormalization()(conv14)
    conv14 = layers.Activation('relu')(conv14)

    nest_conv1 = layers.Conv2D(1, (1, 1), activation='sigmoid')(conv2)
    nest_conv2 = layers.Conv2D(1, (1, 1), activation='sigmoid')(conv5)
    nest_conv3 = layers.Conv2D(1, (1, 1), activation='sigmoid')(conv9)
    nest_conv4 = layers.Conv2D(1, (1, 1), activation='sigmoid')(conv14)

    if deep_supervision:
        model = keras.Model(inputs=[image], outputs=[nest_conv1,
                                                     nest_conv2,
                                                     nest_conv3,
                                                     nest_conv4])
    else:
        model = keras.Model(inputs=[image], outputs=[nest_conv4])

    return model


def transunet(nb_filter=[16, 32, 64, 128, 256], d_inner_hid=128, n_head=16, n_layers=8, d_model=512, dropout=0.2, patch_size=4):
    keras.backend.clear_session()

    # can take arbitrary input size
    image = keras.Input((256, 256, 1), name='input')

    conv0 = layers.Conv2D(nb_filter[0], (3, 3), padding='same')(image)
    conv0 = layers.BatchNormalization()(conv0)
    conv0 = layers.Activation('relu')(conv0)
    conv0 = layers.Conv2D(nb_filter[0], (3, 3), padding='same')(conv0)
    conv0 = layers.BatchNormalization()(conv0)
    conv0 = layers.Activation('relu')(conv0)

    pool0 = layers.MaxPooling2D(pool_size=(2, 2))(conv0)

    conv1 = layers.Conv2D(nb_filter[1], (3, 3), padding='same')(pool0)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.Activation('relu')(conv1)
    conv1 = layers.Conv2D(nb_filter[1], (3, 3), padding='same')(conv1)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.Activation('relu')(conv1)

    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = layers.Conv2D(nb_filter[2], (3, 3), padding='same')(pool1)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.Activation('relu')(conv2)
    conv2 = layers.Conv2D(nb_filter[2], (3, 3), padding='same')(conv2)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.Activation('relu')(conv2)

    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = layers.Conv2D(nb_filter[3], (3, 3), padding='same')(pool2)
    conv3 = layers.BatchNormalization()(conv3)
    conv3 = layers.Activation('relu')(conv3)
    conv3 = layers.Conv2D(nb_filter[3], (3, 3), padding='same')(conv3)
    conv3 = layers.BatchNormalization()(conv3)
    conv3 = layers.Activation('relu')(conv3)
    drop3 = layers.Dropout(0.5)(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(drop3)

# # add transformer part
    image_size = int(pool3.shape[1])
    channels = int(pool3.shape[3])

    num_patches = (image_size // patch_size) ** 2
    patch_dim = channels * patch_size ** 2

    src_seq = pool3

    src_emb = VitImgPatchLayer(patch_size, patch_dim, d_model)(src_seq)
    src_emb = VitPosEncodingLayer(num_patches, d_model)(src_emb)
    src_emb = layers.Dropout(0.2)(src_emb)

    enc_output = SelfAttention(d_model, d_inner_hid, n_head, n_layers, dropout)(
        src_emb, None, active_layers=999)

    x = layers.Lambda(lambda x: x[:, 1:])(enc_output)

    h = w = int(math.sqrt(num_patches))

    reshaped = layers.Reshape((h, w, d_model))(x)
    for _ in range(patch_size//2):  # back to 8×8
        reshaped = layers.UpSampling2D(size=(2, 2))(reshaped)

    pool3 = reshaped


# back to unet
    conv4 = layers.Conv2D(nb_filter[4], (3, 3), padding='same')(pool3)
    conv4 = layers.BatchNormalization()(conv4)
    conv4 = layers.Activation('relu')(conv4)
    conv4 = layers.Conv2D(nb_filter[4], (3, 3), padding='same')(conv4)
    conv4 = layers.BatchNormalization()(conv4)
    conv4 = layers.Activation('relu')(conv4)
    drop4 = layers.Dropout(0.5)(conv4)

    up5 = layers.concatenate(
        [layers.UpSampling2D(size=(2, 2))(drop4), drop3], axis=-1)

    conv5 = layers.Conv2D(nb_filter[3], (3, 3), padding='same')(up5)
    conv5 = layers.BatchNormalization()(conv5)
    conv5 = layers.Activation('relu')(conv5)
    conv5 = layers.Conv2D(nb_filter[3], (3, 3), padding='same')(conv5)
    conv5 = layers.BatchNormalization()(conv5)
    conv5 = layers.Activation('relu')(conv5)

    up6 = layers.concatenate(
        [layers.UpSampling2D(size=(2, 2))(conv5), conv2], axis=-1)

    conv6 = layers.Conv2D(nb_filter[2], (3, 3), padding='same')(up6)
    conv6 = layers.BatchNormalization()(conv6)
    conv6 = layers.Activation('relu')(conv6)
    conv6 = layers.Conv2D(nb_filter[2], (3, 3), padding='same')(conv6)
    conv6 = layers.BatchNormalization()(conv6)
    conv6 = layers.Activation('relu')(conv6)

    up7 = layers.concatenate(
        [layers.UpSampling2D(size=(2, 2))(conv6), conv1], axis=-1)

    conv7 = layers.Conv2D(nb_filter[1], (3, 3), padding='same')(up7)
    conv7 = layers.BatchNormalization()(conv7)
    conv7 = layers.Activation('relu')(conv7)
    conv7 = layers.Conv2D(nb_filter[1], (3, 3), padding='same')(conv7)
    conv7 = layers.BatchNormalization()(conv7)
    conv7 = layers.Activation('relu')(conv7)

    up8 = layers.concatenate(
        [layers.UpSampling2D(size=(2, 2))(conv7), conv0], axis=-1)

    conv8 = layers.Conv2D(nb_filter[0], (3, 3), padding='same')(up8)
    conv8 = layers.BatchNormalization()(conv8)
    conv8 = layers.Activation('relu')(conv8)
    conv8 = layers.Conv2D(nb_filter[0], (3, 3), padding='same')(conv8)
    conv8 = layers.BatchNormalization()(conv8)
    conv8 = layers.Activation('relu')(conv8)

    conv9 = layers.Conv2D(1, (1, 1), activation='sigmoid')(conv8)

    model = keras.Model(inputs=[image], outputs=[conv9])

    return model


class models:
    Unet = unet
    Unetpp = unetpp
    TransUnet = transunet
