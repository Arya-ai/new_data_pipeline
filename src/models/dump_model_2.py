from __future__ import absolute_import
from __future__ import print_function

import keras
from keras.models import Model
from keras.layers import Conv2D, Dense, Flatten, Input, MaxPooling2D
from keras.layers.normalization import BatchNormalization

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

model_savePath = '../../models/VGG19_model_2.hdf5'
input_shape = (256,256,4)
classes = 17

img_input = Input(shape=input_shape, name='img_input')

# Block 1
x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
x = BatchNormalization()(x)

# Block 2
x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
x = BatchNormalization()(x)

# Block 3
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
x = BatchNormalization()(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv4')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
x = BatchNormalization()(x)

# Block 4
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
x = BatchNormalization()(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv4')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
x = BatchNormalization()(x)

# Block 5
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
x = BatchNormalization()(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv4')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
x = BatchNormalization()(x)

x = Flatten(name='flatten')(x)
x = Dense(512, activation='relu', name='fc1')(x)
x = Dense(512, activation='relu', name='fc2')(x)
atm_out = Dense(4, activation='sigmoid', name='atm_preds')(x)
common_out = Dense(7, activation='sigmoid', name='common_preds')(x)
rare_out = Dense(6, activation='sigmoid', name='rare_preds')(x)

atm_count = Dense(5, activation='softmax', name='atm_count_preds')(x)
common_count = Dense(8, activation='softmax', name='common_count_preds')(x)
rare_count = Dense(7, activation='softmax', name='rare_count_preds')(x)

output_layers = [atm_out, common_out, rare_out, atm_count, common_count, rare_count]
model = Model(outputs=output_layers, inputs=img_input, name="vgg19_mod")

model.save(model_savePath)

print("Saved the model at : {}".format(model_savePath))
