from __future__ import print_function
from __future__ import absolute_import

import keras
from keras.models import Model
from keras.layers import Conv2D, Dense, Flatten, Input, MaxPooling2D
from keras.initializers import he_normal
from keras.applications import VGG19
from keras import backend as K

# set random seed
seed = 2017

# he_normal = he_normal(seed)

class VGG19_mod():

    def __init__(self):
        self.model = None
        self.generator = None
        self.n_samples = None
        self.batch_size = None
        self.nb_epochs = None

    def build(self, options):
        # I tried modifying the already defined VGG19 model (by fchollet)
        # but it is quite difficult to change the graph structure 
        # after you get a Keras model instance
        # So, I will redefine the whole network (a bit modified) again

        # because only one input and one output
        # input_shape = options['input_shapes'][0]
        # classes = options['output_shapes'][0]
        input_shape = (256,256,4)
        classes = 17

        img_input = Input(shape=input_shape, name='img_input')

        # Block 1
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

        # Block 2
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

        # Block 3
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv4')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

        # Block 4
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv4')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

        # Block 5
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv4')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        x = Dense(classes, activation='softmax', name='predictions')(x)

        self.model = Model(outputs=x, inputs=img_input, name="vgg19_mod")

        self.model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['accuracy'])
	
	print("Model compiled!")

    def train(self, options):
        self.generator = options['generator']
        self.n_samples = int(options['n_samples'])
        self.batch_size = int(options['batch_size'])
        self.nb_epochs = int(options['epochs'])

        self.n_batches = self.n_samples // self.batch_size

	print("Starting to  train...")
        self.model.fit_generator(self.generator, steps_per_epoch=self.n_batches, epochs=self.nb_epochs, verbose=2)
