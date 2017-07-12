from __future__ import absolute_import
from __future__ import print_function

import keras
from keras.models import Model
from keras.layers import Conv2D, Dense, Flatten, Input, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.initializers import he_normal
# from keras.applications import VGG19
from keras import backend as K
# from keras.callbacks import ModelCheckpoint
import cPickle as pickle
import os, sys

# set random seed
seed = 2017

# he_normal = he_normal(seed)
root_dir = '/home/intern/satellite/amazon-satellite-imagery/models/VGG19_5/'
# pretrained_weights = '/home/intern/satellite/amazon-satellite-imagery/models/VGG19_test/weights.01-2.35.hdf5'

if not os.path.exists(root_dir):
    os.makedirs(root_dir)

train_losses_histFile = root_dir + 'train_loss.pkl'
train_accs_histFile = root_dir + 'train_acc.pkl'
val_losses_histFile = root_dir + 'val_loss.pkl'
val_accs_histFile = root_dir + 'val_acc.pkl'

class VGG19_mod():

    def __init__(self):
        self.model = None
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
        x = Dense(classes, activation='sigmoid', name='predictions')(x)

        self.model = Model(outputs=x, inputs=img_input, name="vgg19_mod")

        # self.model.load_weights(pretrained_weights)
        # print("Pretrained weights loaded!")

        opt = keras.optimizers.Adam(lr=1.0)
        self.model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    
        print("Model compiled!")

    def train(self, options):
        self.train_generator = options['train_generator']
        self.validation_generator = options['validation_generator']
        self.nb_train_samples = int(options['nb_train_samples'])
        self.nb_validation_samples = int(options['nb_validation_samples'])
        self.batch_size = int(options['batch_size'])
        self.nb_epochs = int(options['epochs'])

        print("-----------------------------------------------------------")
        print("Changes:\n\tIncrease learning rate of Adam to 1.0")
        print("\tAdd BatchNorm layers.")
        print("\tUse sigmoid instead of softmax.")
        print("-----------------------------------------------------------\n")

        print("Found {} training samples.".format(self.nb_train_samples))
        print("Found {} validation samples.".format(self.nb_validation_samples))
        print("Training for {} epochs".format(self.nb_epochs))

        self.nb_train_batches = self.nb_train_samples // self.batch_size
        self.nb_validation_batches = self.nb_validation_samples // self.batch_size

        print("Train batches: {}".format(self.nb_train_batches))
        print("Validation batches: {}".format(self.nb_validation_batches))

        callbacks = Callbacks()
        #checkpointer = ModelCheckpoint(filepath=weights_path, verbose=1, save_best_only=True, save_weights_only=True, period=10)
        print("Starting to  train...")
        self.model.fit_generator(self.train_generator, steps_per_epoch=self.nb_train_batches,
                                validation_data=self.validation_generator, validation_steps=self.nb_validation_batches,
                                epochs=self.nb_epochs, verbose=2, callbacks=[callbacks])


# Custom Keras Callbacks
class Callbacks(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []
        print('Model structure: \n', self.model.summary())
        return
 
    def on_train_end(self, logs={}):
        try:
            pickle.dump(self.train_losses, open(train_losses_histFile, 'w'))
            pickle.dump(self.train_accs, open(train_accs_histFile, 'w'))
            pickle.dump(self.val_losses, open(val_losses_histFile, 'w'))
            pickle.dump(self.val_accs, open(val_accs_histFile, 'w'))
            print("Dumped all histories for model after training.")
        except Exception as e:
            print("Error dumping histories.")

        try:
            weights_path = root_dir + 'final_weights.hdf5'
            self.model.save_weights(weights_path)
            print("Dumped model weights at path: {}".format(weights_path))
        except Exception as e:
            print("Can't save model weights. Error: ", e)
        return

    def on_epoch_begin(self, epoch, logs={}):
        sys.stdout.write('[')
        sys.stdout.flush()
        return
 
    def on_epoch_end(self, epoch, logs={}):
        sys.stdout.write('>]')
        sys.stdout.flush()

        self.train_losses.append(logs.get('loss'))
        self.train_accs.append(logs.get('acc'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_accs.append(logs.get('val_acc'))
        print("\nAppended metrics to metrics list.")
        
        try:
            pickle.dump(self.train_losses, open(train_losses_histFile, 'w'))
            pickle.dump(self.train_accs, open(train_accs_histFile, 'w'))
            pickle.dump(self.val_losses, open(val_losses_histFile, 'w'))
            pickle.dump(self.val_accs, open(val_accs_histFile, 'w'))
            print("Dumped all histories for model after {} epochs.".format(epoch))
        except Exception as e:
            print("Error dumping histories.")

        if not epoch%4:
            weights_path = root_dir + 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'.format(epoch=epoch, val_loss=logs['val_loss'])
            try:
                self.model.save_weights(weights_path)
                print("Dumped model weights at path: {}".format(weights_path))
            except Exception as e:
                print("Can't save model weights. Error: ", e)

        return
 
    def on_batch_begin(self, batch, logs={}):
        return
 
    def on_batch_end(self, batch, logs={}):
        # print("Metrics: ", logs)

        # output_tensor = self.model.output
        # variable_tensors = self.model.trainable_weights

        # gradients = K.gradients(output_tensor, variable_tensors)
        # with tf.session() as sess:
        #     sess.run(tf.initialize_all_variables)
        #     sess.run(gradients)

        if batch%10 == 0:
            sys.stdout.write('=')
            sys.stdout.flush()
        return
