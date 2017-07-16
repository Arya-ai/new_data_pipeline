from __future__ import absolute_import
from __future__ import print_function

import keras
from keras.models import Model
from keras.layers import Conv2D, Dense, Flatten, Input, MaxPooling2D
from keras.layers.normalization import BatchNormalization
# from keras.initializers import he_normal
from keras.regularizers import l2
from keras import backend as K
import cPickle as pickle
import os, sys
import numpy as np

# set random seed
seed = 2017

# supress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

root_dir = os.path.abspath('models/VGG19_12/')
print("Root Directory:", root_dir)
# pretrained_weights = 'models/VGG19_10/weights.04-9.90.hdf5'

if not os.path.exists(root_dir):
    os.makedirs(root_dir)

train_losses_histFile = root_dir + 'train_loss.pkl'
out1_losses_histfile = root_dir + 'out1_loss.pkl'
out2_losses_histfile = root_dir + 'out2_loss.pkl'
out3_losses_histfile = root_dir + 'out3_loss.pkl'
out4_losses_histfile = root_dir + 'out4_loss.pkl'
out5_losses_histfile = root_dir + 'out5_loss.pkl'
out6_losses_histfile = root_dir + 'out6_loss.pkl'

out1_accs_histfile = root_dir + 'out1_acc.pkl'
out2_accs_histfile = root_dir + 'out2_acc.pkl'
out3_accs_histfile = root_dir + 'out3_acc.pkl'
out4_accs_histfile = root_dir + 'out4_acc.pkl'
out5_accs_histfile = root_dir + 'out5_acc.pkl'
out6_accs_histfile = root_dir + 'out6_acc.pkl'

val_losses_histFile = root_dir + 'val_loss.pkl'

val_out1_losses_histfile = root_dir + 'val_out1_loss.pkl'
val_out2_losses_histfile = root_dir + 'val_out2_loss.pkl'
val_out3_losses_histfile = root_dir + 'val_out3_loss.pkl'
val_out4_losses_histfile = root_dir + 'val_out4_loss.pkl'
val_out5_losses_histfile = root_dir + 'val_out5_loss.pkl'
val_out6_losses_histfile = root_dir + 'val_out6_loss.pkl'

val_out1_accs_histfile = root_dir + 'val_out1_acc.pkl'
val_out2_accs_histfile = root_dir + 'val_out2_acc.pkl'
val_out3_accs_histfile = root_dir + 'val_out3_acc.pkl'
val_out4_accs_histfile = root_dir + 'val_out4_acc.pkl'
val_out5_accs_histfile = root_dir + 'val_out5_acc.pkl'
val_out6_accs_histfile = root_dir + 'val_out6_acc.pkl'


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
        if 'nb_samples' in options:
            self.nb_samples = float(options['nb_samples'])
        else:
            raise("Can't find number of samples for assigning class weights")
            sys.exit(-1)

        output_dist = None
        if 'data_dists' in options:
            output_dist = options['data_dists'][0]
        else:
            print("Can't find output dist., so no sample weights assigned.")

        if output_dist is not None:
            # atmospheric data distribution
            atm_dist = np.take(output_dist, [1,4,7,9])
            print("Atmospheric Distribution:", atm_dist)
            # the stupidest way to assign weights, never ever do it again
            # self.atm_weights = 1/np.log10(atm_dist)
            ref = self.nb_samples*np.ones(atm_dist.shape)
            self.atm_weights = ref/atm_dist
            self.atm_weights[self.atm_weights == np.inf] = 0
            print("Atmospheric weights:", self.atm_weights)

            # common labels data distribution
            common_dist = np.take(output_dist, [3,6,8,11,12,14,15])
            print("Common labels Distribution:", common_dist)
            # self.common_weights = 1/np.log10(common_dist)
            ref = self.nb_samples*np.ones(common_dist.shape)
            self.common_weights = ref/common_dist
            self.common_weights[self.common_weights == np.inf] = 0
            print("Common weights:", self.common_weights)

            # rare labels data distribution
            rare_dist = np.take(output_dist, [0,2,5,10,13,16])
            print("Rare labels Distribution:", rare_dist)
            # self.rare_weights = 1/np.log10(rare_dist)
            ref = self.nb_samples*np.ones(rare_dist.shape)
            self.rare_weights = ref/rare_dist
            self.rare_weights[self.rare_weights == np.inf] = 0
            print("Rare weights:", self.rare_weights)

        img_input = Input(shape=input_shape, name='img_input')

        # Block 1
        x = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01),
                    bias_regularizer=l2(0.01), name='block1_conv1')(img_input)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01),
                    bias_regularizer=l2(0.01), name='block1_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
        x = BatchNormalization()(x)

        # Block 2
        x = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01),
                    bias_regularizer=l2(0.01), name='block2_conv1')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01),
                    bias_regularizer=l2(0.01), name='block2_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
        x = BatchNormalization()(x)

        # Block 3
        x = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01),
                    bias_regularizer=l2(0.01), name='block3_conv1')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01),
                    bias_regularizer=l2(0.01), name='block3_conv2')(x)
        x = BatchNormalization()(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01),
                    bias_regularizer=l2(0.01), name='block3_conv3')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01),
                    bias_regularizer=l2(0.01), name='block3_conv4')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
        x = BatchNormalization()(x)

        # Block 4
        x = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01),
                    bias_regularizer=l2(0.01), name='block4_conv1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01),
                    bias_regularizer=l2(0.01), name='block4_conv2')(x)
        x = BatchNormalization()(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01),
                    bias_regularizer=l2(0.01), name='block4_conv3')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01),
                    bias_regularizer=l2(0.01), name='block4_conv4')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
        x = BatchNormalization()(x)

        # Block 5
        x = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01),
                    bias_regularizer=l2(0.01), name='block5_conv1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01),
                    bias_regularizer=l2(0.01), name='block5_conv2')(x)
        x = BatchNormalization()(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01),
                    bias_regularizer=l2(0.01), name='block5_conv3')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01),
                    bias_regularizer=l2(0.01), name='block5_conv4')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
        x = BatchNormalization()(x)

        x = Flatten(name='flatten')(x)
        x = Dense(512, activation='tanh', name='fc1')(x)
        x = Dense(512, activation='tanh', name='fc2')(x)

        inter_atm = Dense(128, activation='tanh', kernel_regularizer=l2(0.01),
                        bias_regularizer=l2(0.01), name='inter_atm')(x)
        atm_out = Dense(4, activation='sigmoid', kernel_regularizer=l2(0.01),
                        bias_regularizer=l2(0.01), name='atm_preds')(inter_atm)

        inter_common = Dense(128, activation='tanh', kernel_regularizer=l2(0.01),
                        bias_regularizer=l2(0.01), name='inter_common')(x)
        common_out = Dense(7, activation='sigmoid', kernel_regularizer=l2(0.01),
                        bias_regularizer=l2(0.01), name='common_preds')(inter_common)

        inter_rare = Dense(128, activation='tanh', kernel_regularizer=l2(0.01),
                        bias_regularizer=l2(0.01), name='inter_rare')(x)
        rare_out = Dense(6, activation='sigmoid', kernel_regularizer=l2(0.01),
                        bias_regularizer=l2(0.01), name='rare_preds')(inter_rare)

        atm_count = Dense(5, activation='softmax', kernel_regularizer=l2(0.01),
                        bias_regularizer=l2(0.01), name='atm_count_preds')(x)
        common_count = Dense(8, activation='softmax', kernel_regularizer=l2(0.01),
                        bias_regularizer=l2(0.01), name='common_count_preds')(x)
        rare_count = Dense(7, activation='softmax', kernel_regularizer=l2(0.01),
                        bias_regularizer=l2(0.01), name='rare_count_preds')(x)

        output_layers = [atm_out, common_out, rare_out, atm_count, common_count, rare_count]
        self.model = Model(outputs=output_layers, inputs=img_input, name="vgg19_mod")

        # self.model.load_weights(pretrained_weights)
        # print("Pretrained weights loaded!")

        # opt = keras.optimizers.Adam(lr=0.0005)
        opt = keras.optimizers.SGD(lr=0.01, momentum=0.5, nesterov=True)
        # loss_mapping = {'atm_preds':self.custom_loss(outLabel='atm_preds'),
        #                 'common_preds':self.custom_loss(outLabel='common_preds'),
        #                 'rare_preds':self.custom_loss(outLabel='rare_preds'),
        #                 'atm_count_preds': 'categorical_crossentropy', 'common_count_preds': 'categorical_crossentropy',
        #                 'rare_count_preds': 'categorical_crossentropy'}
        loss_mapping = {'atm_preds':'binary_crossentropy',
                        'common_preds':'binary_crossentropy',
                        'rare_preds':'binary_crossentropy',
                        'atm_count_preds': 'categorical_crossentropy', 
                        'common_count_preds': 'categorical_crossentropy',
                        'rare_count_preds': 'categorical_crossentropy'}

        self.model.compile(loss=loss_mapping, optimizer=opt, metrics=['accuracy'])
    
        print("Model compiled!")

    def custom_loss(self, y_true, y_pred, outLabel=None):
        loss = K.binary_crossentropy(y_true, y_pred)
        # loss = np.sum(loss)
        return weighted_loss

    def train(self, options):
        self.train_generator = options['train_generator']
        self.validation_generator = options['validation_generator']
        self.nb_train_samples = int(options['nb_train_samples'])
        self.nb_validation_samples = int(options['nb_validation_samples'])
        self.batch_size = int(options['batch_size'])
        self.nb_epochs = int(options['epochs'])

        print("-----------------------------------------------------------")
        print("Lower changes succeed upper changes. Newest changes are the last.")
        print("Changes:\n")
        print("\tAdd BatchNorm layers.")
        print("\tUse sigmoid instead of softmax.")
        print("\tSeparate outputs according to separate types of classes.")
        print("\tAdd outputs for count of labels - [softmax:categorical_crossentropy].")
        print("\tAdd regularization: l2(0.01)")
        print("\tAdd intermediate dense layers for first 3 outputs. [Relu: 128]")
        print("\tChange activation of dense layers to tanh")
        print("\tAdd Class weights again.")
        # print("\tDecrease learning rate of Adam to 0.0005")
        print("\tChange optimizer to SGD: lr=0.01, momentum:0.5 with nesterov=True.")
        # print("\tResume training from epoch 6.")
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
        class_weights = {'atm_preds':self.atm_weights, 'common_preds':self.common_weights, 'rare_preds':self.rare_weights}
        print("Starting to  train...")
        self.model.fit_generator(self.train_generator, steps_per_epoch=self.nb_train_batches,
                                validation_data=self.validation_generator, validation_steps=self.nb_validation_batches,
                                epochs=self.nb_epochs, verbose=2, callbacks=[callbacks],
                                class_weight=class_weights)


# Custom Keras Callbacks
class Callbacks(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.train_losses = []
        self.out1_losses = []
        self.out2_losses = []
        self.out3_losses = []
        self.out4_losses = []
        self.out5_losses = []
        self.out6_losses = []
        self.out1_accs = []
        self.out2_accs = []
        self.out3_accs = []
        self.out4_accs = []
        self.out5_accs = []
        self.out6_accs = []

        self.val_losses = []
        self.val_out1_losses = []
        self.val_out2_losses = []
        self.val_out3_losses = []
        self.val_out4_losses = []
        self.val_out5_losses = []
        self.val_out6_losses = []
        self.val_out1_accs = []
        self.val_out2_accs = []
        self.val_out3_accs = []
        self.val_out4_accs = []
        self.val_out5_accs = []
        self.val_out6_accs = []
        # print('Model structure: \n', self.model.summary())
        return
 
    def on_train_end(self, logs={}):
        try:
            pickle.dump(self.train_losses, open(train_losses_histFile, 'w'))
            pickle.dump(self.val_losses, open(val_losses_histFile, 'w'))

            pickle.dump(self.out1_losses, open(out1_losses_histfile, 'w'))
            pickle.dump(self.out2_losses, open(out2_losses_histfile, 'w'))
            pickle.dump(self.out3_losses, open(out3_losses_histfile, 'w'))
            pickle.dump(self.out4_losses, open(out4_losses_histfile, 'w'))
            pickle.dump(self.out5_losses, open(out5_losses_histfile, 'w'))
            pickle.dump(self.out6_losses, open(out6_losses_histfile, 'w'))

            pickle.dump(self.out1_accs, open(out1_accs_histfile, 'w'))
            pickle.dump(self.out2_accs, open(out2_accs_histfile, 'w'))
            pickle.dump(self.out3_accs, open(out3_accs_histfile, 'w'))
            pickle.dump(self.out4_accs, open(out4_accs_histfile, 'w'))
            pickle.dump(self.out5_accs, open(out5_accs_histfile, 'w'))
            pickle.dump(self.out6_accs, open(out6_accs_histfile, 'w'))
            
            pickle.dump(self.val_out1_losses, open(val_out1_losses_histfile, 'w'))
            pickle.dump(self.val_out2_losses, open(val_out2_losses_histfile, 'w'))
            pickle.dump(self.val_out3_losses, open(val_out3_losses_histfile, 'w'))
            pickle.dump(self.val_out4_losses, open(val_out4_losses_histfile, 'w'))
            pickle.dump(self.val_out5_losses, open(val_out5_losses_histfile, 'w'))
            pickle.dump(self.val_out6_losses, open(val_out6_losses_histfile, 'w'))

            pickle.dump(self.val_out1_accs, open(val_out1_accs_histfile, 'w'))
            pickle.dump(self.val_out2_accs, open(val_out2_accs_histfile, 'w'))
            pickle.dump(self.val_out3_accs, open(val_out3_accs_histfile, 'w'))
            pickle.dump(self.val_out4_accs, open(val_out4_accs_histfile, 'w'))
            pickle.dump(self.val_out5_accs, open(val_out5_accs_histfile, 'w'))
            pickle.dump(self.val_out6_accs, open(val_out6_accs_histfile, 'w'))
                        
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
        self.val_losses.append(logs.get('val_loss'))

        self.out1_losses.append(logs.get('atm_preds_loss'))
        self.out2_losses.append(logs.get('common_preds_loss'))
        self.out3_losses.append(logs.get('rare_preds_loss'))
        self.out4_losses.append(logs.get('atm_count_preds_loss'))
        self.out5_losses.append(logs.get('common_count_preds_loss'))
        self.out6_losses.append(logs.get('rare_count_preds_loss'))

        self.out1_accs.append(logs.get('atm_preds_acc'))
        self.out2_accs.append(logs.get('common_preds_acc'))
        self.out3_accs.append(logs.get('rare_preds_acc'))
        self.out4_accs.append(logs.get('atm_count_preds_acc'))
        self.out5_accs.append(logs.get('common_count_preds_acc'))
        self.out6_accs.append(logs.get('rare_count_preds_acc'))

        self.val_out1_losses.append(logs.get('val_atm_preds_loss'))
        self.val_out2_losses.append(logs.get('val_common_preds_loss'))
        self.val_out3_losses.append(logs.get('val_rare_preds_loss'))
        self.val_out4_losses.append(logs.get('val_atm_count_preds_loss'))
        self.val_out5_losses.append(logs.get('val_common_count_preds_loss'))
        self.val_out6_losses.append(logs.get('val_rare_count_preds_loss'))

        self.val_out1_accs.append(logs.get('val_atm_preds_acc'))
        self.val_out2_accs.append(logs.get('val_common_preds_acc'))
        self.val_out3_accs.append(logs.get('val_rare_preds_acc'))
        self.val_out4_accs.append(logs.get('val_atm_count_preds_acc'))
        self.val_out5_accs.append(logs.get('val_common_count_preds_acc'))
        self.val_out6_accs.append(logs.get('val_rare_count_preds_acc'))

        print("\nAppended metrics to metrics list.")
        
        try:
            pickle.dump(self.train_losses, open(train_losses_histFile, 'w'))
            pickle.dump(self.val_losses, open(val_losses_histFile, 'w'))

            pickle.dump(self.out1_losses, open(out1_losses_histfile, 'w'))
            pickle.dump(self.out2_losses, open(out2_losses_histfile, 'w'))
            pickle.dump(self.out3_losses, open(out3_losses_histfile, 'w'))
            pickle.dump(self.out4_losses, open(out4_losses_histfile, 'w'))
            pickle.dump(self.out5_losses, open(out5_losses_histfile, 'w'))
            pickle.dump(self.out6_losses, open(out6_losses_histfile, 'w'))

            pickle.dump(self.out1_accs, open(out1_accs_histfile, 'w'))
            pickle.dump(self.out2_accs, open(out2_accs_histfile, 'w'))
            pickle.dump(self.out3_accs, open(out3_accs_histfile, 'w'))
            pickle.dump(self.out4_accs, open(out4_accs_histfile, 'w'))
            pickle.dump(self.out5_accs, open(out5_accs_histfile, 'w'))
            pickle.dump(self.out6_accs, open(out6_accs_histfile, 'w'))

            pickle.dump(self.val_out1_losses, open(val_out1_losses_histfile, 'w'))
            pickle.dump(self.val_out2_losses, open(val_out2_losses_histfile, 'w'))
            pickle.dump(self.val_out3_losses, open(val_out3_losses_histfile, 'w'))
            pickle.dump(self.val_out4_losses, open(val_out4_losses_histfile, 'w'))
            pickle.dump(self.val_out5_losses, open(val_out5_losses_histfile, 'w'))
            pickle.dump(self.val_out6_losses, open(val_out6_losses_histfile, 'w'))

            pickle.dump(self.val_out1_accs, open(val_out1_accs_histfile, 'w'))
            pickle.dump(self.val_out2_accs, open(val_out2_accs_histfile, 'w'))
            pickle.dump(self.val_out3_accs, open(val_out3_accs_histfile, 'w'))
            pickle.dump(self.val_out4_accs, open(val_out4_accs_histfile, 'w'))
            pickle.dump(self.val_out5_accs, open(val_out5_accs_histfile, 'w'))
            pickle.dump(self.val_out6_accs, open(val_out6_accs_histfile, 'w'))

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
        print(logs, '\n')

        # output_tensor = self.model.output
        # variable_tensors = self.model.trainable_weights

        # gradients = K.gradients(output_tensor, variable_tensors)
        # with tf.session() as sess:
        #     sess.run(tf.initialize_all_variables)
        #     sess.run(gradients)

        if batch%50 == 0:
            sys.stdout.write('\n============ 50 batches processed ============\n')
            sys.stdout.flush()
        return
