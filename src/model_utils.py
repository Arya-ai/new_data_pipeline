from __future__ import print_function
from __future__ import absolute_import
import keras
from keras.models import Model
from keras.layers import Activation, Dense, Input, BatchNormalization, Conv2D
from keras.layers import MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, AveragePooling2D
from keras.applications.imagenet_utils import *
from keras import backend as K
import cPickle


WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.5/inception_v3_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.5/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

layers = []


def conv2d_bn(filters,
              num_row,
              num_col,
              padding='same',
              strides=(1, 1),
              name=None):
	global layers
	bn_axis = 3   # Channels last
	x_1 = Conv2D(filters, (num_row, num_col),strides=strides,padding=padding,use_bias=False)
	x_2 = BatchNormalization(axis=bn_axis, scale=False)
	x_3 = Activation('relu')
	layers+=[x_1,x_2,x_3]
    # return x
def max_pool2d(pool_size=(2,2),strides=None, padding='valid', data_format=None):
	x = MaxPooling2D(pool_size=pool_size, strides=strides)
	global layers
	layers.append(x)

def avg_pool2d(pool_size=(2, 2), strides=None, padding='valid', data_format=None):
	x = AveragePooling2D(pool_size=pool_size, strides=strides, padding=padding)
	global layers
	layers.append(x)

def InceptionV3(include_top=True,
                weights='imagenet',
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=1000):

    # Determine proper input shape
    # input_shape = _obtain_input_shape(
    #     input_shape,
    #     default_size=299,
    #     min_size=139,
    #     data_format=K.image_data_format(),
    #     include_top=include_top)

    # if input_tensor is None:
    #     img_input = Input(shape=input_shape)
    # else:
    #     if not K.is_keras_tensor(input_tensor):
    #         img_input = Input(tensor=input_tensor, shape=input_shape)
    #     else:
    #         img_input = input_tensor

    # if K.image_data_format() == 'channels_first':
    #     channel_axis = 1
    # else:
    #     channel_axis = 3

	conv2d_bn(32, 3, 3, strides=(2, 2), padding='valid')     # x = img_input
	conv2d_bn(32, 3, 3, padding='valid')                     # x = x
	conv2d_bn(64, 3, 3)										# x = x
	max_pool2d((3, 3), strides=(2, 2))						# x = x

	conv2d_bn(80, 1, 1, padding='valid')					# x = x
	conv2d_bn(192, 3, 3, padding='valid')					# x = x
	max_pool2d((3, 3), strides=(2, 2))						# x = x

	# mixed 0, 1, 2: 35 x 35 x 256
	conv2d_bn(64, 1, 1)										#branch1x1 = x

	conv2d_bn(48, 1, 1)										#branch5x5 = x
	conv2d_bn(64, 5, 5)										#branch5x5 = branch5x5

	conv2d_bn(64, 1, 1)
	conv2d_bn(96, 3, 3)
	conv2d_bn(96, 3, 3)

	avg_pool2d((3, 3), strides=(1, 1), padding='same')
	conv2d_bn(32, 1, 1)
	# x = layers.concatenate(
	#     [branch1x1, branch5x5, branch3x3dbl, branch_pool],
	#     axis=channel_axis,
	#     name='mixed0')

	# mixed 1: 35 x 35 x 256
	conv2d_bn(64, 1, 1)

	conv2d_bn(48, 1, 1)
	conv2d_bn(64, 5, 5)

	conv2d_bn(64, 1, 1)
	conv2d_bn(96, 3, 3)
	conv2d_bn(96, 3, 3)

	avg_pool2d((3, 3), strides=(1, 1), padding='same')
	conv2d_bn(64, 1, 1)
	# x = layers.concatenate(
	#     [branch1x1, branch5x5, branch3x3dbl, branch_pool],
	#     axis=channel_axis,
	#     name='mixed1')

	# mixed 2: 35 x 35 x 256
	conv2d_bn(64, 1, 1)

	conv2d_bn(48, 1, 1)
	conv2d_bn(64, 5, 5)

	conv2d_bn(64, 1, 1)
	conv2d_bn(96, 3, 3)
	conv2d_bn(96, 3, 3)

	avg_pool2d((3, 3), strides=(1, 1), padding='same')
	conv2d_bn(64, 1, 1)
	# x = layers.concatenate(
	#     [branch1x1, branch5x5, branch3x3dbl, branch_pool],
	#     axis=channel_axis,
	#     name='mixed2')

	# mixed 3: 17 x 17 x 768
	conv2d_bn(384, 3, 3, strides=(2, 2), padding='valid')

	conv2d_bn(64, 1, 1)
	conv2d_bn(96, 3, 3)
	conv2d_bn(96, 3, 3, strides=(2, 2), padding='valid')

	max_pool2d((3, 3), strides=(2, 2))
	# x = layers.concatenate(
	#     [branch3x3, branch3x3dbl, branch_pool], axis=channel_axis, name='mixed3')

	# mixed 4: 17 x 17 x 768
	conv2d_bn(192, 1, 1)

	conv2d_bn(128, 1, 1)
	conv2d_bn(128, 1, 7)
	conv2d_bn(192, 7, 1)

	conv2d_bn(128, 1, 1)
	conv2d_bn(128, 7, 1)
	conv2d_bn(128, 1, 7)
	conv2d_bn(128, 7, 1)
	conv2d_bn(192, 1, 7)

	avg_pool2d((3, 3), strides=(1, 1), padding='same')
	conv2d_bn(192, 1, 1)
	# x = layers.concatenate(
	#     [branch1x1, branch7x7, branch7x7dbl, branch_pool],
	#     axis=channel_axis,
	#     name='mixed4')

	# mixed 5, 6: 17 x 17 x 768
	for i in range(2):
	    conv2d_bn(192, 1, 1)

	    conv2d_bn(160, 1, 1)
	    conv2d_bn(160, 1, 7)
	    conv2d_bn(192, 7, 1)

	    conv2d_bn(160, 1, 1)
	    conv2d_bn(160, 7, 1)
	    conv2d_bn(160, 1, 7)
	    conv2d_bn(160, 7, 1)
	    conv2d_bn(192, 1, 7)

	    avg_pool2d((3, 3), strides=(1, 1), padding='same')
	    conv2d_bn(192, 1, 1)
	    # x = layers.concatenate(
	    #     [branch1x1, branch7x7, branch7x7dbl, branch_pool],
	    #     axis=channel_axis,
	    #     name='mixed' + str(5 + i))

	# mixed 7: 17 x 17 x 768
	conv2d_bn(192, 1, 1)
	conv2d_bn(192, 1, 1)
	conv2d_bn(192, 1, 7)
	conv2d_bn(192, 7, 1)

	conv2d_bn(192, 1, 1)
	conv2d_bn(192, 7, 1)
	conv2d_bn(192, 1, 7)
	conv2d_bn(192, 7, 1)
	conv2d_bn(192, 1, 7)

	avg_pool2d((3, 3), strides=(1, 1), padding='same')
	conv2d_bn(192, 1, 1)
	# x = layers.concatenate(
	#     [branch1x1, branch7x7, branch7x7dbl, branch_pool],
	#     axis=channel_axis,
	#     name='mixed7')

	# mixed 8: 8 x 8 x 1280
	conv2d_bn(192, 1, 1)
	conv2d_bn(320, 3, 3,strides=(2, 2), padding='valid')

	conv2d_bn(192, 1, 1)
	conv2d_bn(192, 1, 7)
	conv2d_bn(192, 7, 1)
	conv2d_bn(192, 3, 3, strides=(2, 2), padding='valid')

	max_pool2d((3, 3), strides=(2, 2))
	# x = layers.concatenate(
	#     [branch3x3, branch7x7x3, branch_pool], axis=channel_axis, name='mixed8')

	# mixed 9: 8 x 8 x 2048
	for i in range(2):
	    conv2d_bn(320, 1, 1)
	    conv2d_bn(384, 1, 1)
	    conv2d_bn(384, 1, 3)
	    conv2d_bn(384, 3, 1)
	    # branch3x3 = layers.concatenate(
	    #     [branch3x3_1, branch3x3_2], axis=channel_axis, name='mixed9_' + str(i))

	    conv2d_bn(448, 1, 1)
	    conv2d_bn(384, 3, 3)
	    conv2d_bn(384, 1, 3)
	    conv2d_bn(384, 3, 1)
	    # branch3x3dbl = layers.concatenate(
	    #     [branch3x3dbl_1, branch3x3dbl_2], axis=channel_axis)

	    avg_pool2d((3, 3), strides=(1, 1), padding='same')
	    conv2d_bn(192, 1, 1)
	    # x = layers.concatenate(
	    #     [branch1x1, branch3x3, branch3x3dbl, branch_pool],
	    #     axis=channel_axis,
	    #     name='mixed' + str(9 + i))
	if include_top:
	    # Classification block
	    layers.append(GlobalAveragePooling2D(name='avg_pool'))
	    layers.append(Dense(classes, activation='softmax', name='predictions'))
	else:
	    if pooling == 'avg':
	        layers.append(GlobalAveragePooling2D())
	    elif pooling == 'max':
	        layers.append(GlobalMaxPooling2D())

	# Ensure that the model takes into account
	# any potential predecessors of `input_tensor`.
	# if input_tensor is not None:
	#     inputs = get_source_inputs(input_tensor)
	# else:
	#     inputs = img_input
	# # Create model.
	# model = Model(inputs, x, name='inception_v3')

	# # load weights
	# if weights == 'imagenet':
	#     if K.image_data_format() == 'channels_first':
	#         if K.backend() == 'tensorflow':
	#             warnings.warn('You are using the TensorFlow backend, yet you '
	#                           'are using the Theano '
	#                           'image data format convention '
	#                           '(`image_data_format="channels_first"`). '
	#                           'For best performance, set '
	#                           '`image_data_format="channels_last"` in '
	#                           'your Keras config '
	#                           'at ~/.keras/keras.json.')
	#     if include_top:
	#         weights_path = get_file(
	#             'inception_v3_weights_tf_dim_ordering_tf_kernels.h5',
	#             WEIGHTS_PATH,
	#             cache_subdir='models',
	#             md5_hash='9a0d58056eeedaa3f26cb7ebd46da564')
	#     else:
	#         weights_path = get_file(
	#             'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5',
	#             WEIGHTS_PATH_NO_TOP,
	#             cache_subdir='models',
	#             md5_hash='bcbd6486424b2319ff4ef7d526e38f63')
	#     model.load_weights(weights_path)
	# return model

InceptionV3(include_top=False,weights='imagenet',input_tensor=None,input_shape=None, pooling='avg')

pkl = open("layers.pkl","wb")
cPickle.dump(layers, pkl)

def preprocess_input(x):
	x /= 255.
	x -= 0.5
	x *= 2.
	return x
