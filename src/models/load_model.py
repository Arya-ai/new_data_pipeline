from __future__ import absolute_import
from __future__ import print_function

import keras
from keras.models import load_model, Model
import os
import sys

# to make sure tf uses cpu, cause gpu busy
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

model_savePath = '../../models/VGG19_mod.hdf5'
model_weights = ''

if os.path.exists(model_savePath):
	model = load_model(model_savePath)
	print("Model loaded!")
else:
	print("Model not found!")
	sys.exit(-1)

if os.path.exists(model_weights):
	model.load_weigths(model_weights)
	print("Weights loaded!")
else:
	print("Model weights not found!")

print("Model Summary:")
model.summary()
