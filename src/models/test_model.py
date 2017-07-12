from __future__ import absolute_import
from __future__ import print_function

from keras.models import load_model
from keras.optimizers import Adam
import os, sys, gc
import lmdb
import cPickle as pickle
import numpy as np

sys.path.append('/home/intern/satellite/amazon-satellite-imagery/')
from datum_pb2 import Datum

# to make sure tf uses cpu, cause gpu busy
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

model_savePath = '../../models/VGG19_model.hdf5'
model_weights = '../../models/VGG19_5/weights.08-1.55.hdf5'
test_keys_pickle = '../../models/VGG19_5/test_keys.pkl'
LMDB_DIR = '../../data/interim/lmdb/traindb'
batch_size = 32
root_dir = '../../models/VGG19_5/'

if os.path.exists(model_savePath):
	model = load_model(model_savePath)
	print("Model loaded!")
else:
	print("Model not found!")
	sys.exit(-1)

model.summary()

if os.path.exists(model_weights):
	model.load_weights(model_weights)
	print("Weights loaded!")
else:
	print("Model weights not found!")
	sys.exit(-1)

opt = Adam(lr=1.0)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
print("Model compiled!")

try:
	env = lmdb.open(LMDB_DIR, max_dbs=3, readonly=True)
except Exception as e:
	print("Error opening lmdb:", e)
	sys.exit(-1)

inputDBHandle = env.open_db('datumdb0')
outputDBHandle = env.open_db('labeldb0')

test_keys = pickle.load(open(test_keys_pickle, 'r'))

txn = env.begin(write=False)
inputCursor = txn.cursor(inputDBHandle)
outputCursor = txn.cursor(outputDBHandle)

nb_batches = len(test_keys) // batch_size

metrics = []
metrics_pickle = root_dir + 'test_batch_metrics.pkl'

idx = 0
for batch in xrange(1, nb_batches + 1):
	batch_inputs = []
	batch_outputs = []

	for sample in xrange(batch_size):
		key = test_keys[idx]
		try:
			raw_datum = inputCursor.get(str(key))
			datum = Datum()
			datum.ParseFromString(raw_datum)
		except Exception as e:
			print("Problem reading training output with key: {} with index: {}".format(key, idx))
			print("Error:", e)
			sys.exit(-1)

		if datum.imgdata.data:
			flat_x = np.fromstring(datum.imgdata.data, dtype='float32')
			x = flat_x.reshape((datum.imgdata.height, datum.imgdata.width, datum.imgdata.channels))
			for i in range(x.shape[-1]):
				x[:,:,i] = (x[:,:,i] - x[:,:,i].min())/(x[:,:,i].max() - x[:,:,i].min())
			batch_inputs.append(x)
		elif datum.numeric.data:
			flat_x = np.fromstring(datum.numeric.data, dtype='float32')
			batch_inputs.append(flat_x)
		del datum 		# release memory

		try:
			raw_datum = outputCursor.get(str(key))
			datum = Datum()
			datum.ParseFromString(raw_datum)
		except Exception as e:
			print("Problem reading training output with key: {} with index: {}".format(key, idx))
			print("Error:", e)
			sys.exit(-1)

		if datum.imgdata.data:
			flat_x = np.fromstring(datum.imgdata.data, dtype='float32')
			x = flat_x.reshape((datum.imgdata.height, datum.imgdata.width, datum.imgdata.channels))
			for i in range(x.shape[-1]):
				x[:,:,i] = (x[:,:,i] - x[:,:,i].min())/(x[:,:,i].max() - x[:,:,i].min())
			batch_outputs.append(x)
		elif datum.numeric.data:
			flat_x = np.fromstring(datum.numeric.data, dtype='float32')
			batch_outputs.append(flat_x)
		del datum 		# release memory

		idx += 1

	batch_inputs = np.array(batch_inputs)
	batch_outputs = np.array(batch_outputs)

	logs = model.test_on_batch(batch_inputs, batch_outputs)
	print("Metrics for batch {}:".format(batch), logs)
	metrics.append(logs)

	pickle.dump(metrics, open(metrics_pickle, 'w'))

	gc.collect()

print("Testing complete.")
try:
	txn.close()
	print("LMDB transaction closed.")
except Exception as e:
	print("Error closing lmdb transaction: ", e)
