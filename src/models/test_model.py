from __future__ import absolute_import
from __future__ import print_function

import keras
from keras import backend as K
from keras.models import load_model
# from keras.optimizers import Adam
import os, sys, gc
import lmdb
import cPickle as pickle
import numpy as np
from sklearn.metrics import fbeta_score

sys.path.append('/home/intern/satellite/amazon-satellite-imagery/')
from datum_pb2 import Datum

# to make sure tf uses cpu, cause gpu busy
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# supress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

model_savePath = '../../models/VGG19_model_4.hdf5'
model_weights = '../../models/VGG19_12weights.08-1.88.hdf5'
test_keys_pickle = '../../models/VGG19_5/test_keys.pkl'
LMDB_DIR = '../../data/interim/lmdb/traindb'
batch_size = 32
root_dir = '../../models/VGG19_12/'

if os.path.exists(model_savePath):
	model = load_model(model_savePath)
	print("Model loaded!")
else:
	print("Model not found!")
	sys.exit(-1)

if os.path.exists(model_weights):
	model.load_weights(model_weights)
	print("Weights loaded!")
else:
	print("Model weights not found!")
	sys.exit(-1)

loss_mapping = {'atm_preds':'binary_crossentropy',
                'common_preds':'binary_crossentropy',
                'rare_preds':'binary_crossentropy',
                'atm_count_preds': 'categorical_crossentropy', 
                'common_count_preds': 'categorical_crossentropy',
                'rare_count_preds': 'categorical_crossentropy'}

opt = keras.optimizers.SGD(lr=0.01, momentum=0.5, nesterov=True)
model.compile(loss=loss_mapping, optimizer=opt, metrics=['accuracy'])
print("Metrics for the model: ", model.metrics_names)

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

print("Found {} files for testing".format(len(test_keys)))
nb_batches = len(test_keys) // batch_size
print("Testing {} batches".format(nb_batches))

metrics = []
metrics_pickle = root_dir + 'test_batch_metrics.pkl'

FBetaScores = []
FBetaScores_pickle = root_dir + 'fbeta_batch_scores.pkl'

idx = 0
for batch in xrange(1, nb_batches + 1):
	batch_inputs = []
	batch_outputs_1 = []
	batch_outputs_2 = []
	batch_outputs_3 = []
	outputs = []

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
			out_1 = np.take(flat_x, [1,4,7,9])
			out_2 = np.take(flat_x, [3,6,8,11,12,14,15])
			out_3 = np.take(flat_x, [0,2,5,10,13,16])
			batch_outputs_1.append(out_1)
			batch_outputs_2.append(out_2)
			batch_outputs_3.append(out_3)
		del datum		# release memory

		idx += 1

	batch_inputs = np.array(batch_inputs)
	batch_outputs_1 = np.asarray(batch_outputs_1, dtype='int')
	oneHot_count_1 = np.zeros((batch_size, 5), dtype='int')
	count_1 = batch_outputs_1.sum(axis=1)
	oneHot_count_1[tuple(range(batch_size)), tuple(count_1)] = 1

	batch_outputs_2 = np.asarray(batch_outputs_2, dtype='int')
	oneHot_count_2 = np.zeros((batch_size, 8), dtype='int')
	count_2 = batch_outputs_2.sum(axis=1)
	oneHot_count_2[tuple(range(batch_size)), tuple(count_2)] = 1

	batch_outputs_3 = np.asarray(batch_outputs_3, dtype='int')
	oneHot_count_3 = np.zeros((batch_size, 7), dtype='int')
	count_3 = batch_outputs_3.sum(axis=1)
	oneHot_count_3[tuple(range(batch_size)), tuple(count_3)] = 1

	outputs.append(batch_outputs_1)
	outputs.append(batch_outputs_2)
	outputs.append(batch_outputs_3)
	outputs.append(oneHot_count_1)
	outputs.append(oneHot_count_2)
	outputs.append(oneHot_count_3)

	batch_preds = model.predict_on_batch(batch_inputs)
	logs = model.test_on_batch(batch_inputs, outputs)

	oneHot_preds = []
	for preds in zip(*batch_preds):
		sorted_preds = [np.argsort(pred)[::-1] for pred in preds[:3]]
		oneHots = [np.zeros(pred.shape, dtype='int') for pred in sorted_preds]
		counts = [np.argmax(np_count) for np_count in preds[3:]]
		for count, pred, oneHot in zip(counts, sorted_preds, oneHots):
		    for index in xrange(count):
		        oneHot[pred[index]] = 1

		oneHot_preds.append(oneHots)

	oneHot_preds = np.array(oneHot_preds)
	shape = (oneHot_preds[:,0].shape[0], oneHot_preds[0,0].shape[0])
	y_preds_1 = np.concatenate(oneHot_preds[:,0]).reshape(shape)

	shape = (oneHot_preds[:,1].shape[0], oneHot_preds[0,1].shape[0])
	y_preds_2 = np.concatenate(oneHot_preds[:,1]).reshape(shape)

	shape = (oneHot_preds[:,2].shape[0], oneHot_preds[0,2].shape[0])
	y_preds_3 = np.concatenate(oneHot_preds[:,2]).reshape(shape)

	oneHot_preds = np.asarray(oneHot_preds)
	fbeta_scores = []
	fbeta_scores.append(fbeta_score(batch_outputs_1, y_preds_1, beta=2, average='samples'))
	fbeta_scores.append(fbeta_score(batch_outputs_2, y_preds_2, beta=2, average='samples'))
	fbeta_scores.append(fbeta_score(batch_outputs_3, y_preds_3, beta=2, average='samples'))

	FBetaScores.append(fbeta_scores)
	metrics.append(logs)
	pickle.dump(metrics, open(metrics_pickle, 'w'))
	pickle.dump(FBetaScores, open(FBetaScores_pickle, 'w'))
	print("Metrics for batch {}:".format(batch), logs)
	print("Fbeta scores for batch {}:".format(batch), fbeta_scores)

	gc.collect()

print("Testing complete.")
try:
	env.close()
	print("LMDB transaction closed.")
except Exception as e:
	print("Error closing lmdb transaction: ", e)
