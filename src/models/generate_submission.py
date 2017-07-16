from __future__ import absolute_import
from __future__ import print_function

from keras.models import load_model
import os, sys, math, re
import numpy as np
import pandas as pd
from skimage import io

# to make sure tf uses cpu, cause gpu busy
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

# supress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

model_savePath = os.path.abspath('../../models/VGG19_model_4.hdf5')
model_weights = os.path.abspath('../../models/VGG19_12/weights.24-2.08.hdf5')
batch_size = 32
root_dir = os.path.abspath('../../models/VGG19_12/')
test_dir = os.path.abspath('../../data/raw/test-tif-v2')
submission_df_file = os.path.join(root_dir, 'submission_SGD_25.csv')

# get rid of previous results
if os.path.exists(submission_df_file):
	os.remove(submission_df_file)

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

nb_test_samples = len([name for name in os.listdir(test_dir) if os.path.isfile(os.path.join(test_dir, name))])
# one more batch to cover the last files
nb_batches = int(math.ceil(nb_test_samples/(batch_size*1.0)))

print("Found {} testing samples".format(nb_test_samples))
print("Batches:", nb_batches)

submission_df = pd.DataFrame(columns=['image_name', 'tags'])
labels = [['clear', 'cloudy', 'haze', 'partly_cloudy'],
		  ['primary', 'water', 'cultivation', 'habitation', 'bare_ground', 'agriculture', 'road',],
		  ['slash_burn', 'blooming', 'conventional_mine', 'artisinal_mine', 'blow_down', 'selective_logging']]

batch = 0

for batch in xrange(nb_batches):
	batch_inputs = []
	batch_file_names = []
	upper_bound = min((batch + 1)*batch_size, nb_test_samples)
	for file in os.listdir(test_dir)[batch*batch_size: upper_bound]:

		image_path = os.path.join(test_dir, file)
		img = io.imread(image_path).astype(np.float32)
		for i in range(img.shape[-1]):
			img[:,:,i] = (img[:,:,i] - img[:,:,i].min())/(img[:,:,i].max() - img[:,:,i].min())

		batch_file_names.append(image_path.split('/')[-1].split('.')[0])
		batch_inputs.append(img)

	batch_inputs = np.array(batch_inputs)
	batch_preds = model.predict_on_batch(batch_inputs)

	for file_name, preds in zip(batch_file_names, zip(*batch_preds)):
		sorted_preds = [np.argsort(pred)[::-1] for pred in preds[:3]]
		counts = [np.argmax(np_count) for np_count in preds[3:]]
		tags = ''
		for count, pred, label in zip(counts, sorted_preds, labels):
			for idx in xrange(count):
				# print label[pred[idx]]
				tags = tags + label[pred[idx]] + ' '
		tags = tags.strip()
		file_idx = re.findall('(\d+)', file_name)[0]
		submission_df.loc[file_idx] = [file_name, tags]

	print("Done with batch {}".format(batch))

try:
	submission_df.to_csv(submission_df_file, index=False)
except Exception as e:
	print("Error:", e)

print("Done.")