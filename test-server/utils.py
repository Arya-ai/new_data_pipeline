from __future__ import print_function
import cv2
import numpy as np
import sys
from keras.models import load_model

# to ensure CUDA uses cpu for computations, cause training on GPU
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

model_path = os.path.abspath('../amazon-satellite-imagery/models/VGG19_model_4.hdf5')
weights_path = os.path.abspath('../amazon-satellite-imagery/models/VGG19_12weights.44-2.53.hdf5')
print("New weights loaded!")

def get_image_from_buffer(image_buffer):
	img = np.asarray(image_buffer, dtype=np.uint8)
	img = cv2.imdecode(img, -1)
	img = np.asarray(img, dtype='float')
	print("Decoded image shape:", img.shape)

	if len(img.shape) < 3:
		print("Error: Uploaded image contains {} channels.".format(len(img.shape)) + \
			"Uploaded images should have atleast 3 channels.")
		return None

	if len(img.shape) == 3 and img.shape[2]>4:	#tiff image
		img = img[:, :, :4]
	#        elif len(img.shape)>2:
	#                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	elif img.shape[2] == 3:	# jpg image
		# img = img[:,:,:3]

		# workaround to add an extra channel to .jpg images 
		# to make up for NIR channel

		# print("orig image:", img.shape)
		# print("last axis:", img[:,:,2].shape)
		img = np.concatenate((img, img[:,:,2:]), axis=-1)

	return img

def loadModel():
	model = load_model(model_path)
	model.load_weights(weights_path)
	# print("Model loaded with random weights.")
	return model

def predict(model, images):
	'''
	inputs:
	------
		model: A keras model instance

		images: a single image (or) a list of images

	outputs:
	-------
		confidence_ndarray: Prediction (or) a list of predictions for the image(s)
	'''

	if isinstance(images, list):	# bunch of images
		# Normalization
		for image in images:
			for i in range(image.shape[-1]):
				image[:,:,i] = (image[:,:,i] - image[:,:,i].min())/(image[:,:,i].max() - image[:,:,i].min())

		batch = np.array(images)
	else:
		# Normalization
		for i in range(images.shape[-1]):
				images[:,:,i] = (images[:,:,i] - images[:,:,i].min())/(images[:,:,i].max() - images[:,:,i].min())

		# to include the batch size as 1
		batch = [images]
		batch = np.array(batch)

	print("Got images of size {}".format(batch.shape))

	try:
		confidence_ndarray = model.predict(batch)
		return confidence_ndarray
	except Exception as e:
		print("can't predict labels for the image. Error: " + repr(e))
		return None

def split(image):
	'''
	image is a numpy array with one or two dimensions greater than 256
	
	inputs:
	------
		image: A numpy array

	outputs:
	-------
		split_images: A list of splitted numpy arrays of size (256,256,nChannels)
	'''

	height_ratio = image.shape[0]/256.0
	width_ratio = image.shape[1]/256.0

	print("height_ratio:", height_ratio)
	print("width_ratio:", width_ratio)

	split_images = []

	for hsplit in xrange(int(width_ratio)):
		for vsplit in xrange(int(height_ratio)):
			cropped = image[ vsplit * 256 : (vsplit + 1) * 256, hsplit * 256 : (hsplit + 1) * 256,:]
			split_images.append(cropped)

		if height_ratio > float(int(height_ratio)):
			cropped = image[-256 : , hsplit * 256 : (hsplit + 1) * 256, :]
			split_images.append(cropped)

	if width_ratio > float(int(width_ratio)):
		for vsplit in xrange(int(height_ratio)):
			cropped = image[vsplit * 256 : (vsplit + 1) * 256, -256 :, :]
			split_images.append(cropped)

		if height_ratio > float(int(height_ratio)):
			cropped = image[-256 :, -256 : ,:]
			split_images.append(cropped)

	return split_images


def get_json_from_ndarray(confidence_ndarray=None):
	atm_labels = ['clear', 'cloudy', 'haze', 'partly_cloudy']
	common_labels = ['primary', 'water', 'cultivation', 'habitation', 'bare_ground', 'agriculture', 'road',]
	rare_labels = ['slash_burn', 'blooming', 'conventional_mine', 'artisinal_mine', 'blow_down', 'selective_logging']

	atm_json = {}
	common_json = {}
	rare_json = {}

	atm, common, rare, atm_count, common_count, rare_count = confidence_ndarray
	atm_count = np.argmax(atm_count[0])
	common_count = np.argmax(common_count[0])
	rare_count = np.argmax(rare_count[0])

	# [::-1] - reverses after argsort
	sorted_atm_idxs = np.argsort(atm[0])[::-1]
	sorted_common_idxs = np.argsort(common[0])[::-1]
	sorted_rare_idxs = np.argsort(rare[0])[::-1]

	for idx in xrange(atm_count):
		atm_json[atm_labels[sorted_atm_idxs[idx]]] = float(atm[0][sorted_atm_idxs[idx]])

	for idx in xrange(common_count):
		common_json[common_labels[sorted_common_idxs[idx]]] = float(common[0][sorted_common_idxs[idx]])

	for idx in xrange(rare_count):
		rare_json[rare_labels[sorted_rare_idxs[idx]]] = float(rare[0][sorted_rare_idxs[idx]])


	common_json.update(rare_json)
	print("atm:", atm_json)
	print("land:", common_json)
	# rare_json = dict((label, confidence) for label, confidence in zip(rare_labels, rare[0].tolist()))

	return (atm_json, common_json)