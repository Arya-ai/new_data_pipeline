import keras
from keras.models import Model
from keras.layers import Activation, Dense, Input, BatchNormalization, Conv2D, Lambda, AveragePooling3D, Flatten, Conv3D, LSTM
from keras.layers import MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, AveragePooling2D, AveragePooling1D
from keras.layers.merge import concatenate
from keras.applications.imagenet_utils import *
from keras import backend as K
import pickle, sys, os
import numpy as np
from model import *
from datum_pb2 import Datum
import skvideo.io
from pymongo import MongoClient
from model import *

client = MongoClient()

NUM_CAPTIONS = 200000
TRAIN_SIZE = 160000
TEST_SIZE = 40000
BATCH_SIZE = 2
MAX_FRAMES = 600
NUM_CLIPS = 20
CLIP_SIZE = 30
MAX_CAP_LEN = 73
EPOCHS = 5
NUM_BATCHES = NUM_CAPTIONS/BATCH_SIZE
INPUTSHAPE_2D = (240,320,3)
INPUTSHAPE_3D = (30,240,320,3)

path = "/home/intern/video_caption/video_captions/data/MSR-VTT/"
env = lmdb.open(path + 'caption_db')
txn = env.begin()

def data_generator():
	inputs = []
	outputs = []
	inds = np.arange(NUM_CAPTIONS)
	train_idx = inds[:TRAIN_SIZE]
	test_idx = inds[TRAIN_SIZE:]

	while True:
		for i in xrange(NUM_BATCHES):

			#######
			#INPUTS AND OUTPUTS
			########
			all_frames = []
			all_clips = []
			all_words = []
			all_outs = []         #<<<<<-------##### For OUTPUTS  #####
			for j in xrange(BATCH_SIZE):
				ind = str(i*BATCH_SIZE + j)
				raw_datum = txn.get(str(train_idx[ind]))
				datum = Datum().numeric
				datum.ParseFromString(raw_datum)
				video_id = datum.identifier
				###################################################
				#ALL FRAMES
				vid = skvideo.io.vread(path + "TrainValVideo/"+ video_id + ".mp4")
				nframes = vid.shape[0]
				if nframes > MAX_FRAMES:
					temp = np.linspace(0,nframes-1,num=MAX_FRAMES,dtype=int)
					vid = vid[temp]
				elif nframes < MAX_FRAMES:
					pad_zeros_video = np.zeros((MAX_FRAMES-nframes,240,320,3))
					vid = np.concatenate((vid,pad_zeros_video),axis=0)
				if len(all_frames==0):
					all_frames = [np.expand_dims(frame,axis=0) for frame in vid]
				else:
					for i in xrange(MAX_FRAMES):
						frame = np.expand_dims(vid[i],axis=0)
						all_frames[i] = np.concatenate((all_frames[i],frame),axis=0)
				#####################################################################
				#CLIPS
				if len(all_clips)==0:
					for i in xrange(NUM_CLIPS):
						clip = vid[CLIP_SIZE*i:CLIP_SIZE*(i+1)]
						all_clips.append(np.expand_dims(clip, axis=0))
				else:
					for i in xrange(NUM_CLIPS):
						clip = vid[CLIP_SIZE*i:CLIP_SIZE*(i+1)]
						clip = np.expand_dims(clip,axis=0)
						all_clips[i] = np.concatenate((all_clips[i],clip),axis=0)
				####################################################
				#CAPTION VECTOR
				caption = np.fromstring(datum.data,dtype="float32")
				n_words = caption.shape[0]/300
				caption = caption.reshape(n_words,300)
				# caption = np.concatenate((start_vec,caption,end_vec),axis=0)
				out = np.concatenate((caption,end_vec),axis=0)    ######For OUTPUT

				caption = np.concatenate((start_vec,caption),axis=0)
				pad_zeros_caption = np.zeros((MAX_CAP_LEN-caption.shape[0],300))
				caption = np.concatenate((caption,pad_zeros_caption),axis=0)
				if len(all_words)==0:
					all_words = [np.expand_dims(np.expand_dims(word,axis=0),axis=0) for word in caption]
				else:
					for i in xrange(MAX_CAP_LEN):
						word = np.expand_dims(np.expand_dims(caption[i],axis=0),axis=0)
						all_words[i] = np.concatenate((all_words[i],word),axis=0)
				###################################################
				#OUTPUT
				pad_zeros_out = np.zeros((MAX_CAP_LEN-out.shape[0],300))
				out = np.concatenate((out,pad_zeros_out),axis=0)
				out = np.expand_dims(out,axis=0)
				if len(all_outs)==0:
					all_outs = [out]
				else:
					all_outs[0] = np.concatenate((all_outs[0], out), axis = 0)
				
			inputs += all_frames
			inputs += all_clips
			inputs += all_words
			outputs = all_outs

			yield (inputs,outputs)


if __name__ == "__main__":
	# args = sys.argv[1:]
	# if len(args) <= 1:
	# 	print "Usage: python train.py [source_dir]"
	db = client.vectorsdb
	start_vec = db.word_vector.find({"<start>":{"$exists":1}})
	start_vec = list(start_vec)
	start_vec = cPickle.loads(vec[0]["<start>"])
	start_vec = np.expand_dims(start_vec,axis=0)    # Use in Inputs
	end_vec = db.word_vector.find({"<end>":{"$exists":1}})
	end_vec = list(end_vec)
	end_vec = cPickle.loads(vec[0]["<end>"])
	end_vec = np.expand_dims(end_vec,axis=0)	    # Use in Outputs
	model = create_model(MAX_FRAMES,NUM_CLIPS,MAX_CAP_LEN,INPUTSHAPE_2D,INPUTSHAPE_3D)
	model.fit_generator(data_generator, steps_per_epoch = 100000)
