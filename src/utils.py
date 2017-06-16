# import cv2
# import ipdb
import numpy as np
import os, sys
from pymongo import MongoClient
import json
import lmdb
from datum_pb2 import Datum
import cPickle
from tqdm import tqdm

path = "/home/intern/video_caption/video_captions/data/MSR-VTT/"
# os.system('protoc -I={0} --python_out={1} {0}datum.proto'.format("/home/intern/video_caption/video_captions/data/", "/home/intern/video_caption/video_captions/src/"))

env = lmdb.open(path + 'caption_db',map_size = 4294967296)

client = MongoClient()
db = client.vectorsdb

def write_caption_to_lmdb(video_id, caption, index):

	# Datum creation for caption
	caption_datum = Datum().numeric
	caption_datum.size.dim = 2
	caption_datum.identifier = video_id

	vecs = []
	words = caption.split()
	for i in words:
		vec = db.word_vector.find({i:{"$exists":1}})
		vec = list(vec)
		vec = cPickle.loads(vec[0][i])
		vecs.append(vec)

	vecs = np.asarray(vecs)

	caption_datum.data = vecs.tobytes()

	# str_id = '{:08}'.format(index)

	with env.begin(write=True) as txn:
		txn.put(index,caption_datum.SerializeToString())

if __name__ == '__main__':

	#Captions
	cap_path = path + "videodatainfo_new.json"
	data = json.load(open(cap_path))
	count = 0

	out = open("map_dict.pkl","wb")

	for i in tqdm(data['sentences']):
		index = str(count).encode('ascii')
		write_caption_to_lmdb(i['video_id'],i['caption'], index)
		# if count%50000==0:
		# 	print "Completed Reading caption "+str(count)
		count += 1











