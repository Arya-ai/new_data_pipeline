from pymongo import MongoClient
import json
import gensim
from bson import Binary
import cPickle

path = "/home/intern/video_caption/video_captions/data/MSR-VTT/"
data = json.load(open(path + "videodatainfo_new.json"))

sentences = []

for i in data['sentences']:
	sentences.append(i['caption'].split())

word_list = list(set([word for line in sentences for word in line]))+["<start>","<end>"]

client = MongoClient()
db = client.vectorsdb

model = gensim.models.KeyedVectors.load_word2vec_format(path + "model.vec")

#Insert One by one

# for i in word_list:
# 	db.word_vector.insert_one(
# 	{
# 		i: Binary(cPickle.loads(model[i],protocol=2))
# 	}
# 		)

# Insert all at a time

result = db.word_vector.insert_many([{i: Binary(cPickle.dumps(model[i],protocol=2))} for i in word_list])
