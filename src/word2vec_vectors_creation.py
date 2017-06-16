import json, gensim
import os,os.path

path = "/home/intern/video_caption/video_captions/data/MSR-VTT/"
data = json.load(open(path + "videodatainfo_2017.json"))

sentences = []

for i in data['sentences']:
	sentences.append(i['caption'].split())

word_list = [word for line in sentences for word in line]

# Number of words in our data
print "Number of words in dataset: " + str(len(word_list))

# Train the word2vec model on our data
model = gensim.models.Word2Vec(sentences,min_count = 1)

word_vectors = model.wv

# Save the model in word2vec bin format
if not os.path.isfile(path + "model.bin"):
	model.wv.save_word2vec_format(path + "model.bin",binary = True)
if not os.path.isfile(path + "model.vec"):
	model.wv.save_word2vec_format(path + "model.vec")