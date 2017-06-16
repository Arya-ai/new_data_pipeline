import json
import os,os.path
import subprocess as sp

path = "/home/intern/video_caption/video_captions/data/MSR-VTT/"
data = json.load(open(path + "videodatainfo_new.json"))

outfile = open("/home/intern/video_caption/video_captions/data/word_vectors/text.txt","w")

for i in data['sentences']:
	outfile.write("<start> " + i['caption'].encode("utf-8") + " <end>" + " ")
outfile.close()
