import json,re
path = "/home/intern/video_caption/video_captions/data/MSR-VTT/"

file = open(path + "train_val_videodatainfo.json")
data = json.load(file)
# file.close()

for i in data['sentences']:
	i['caption'] = re.sub('[^0-9a-zA-Z ]',' ',i['caption'])
	
file = open(path + "train_val_videodatainfo.json","w+")
file.seek(0)
file.truncate()
json.dump(data,file)
# file.close()