import json,gensim
import re
path = "/home/intern/video_caption/video_captions/data/MSR-VTT/"
file1 = open(path+"train_val_videodatainfo.json")
temp=open(path+"tem2.txt","w")
from tqdm import tqdm
data = json.load(file1)

headers = {
    # Request headers
    'Content-Type': 'application/x-www-form-urlencoded',
    'Ocp-Apim-Subscription-Key': '698eb09c90814db189397c71e67707f6',
}

params = urllib.urlencode({
    # Request parameters
    'mode': 'spell',
    'mkt': 'en-us',
})

for i in tqdm(data['sentences']):
    s = i['caption']
    s = s.lower()
    s = re.sub('[^0-9a-zA-Z ]', ' ', s)
    s = re.sub( '\s+', ' ', s ).strip()
    text = s.split()
    out = ""
    for j in text:
        if j in vocab or re.search(r'\d', j):
            out+=j+" "
        else:
            try:
                conn = httplib.HTTPSConnection('api.cognitive.microsoft.com')
                conn.request("POST", "/bing/v5.0/spellcheck/?%s" % params, "Text="+j, headers)
                response = conn.getresponse()
                data1 = response.read()
                data1 = json.loads(data1)
                #print sug
                # print type(data['flaggedTokens'])
                #print data['flaggedTokens']
                # print j
                sug = data1['flaggedTokens'][0]['suggestions'][0]['suggestion']
                # print sug
                #temp.write(j+"                     "+sug)
                out+=sug+" "
                conn.close()
            except Exception as e:
                #print count, j
                count+=1
                out+=j+" "
                #print "[Errno] {1}".format(e.strerror)
    out = out.lower()
    out = re.sub('[^0-9a-zA-Z ]', ' ', out)
    out = re.sub( '\s+', ' ', out ).strip()
    #temp.write(i['caption']+"       "+out+"\n")
    i['caption'] = out

with open(path+"train_val_new.json","w+") as out:
	json.dump(data,out)


##To change some strings to <num> and <unk>
data_new = json.load(open(path+"train_val_new.json"))

for i in data_new['sentences']:
    s = i['caption']
    text = s.split()
    for j in text:
        if re.search(r'\d',j):
            text[text.index(j)] = "<num>"
        elif j not in vocab:
            text[text.index(j)] = "<unk>"
    out = " ".join(text)
    out = out.lower()
    out = re.sub('[^0-9a-zA-Z <>]', ' ', out)
    out = re.sub( '\s+', ' ', out ).strip()
    i['caption'] = out

with open(path+"train_val_new.json","w+") as out:
    out.seek(0)
    out.truncate()
    json.dump(data_new,out)


file1.close()
temp.close()