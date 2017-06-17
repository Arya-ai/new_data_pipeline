
# coding: utf-8

# Data Preprocessing Pipeline for this [competition](https://www.kaggle.com/c/planet-understanding-the-amazon-from-space/).

# In[15]:


import os, sys
from subprocess import Popen, PIPE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tifffile as tiff
import cv2
import random
import lmdb
from tqdm import tqdm


# In[11]:


# append the path for custom modules
sys.path.append('../pyimgsaliency/')
import pyimgsaliency

sys.path.append('../new_data_pipeline/')
from datum_pb2 import Datum


# In[4]:


proc = Popen('du -sh ../data/*', shell=True, stdout=PIPE, stderr=PIPE)
print proc.communicate()[0]


# In[6]:


tiff_img_train = '../data/train-tif-v2/'
tiff_img_test = '../data/test-tif-v2/'
lmdb_dir = '../lmdb/'


# In[7]:


df_train = pd.read_csv('../data/train_v2.csv')
print df_train.info()
df_train.head()


# In[8]:


tags = df_train['tags'].apply(lambda x: x.split(' '))

labels = []
for i in xrange(len(df_train)):
    labels.append(tags.values[i])

labels = set([item for sublist in labels for item in sublist])
print "{} unique labels: ".format(len(labels))
for label in labels: print label


# In[9]:


labelmap = {l:i for i, l in enumerate(labels)}
print "labelmap:", labelmap


# In[31]:


# write to LMDB

env = lmdb.open(lmdb_dir + 'traindb', max_dbs=2, map_size=26843545600)
labeldb = env.open_db('labeldb')
key = 0

for fname, tags in tqdm(df_train.values, miniters=500):
    key += 1
    im = tiff.imread(tiff_img_train + fname + '.tif')
    targets = np.zeros(17, dtype='uint8')
    for t in tags.split(' '):
        targets[labelmap[t]] = 1
    
    datum = Datum()
    imageDatum = datum.imgdata.add()
    imageDatum.data = im.tobytes()
    imageDatum.identifier = str(key)
    
    label = Datum()
    labelDatum = label.classs
    labelDatum.multilabel = targets.tobytes()
    labelDatum.identifier = str(key)
    
    with env.begin(write=True) as txn:
        txn.put(str(key).encode('ascii'), datum.SerializeToString())
        
    with env.begin(write=True, db=labeldb) as txn:
        txn.put(str(key).encode('ascii'), label.SerializeToString())
    
env.close()


# In[13]:


# some basic tests
try:
    assert os.path.exists(lmdb_dir + 'datumdb')
except AssertionError:
    print "lmdb database was not created."

