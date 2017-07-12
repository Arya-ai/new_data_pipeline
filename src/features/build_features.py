
# coding: utf-8

# In[1]:


import os, sys
from subprocess import Popen, PIPE
import pandas as pd
import numpy as np
import tifffile as tiff
import random
from tqdm import tqdm


# In[3]:


df_train = pd.read_csv('../data/raw/train_v2.csv')
print df_train.info()
df_train.head()


# In[4]:


tags = df_train['tags'].apply(lambda x: x.split(' '))

labels = []
for i in xrange(len(df_train)):
    labels.append(tags.values[i])

labels = set([item for sublist in labels for item in sublist])
print "{} unique labels: ".format(len(labels))
for label in labels: print label


# In[5]:


labelmap = {l:i for i, l in enumerate(labels)}
print "labelmap:", labelmap


# In[6]:


# create oneHot vector

oneHotOutput = []
for _, tags in tqdm(df_train.values, miniters=500):
    targets = np.zeros(17, dtype='uint8')
    for t in tags.split(' '):
        targets[labelmap[t]] = 1
    
    oneHotOutput.append(targets)


# In[7]:


oneHotOutput = np.array(oneHotOutput)
oneHot_df = pd.DataFrame(oneHotOutput)
print oneHot_df.info()
oneHot_df.head()


# In[10]:


image_binding_df = df_train['image_name']
print image_binding_df.count()
image_binding_df.head()


# In[8]:


oneHot_df.to_csv('../data/processed/output.csv')

image_binding_df.to_csv('../data/processed/image_binding.csv', header=True, index=False)

