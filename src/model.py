from __future__ import absolute_import
import keras
from keras.models import Model
from keras.layers import Activation, Dense, Input, BatchNormalization, Conv2D, Lambda, AveragePooling3D, Flatten, Conv3D, LSTM
from keras.layers import MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, AveragePooling2D, AveragePooling1D
from keras.layers.merge import concatenate
from keras.applications.imagenet_utils import *
from keras.utils.data_utils import get_file
from keras import backend as K
import pickle
import numpy as np


def create_model(nframes,nclips,max_cap_len,input_shape2d,input_shape3d):

    input_layers=[]
    outputs1=[]

    
    layer_1 = Conv2D(64, (3, 3), activation='relu', padding='same')
    layer_2 = Conv2D(64, (3, 3), activation='relu', padding='same')
    layer_3 = Lambda(lambda x:K.expand_dims(x,1))

    for i in xrange(nframes):
        inp = Input(shape=input_shape2d)                #shape = shape of one sample
        input_layers.append(inp)
        x = layer_1(inp)
        y = layer_2(x)
        out = layer_3(y)
        outputs1.append(out)
    con1 = concatenate(outputs1,axis=1)
    avg_pool2d = AveragePooling3D(pool_size=(nframes,2,2))(con1)
    f1 = Flatten()(avg_pool2d)
    dense1 = Dense(256)(f1)

    layer_4 = Conv3D(64,(3,3,3),activation = 'relu',padding='same')

    outputs2 = []

    for i in xrange(nclips):
        inp = Input(shape=input_shape3d)
        input_layers.append(inp)
        out = layer_4(inp)  
        outputs2.append(out)
    con2 = concatenate(outputs2,axis=1)
    avg_pool3d = AveragePooling3D(pool_size=(nframes,2,2))(con2)
    f2 = Flatten()(avg_pool3d)
    dense2 = Dense(256)(f2)

    vid_feat = concatenate([dense1,dense2])
    final_vid_feat = Lambda(lambda x:K.expand_dims(x,1))(vid_feat)

    layer_5 = Dense(512)
    cap_outs = []

    for i in xrange(max_cap_len):
        inp = Input(shape=(1,300))
        input_layers.append(inp)
        out = layer_5(inp)
        cap_outs.append(out)

    cap_final = concatenate(cap_outs,axis=1)
    lstm_input = concatenate([final_vid_feat,cap_final],axis=1)

    output = LSTM(300, return_sequences=True)(lstm_input)

    model = Model(inputs = input_layers,outputs = output)

    model.compile(optimizer="sgd",loss="mean_squared_logarithmic_error",metric="accuracy")

    print "compilation done"
    return model

def pred(model):
    ins=[]

    for i in xrange(50):
        ins.append(np.random.randn(1,240,240,3))
    for i in xrange(5):
        ins.append(np.random.randn(1,10,240,240,3))
    for i in xrange(72):
        ins.append(np.random.randn(1,1,300))
    out = []
    out.append(np.random.randn(1,73,512))

    # f = open('datafile','wb')
    # f.write(pickle.dumps(ins))
    # f.close()
    
    # print model.summary()
    # model.fit(ins,out,epochs = 1, verbose=1)
    preds = model.predict(ins,batch_size=1,verbose=1)

    print preds.shape
    # final_vid_feat = Dense(300)(vid_feat)     #Donot downsample....It will lose information


if __name__ == '__main__':
    nframes = 50  # Temporary
    input_shape2d = (240,240,3)
    input_shape3d = (10,240,240,3)    # Temporary
    nclips = nframes/10
    max_cap_len = 72
    model = create_model(nframes,nclips,max_cap_len,input_shape2d,input_shape3d)
    pred(model)
    # print model.predict()