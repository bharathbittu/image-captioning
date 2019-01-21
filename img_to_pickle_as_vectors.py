import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Conv2D,Dense,Embedding,Input,add,Flatten,LSTM,Dropout,MaxPool2D,TimeDistributed
from keras.models import Sequential,Model,load_model
from keras.applications.inception_v3 import preprocess_input
from sklearn.model_selection import train_test_split
import glob
from pickle import dump,load
from PIL import Image
import cv2
mod=load_model('models/inception.h5')
def encode(img_path):
    img=Image.open(img_path)
    img=np.array(img)
    img=cv2.resize(img,(299,299))
    im=np.expand_dims(img,axis=0)
    img=preprocess_input(im)
    img=mod.predict(img)
    img=np.reshape(img,img.shape[1])
    #print('1')
    return img
images='Flicker8k_Dataset/' #download it 
img=glob.glob(images+'*.jpg')
train=[]
test=[]
train_names=list(open('imagerec/Flickr_8k.trainImages.txt','r').read().strip().split('\n'))
for i in img:
    if i[len(images):] in train_names:
        train.append(i)
test_names=list(open('imagerec/Flickr_8k.testImages.txt','r').read().strip().split('\n'))
        
for i in img:
    if i[len(images):] in test_names:
        test.append(i)
#encode
en_test={}
en_train={}
for i in train:
    en_train[i[len(images):]]=encode(i)

with open("pickle/en_train.pkl",'wb') as en_trainr:
    dump(en_train,en_trainr)
