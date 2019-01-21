from keras.models import load_model
import cv2
import numpy as np
from PIL import Image
from pickle import load,dump
from matplotlib import pyplot as plt
from keras.preprocessing.sequence import pad_sequences
from keras.applications.inception_v3 import preprocess_input
model=load_model('models/inception.h5')
kl=[]
def encode(img_path):
    img=Image.open(img_path)
    #img.show()
    img=np.array(img)
    
    img=img[0:,0:,0:3]
    img=cv2.resize(img,(299,299))
    im=np.expand_dims(img,axis=0)
    img=preprocess_input(im)
    img=model.predict(img)
    #img=np.reshape(img,img.shape[1])
    #print('1')
    return img
img_path='test/95728660_d47de66544 - Copy.jpg'
img=Image.open(img_path)
    #img.show()
img=np.array(img)
en=encode(img_path)
with open("pickle/voc_name.pkl",'rb') as k:
  to_word=load(k)
with open("pickle/voc_num.pkl",'rb') as k:
  to_in=load(k)
mod=load_model('models/best.h5')
photo=en
in_text = 'startseq'
prop=1
b_w=0
k1=[]
k2=[]
def beam(in_text,prop,b_w,photo):
    if b_w==6:
        k1.append(in_text)
        k2.append(prop)
        return 0
    else:    
        sequence = [to_in[w] for w in in_text.split() if w in to_in]
        sequence = pad_sequences([sequence], maxlen=34)
        kl=sequence
        photo=np.array(photo)
        sequence=np.array(sequence)
        y = mod.predict([photo,sequence], verbose=0)
        y=y.reshape(1951)
        y=list(y)
        n1 = np.argmax(y)
        #print(y.shape)
        prop1=y[n1]
        word1 = to_word[n1]
        del(y[n1])
        n2 = np.argmax(y)
        prop2=y[n2]
        word2 = to_word[n2]
        del(y[n2])
        n3 = np.argmax(y)
        prop3=y[n3]
        word3 = to_word[n3]
        return beam(in_text+' '+word1,prop1,b_w+1,photo),beam(in_text+' '+word2,prop2,b_w+1,photo),beam(in_text+' '+word3,prop3,b_w+1,photo)
for i in range(4):
    
    t=beam(in_text,prop,b_w,photo)
    t1=np.argmax(k2)
    in_text=k1[t1]
    if 'endseq' in in_text:
        break
    k1=[]
    k2=[]
k=''    
t=in_text.split(' ')
for i in t:
    if i=='endseq':
        break
    else:
      if i!='startseq':  
        k=k+' '+i
print(k)        
max_length=34
def greedySearch(photo):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = [to_in[w] for w in in_text.split() if w in to_in]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = mod.predict([photo,sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = to_word[yhat]
        in_text += ' ' + word
        if word == 'endseq':
            break
    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final
l=greedySearch(photo)
print(l)
plt.xlabel(l)
plt.imshow(img)
plt.show()
