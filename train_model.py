from pickle import load,dump
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Conv2D,Dense,Embedding,Input,add,Flatten,LSTM,Dropout,MaxPool2D,TimeDistributed
from keras.models import Sequential,Model,load_model
from keras.applications.inception_v3 import preprocess_input
from sklearn.model_selection import train_test_split
with open("pickle/voc_name.pkl",'rb') as k:
  to_word=load(k)
with open("pickle/voc_num.pkl",'rb') as k:
  to_in=load(k)
with open("pickle/total.pkl",'rb') as k:
  descript=load(k)          
with open("pickle/en_train.pkl",'rb') as k:
  train_img=load(k)
with open("pickle/embed.pkl",'rb') as k:
  embedding_matrix=load(k)
embedding_matrix=embedding_matrix.reshape(1951,200)
k=[]
def data_generator(descript,train_img,to_in, num_photos_per_batch):
  X2=list()
  y=list()
  X1=list()
  n=0

    # loop for ever over images
  while 1:
    
    for des in descript.items():
      key=des[0]
      des=des[1]
      if key+'.jpg' in train_img:
        img_data=train_img[key+'.jpg']
        p=0
      
      else:
        p=1
      if p==0:  
        n=n+1
        for d in des:
          seq=[to_in[word] for word in d.split() if word in to_in]
          for i in range(1,len(seq)):
            in1=seq[:i]  
            in1=pad_sequences([in1], maxlen=34)[0]
            out=to_categorical(seq[i], num_classes=1951)
            
            X1.append(img_data)
            X2.append(in1)
            y.append(out) 
          if n==num_photos_per_batch:
              #print(y)
              #print(np.array(X1).shape,np.array(X2).shape,np.array(y).shape)
              yield [[np.array(X1), np.array(X2)], np.array(y)]
              X1, X2, y = list(), list(), list()
              n=0
inputs1=Input(shape=(2048,))
p1=Dropout(0.5)(inputs1)
p2=Dense(256,activation='relu')(p1)

in2=Input(shape=(34,))
b1=Embedding(1951,200,mask_zero=True)(in2)
b2=Dropout(0.5)(b1)
b3=LSTM(256)(b2)

d1=add([p2,b3])
d2=Dense(256,activation='relu')(d1)
outputs=Dense(1951,activation='softmax')(d2)
model=Model(inputs=[inputs1,in2],outputs=outputs)
model.layers[2].set_weights([embedding_matrix])
#model.layers[2].trainable = False

model.compile(loss='categorical_crossentropy', optimizer='adam')


epochs = 10
number_pics_per_bath = 3
steps = len(train_img)//number_pics_per_bath

for i in range(epochs):

    generator = data_generator(descript,train_img,to_in, number_pics_per_bath)
    k=generator
    model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)
    model.save('model' + str(i) + '.h5')
