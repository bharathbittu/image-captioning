from pickle import load,dump
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Conv2D,Dense,Embedding,Input,add,Flatten,LSTM,Dropout,MaxPool2D,TimeDistributed
from keras.models import Sequential,Model,load_model
from keras.applications.inception_v3 import preprocess_input
from sklearn.model_selection import train_test_split


glove_dir = ''
embeddings_index = {} # empty dictionary
f = open('glove.6B.200d.txt', encoding="utf-8")

for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

embedding_dim = 200

# Get 200-dim dense vector for each of the 10000 words in out vocabulary
embedding_matrix = np.zeros((1951, embedding_dim))

for word, i in to_in.items():
    #if i < max_words:
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # Words not found in the embedding index will be all zeros
        embedding_matrix[i] = embedding_vector
with open("embed.pkl",'wb') as k:
    dump(embedding_matrix,k)
