import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
import string
from pickle import dump
text=open("flick_text/Flickr8k.token.txt",'r')
txt=text.read()

high=dict()
i=[]
p=1
count={}
voc_no={}
voc_name={}
#declaring a puntuation table
tb=str.maketrans('','',string.punctuation)
for line in txt.split('\n'):
    tkns=line.split()
    #if(tkns == dict()):
    img_id=tkns[0]
    cap=tkns[1:]
    #converting into lower alphabets
    cap=[word.lower() for word in cap]
    #removing puntuations
    cap=[word.translate(tb) for word in cap]
    #removing word 'a'
    cap=[word for word in cap if len(word)>1]
    #removing numbers
    cap=[word for word in cap if word.isalpha()]
    for word in cap:
        i.append(word)
    cap='startseq '+' '.join(cap)+' endseq'
    img_id=img_id.split('.')[0]
    if img_id not in high:
        high[img_id]=list()
    high[img_id].append(cap)
    
#converting into  a set of words #8763
all_desc = set()
for key in high.keys():
    [all_desc.update(d.split()) for d in high[key]]
#getting count of words
for k in i:
    count[k]=count.get(k,0)+1
    if count[k]==10:
        voc_no[k]=voc_no.get(k,0)+p
        voc_name[p]=k
        p=p+1
voc_no['startseq']=1948
voc_name[1948]='startseq'
voc_no['endseq']=1949
voc_name[1949]='endseq'
#we got most repeatative words    

#now save the dictionary as pickle type

with open("pickle/voc_num.pkl",'wb') as voc_num:
    dump(voc_no,voc_num)

with open("pickle/voc_name.pkl",'wb') as voc_nam:
    dump(voc_name,voc_nam)
with open("pickle/total.pkl",'wb') as tot:
    dump(high,tot)
