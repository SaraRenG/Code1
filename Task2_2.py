#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 20:23:27 2020
Task2: Manglish tweet/comment classification (Method 2 : Doc2Vec embedding)
@author: sara
"""

#Read from excel
import xlrd 

import numpy as np
np.random.seed(1337) #for result reproducibility


##For preprocessing
import preprocessor as p
import PreProcessTweets as ppt
import gensim
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense,LSTM,Input
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint


idl = []
comm = []
label = []
datadict = {}
dataraw = {}
dict_mal = {}

cmtlist = []
wordlist = []


##preprocessing- remove @USER, repeated letters
def preprocess(dlist): 
    cmt_list = []
    word_list = []
    processd_dict = {}
    pt = ppt.PreProcessTweets()
    for i in dlist.keys():
        print (i)
        cleaned_data = p.clean(str(dlist[i][0]))
        cleaned_data = pt.processTweets(str(cleaned_data))
        datadict[i] = (cleaned_data,dlist[i][1])
        dl =" "
        cmtlist.append(dl.join(datadict[i][0]))
        processd_dict[i]= dl.join(datadict[i][0])
        for elem in cleaned_data:
            wordlist.append(elem)
    return word_list,cmt_list,processd_dict

#Open sheet location
def opensheet(location,sheet_name):
    wb = xlrd.open_workbook(location)
    sheet = wb.sheet_by_name(sheet_name)
    print (sheet.cell_value(0,0))
    print ('***********')
    return sheet

## Tag the sentences/documents
def tag(clist, tokens_only= False):     
    for i, line in enumerate(clist):
        tokens = gensim.utils.simple_preprocess(line)
        if tokens_only:
            yield tokens
        else:
            yield gensim.models.doc2vec.TaggedDocument(tokens, [i])


## Read data from file
loc = ("Malayalam_offensive_data_Training-YT.xlsx") 
dataloc = ("malayalam_hasoc_tanglish_test_without_labels_copy.xlsx")

sheet = opensheet(loc,'Sheet1')
print(sheet.cell_value(0, 0) )
for i in range(1,sheet.nrows):
    idl.append(sheet.cell_value(i, 0))
    cmmt = sheet.cell_value(i,1)
    comm.append(cmmt)    
    label.append(sheet.cell_value(i,2))    
    dataraw [sheet.cell_value(i,0)] = (cmmt,sheet.cell_value(i,2))
    
#testdata
test_data = []
test_data_dict ={}
sheet = opensheet(dataloc,'malayalam_hasoc_tanglish_test_without_labels')
for i in range(0,sheet.nrows):
    test_data.append(sheet.cell_value(i, 1))
    test_data_dict [sheet.cell_value(i,0)] = (sheet.cell_value(i,1),'NULL')
    
processddict ={}
test_processddict = {}
wordlist,cmtlist,processddict = preprocess(dataraw)
test_wordlist,test_cmtlist,test_processddict = preprocess(test_data_dict)


##tagged sentences for DOC2VEC model  training     
train_corpus = list(tag(cmtlist))
test_corpus = list(tag(test_cmtlist))
corpus = train_corpus+ test_corpus


d2vmodel = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=10,workers=1)
d2vmodel.build_vocab(corpus)
d2vmodel.train(corpus, total_examples=d2vmodel.corpus_count, epochs=d2vmodel.epochs)      

input_X={}
Xlist=[]
testlist=[]
ppdict = list(tag(processddict.values()))
for doc_id in range(len(ppdict)):
    inferred_vector = d2vmodel.infer_vector(list(ppdict[doc_id].words))
    Xlist.append(inferred_vector)
    input_X[doc_id] = inferred_vector
 
test_X={}
test_ppdict = list(tag(test_processddict.values()))
for doc_id in range(len(test_ppdict)):
    test_inferred_vector = d2vmodel.infer_vector(list(test_ppdict[doc_id].words))
    testlist.append(test_inferred_vector)
    test_X[doc_id] = test_inferred_vector
    
    
encoder = LabelEncoder()
encoder.fit(label)
encoded_Y = encoder.transform(label)

Xlist_s = np.stack(Xlist,axis=0)
testlist_s = np.stack(testlist,axis=0)

trainX = Xlist_s[0:3599]
trainY = encoded_Y[0:3599]
valX = Xlist_s[3600:4000]
valY = encoded_Y[3600:4000]

model = Sequential()
model.add(Dense(50, activation='relu',name="layer1"))
model.add(Dense(25,activation='relu',name="layer2"))
model.add(Dense(10,activation='relu',name="layer3"))
model.add(Dense(1,activation='sigmoid',name="layer4"))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#callbacks = [EarlyStopping(monitor='val_acc', patience=5),
#             ModelCheckpoint(filepath='best_model_task2_2.h5', monitor='val_acc', save_best_only=True)]
history = model.fit(trainX, trainY,validation_data=(valX,valY), epochs=50) # callbacks = callbacks,
y_pred_val= model.predict(testlist_s)
y_pred = model.predict_classes(testlist_s)

def plot_history(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
plot_history(history)