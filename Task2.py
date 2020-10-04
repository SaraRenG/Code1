#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 19:58:56 2020
Task2: Manglish tweet/comment classification (Method 1 : keras embedding)
@author: sara
"""
#Read from excel
import xlrd 
from matplotlib import pyplot as plt
import numpy as np
np.random.seed(1337) #for result reproducibility

##For preprocessing
import preprocessor as p
import PreProcessTweets as ppt
from sklearn.model_selection import train_test_split
import sklearn.preprocessing as preprocessing

from keras.models import Sequential
from keras.layers import Dense,Embedding, LSTM, Conv1D,MaxPooling1D

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping, ModelCheckpoint

idl = []
comm = []
label = []
datadict = {}
dataraw = {}
dict_mal = {}


##preprocessing- remove @USER, repeated letters
def preprocess(dlist): 
    cmt_list = []
    word_list = []
    pt = ppt.PreProcessTweets()
    for i in dlist.keys():
        print (i)
        cleaned_data = p.clean(str(dlist[i][0]))
        cleaned_data = pt.processTweets(str(cleaned_data))
        datadict[i] = (cleaned_data,dlist[i][1])
        dl =" "
        cmt_list.append(dl.join(datadict[i][0]))
        for elem in cleaned_data:
            word_list.append(elem)    
    return word_list,cmt_list

#Open sheet location
def opensheet(location,sheet_name):
    wb = xlrd.open_workbook(location)
    sheet = wb.sheet_by_name(sheet_name)
    print (sheet.cell_value(0,0))
    print ('***********')
    return sheet


## Read data from file
loc = ("Malayalam_offensive_data_Training-YT.xlsx") 
dataloc = ("malayalam_hasoc_tanglish_test_without_labels_copy.xlsx")

#Training data
sheet = opensheet(loc,'Sheet1')
for i in range(1,sheet.nrows):
    idl.append(sheet.cell_value(i, 0))
    ##comment transliterated
    cmmt = sheet.cell_value(i,1)
    print (type(cmmt))
#    if type(cmmt) !=int:
#        transcmmt = transliterate(cmmt, xsanscript.ITRANS, xsanscript.MALAYALAM)
#    else:
#        transcmmt ='EMPTY'
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



wordlist,cmtlist = preprocess(dataraw)
test_wordlist,test_cmtlist = preprocess(test_data_dict)


vocabulary_size = 5000 #22427
tokenizer = Tokenizer(num_words= vocabulary_size)
tokenizer.fit_on_texts(cmtlist)
sequences = tokenizer.texts_to_sequences(cmtlist)
data = pad_sequences(sequences, maxlen=50,padding='pre')

#for testdata
tokenizer.fit_on_texts(test_cmtlist)
test_sequences = tokenizer.texts_to_sequences(test_cmtlist)
testdata = pad_sequences(test_sequences,maxlen = 50, padding='pre')


#train_x, test_x, train_y, test_y = train_test_split(data, label)

# label encode the target variable 
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(label)
#test_y = encoder.fit_transform(data)


## Network architecture
model = Sequential()
model.add(Embedding(vocabulary_size, 50, input_length=50)) #57
model.add(LSTM(50, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
callbacks = [EarlyStopping(monitor='val_loss', patience=2),
             ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]

## Fit the model
history = model.fit(data, train_y,callbacks = callbacks,validation_split=0.2, epochs=5) # 

########################################################
from keras.models import load_model
saved_model = load_model('best_model.h5')
train_acc = saved_model.evaluate(data, train_y, verbose=0)
test_y = saved_model.predict(testdata)
test_y_classes = saved_model.predict_classes(testdata)


 


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
