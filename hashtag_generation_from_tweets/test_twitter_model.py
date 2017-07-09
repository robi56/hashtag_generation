
# author - Richard Liao
# Dec 26 2016
import numpy as np
import pandas as pd
import cPickle
from collections import defaultdict
import re

from bs4 import BeautifulSoup

import sys
import os

os.environ['KERAS_BACKEND']='tensorflow'

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout
from keras.models import Model

import codecs
import json
from gensim.models.word2vec import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np
from collections import Counter, defaultdict
import random
import unidecode
from sklearn.preprocessing import MultiLabelBinarizer
from keras.models import model_from_yaml


MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2


def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()







from datautils import loadTwitterData

[contents,labels]=loadTwitterData('twitterdata.json')
#[contents, labels, urls]=loadYouTubeData('youtube_data_sample.json')


#dictionary = buildingDictionaryForOutputLabels(labels)

#print contents
#print labels
'''
dictfile = open('dictionary.txt', 'w')
for label in dictionary:
  dictfile.write("%s\n" % label)

dictfile.close()
'''
dictfile =open('dictionary.txt', 'r')
dict=[]
with open('dictionary.txt','r') as file:
    dict=file.readlines()

dictionary=[label.replace('\n','') for label in dict]

#print 'dict contains..'
#print dict


tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(contents)
sequences = tokenizer.texts_to_sequences(contents)

print "Total Tweets:"+str(len(contents))
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
print "Total Instance: "+str(len(data))

# load YAML and create model
yaml_file = open('model_twitter_sample_2500_epoch.yaml', 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
loaded_model = model_from_yaml(loaded_model_yaml)
# load weights into new model
loaded_model.load_weights("model_twitter_2500_epoch.h5")
print("Loaded model from disk")
loaded_model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

prediction = loaded_model.predict(data)
#print(prediction)

import numpy as np
model_output_sort_reverse=np.argsort(prediction)

print "Total Label: "+str(len(model_output_sort_reverse[0]))
model_output_sort_by_value=-np.sort(-prediction)
model_output_sort = ([out[::-1] for out in model_output_sort_reverse])
print "..............."

'''
for index in range(0,len(model_output_sort)):
    print model_output_sort[index]
    print " : "
    print model_output_sort_by_value[index]

print "......................"
'''
predicated_output=[]
prediction_output_score=[]
for index in range(0,len(model_output_sort)):
    count =0
    current_output=[]
    current_output_score=[]
 #   print "text:"+str(contents[index])
  #  print "original:"+str(labels[index])
    tweet_text = contents[index]
    selectedIndex=0
    for sub in model_output_sort[index] :
      # print dictionary[sub]
       if dictionary[sub] in tweet_text:
          current_output.append(dictionary[sub])
          count=count+1
          current_output_score.append(model_output_sort_by_value[index][selectedIndex])
       elif dictionary[sub].lower() in tweet_text.lower():
            current_output.append(dictionary[sub])
            current_output_score.append(model_output_sort_by_value[index][selectedIndex])
            count=count+1


       selectedIndex=selectedIndex+1
       if count>2:
           break
      # print '\t'


    predicated_output.append(current_output)
    prediction_output_score.append(current_output_score)
   # print "..............."
    index=index+1

#calculate accuracy
acc = 0.0

for i in range(0,len(labels)):
    truelabelset = labels[i]   # true keywords for i_th element
    candidatelabelset=predicated_output[i] #predicated keywords for i_th element
    candidatelabelsetscore= prediction_output_score[i] #output scores
    match=0
    suggest_hastag=[]
    suggest_hastag_score=[]
    for index in range(0,len(candidatelabelset)):
        label=candidatelabelset[index]
        if label in truelabelset:
            match = match+1
            suggest_hastag.append(label)
            suggest_hastag_score.append(candidatelabelsetscore[index])
            if len(suggest_hastag)==len(candidatelabelset):
                break
    print "text:"+str(contents[i])
    print "original:"+str(labels[i])
    print "accepted_hashtag:"+str(suggest_hastag)
    print "accepted hashtag score"+str(suggest_hastag_score)

    acc = acc + match/float(len(truelabelset)) # accuracy = match/truelabelset


avg_acc = acc/len(labels)
print "Accuracy: "+str(avg_acc)

