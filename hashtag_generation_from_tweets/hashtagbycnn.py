
# author - Richard Liao
# Dec 26 2016
#Modified by Rabindra Nath Nandi for Hasttag generation
import re
import os
os.environ['KERAS_BACKEND']='tensorflow'
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout
from keras.models import Model
from tensorflow.contrib.keras import layers
import codecs
import json

import numpy as np
import unidecode
from sklearn.preprocessing import LabelBinarizer



MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2

#MODEL BASIC PARAMETER
CONV_NUM_FILTER=128
CONV_FILTER_LENGTH=5
BATCH_SIZE=128
POOL_LENGTH=5
GLOBAL_POOL_LENGTH=35
DENSE_LAYER_OUT_DIM=128


def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()




#Experiment (Considering single instance for per category)

from datautils import loadTwitterData
from datautils import buildingDictionaryForOutputLabels
from datautils import numericalRepresentationOutputLabels

[contents, labels] = loadTwitterData('twitterdata.json')
dictionary = buildingDictionaryForOutputLabels(labels)

print 'Content Length'
print len(contents)
print 'Dictionary Length'
print len(dictionary)
numerical_labels=numericalRepresentationOutputLabels(dictionary,labels)
print 'Numerical Length'
print numerical_labels

testData=[]
newXtrain=[]
newYtrain=[]
count=0
for count in range(0,len(numerical_labels)):
    element=numerical_labels[count]
    for cat in element:
        newXtrain.append(contents[count])
        newYtrain.append(cat)

newYtrain=np.asarray(newYtrain)
lb = LabelBinarizer()
newYtrain=lb.fit_transform(newYtrain)

tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(newXtrain)
sequences = tokenizer.texts_to_sequences(newXtrain)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
testData=data
#labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', newYtrain.shape)


indices = np.arange(data.shape[0])
np.random.shuffle(indices)
indices=np.asarray(indices)
print indices
new_data = data[indices]
new_labels = newYtrain[indices]
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = new_data[:-nb_validation_samples]
y_train = new_labels[:-nb_validation_samples]
x_val = new_data[-nb_validation_samples:]
y_val = new_labels[-nb_validation_samples:]

print('Number of Instances in traing and validation set ')
print y_train.sum(axis=0)
print y_val.sum(axis=0)



#Word Embedding part
GLOVE_DIR = "glob/"
embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Total %s word vectors in Glove 6B 100d.' % len(embeddings_index))

embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=True)


# Model Part

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
l_cov1= Conv1D(CONV_NUM_FILTER, CONV_FILTER_LENGTH, activation='relu')(embedded_sequences)
l_pool1 = MaxPooling1D(pool_length=POOL_LENGTH)(l_cov1)
l_cov2 = Conv1D(CONV_NUM_FILTER, CONV_FILTER_LENGTH, activation='relu')(l_pool1)
l_pool2 = MaxPooling1D(pool_length=POOL_LENGTH)(l_cov2)
l_cov3 = Conv1D(CONV_NUM_FILTER, CONV_FILTER_LENGTH, activation='relu')(l_pool2)
l_pool3 = MaxPooling1D(pool_length=GLOBAL_POOL_LENGTH)(l_cov3)  # global max pooling
l_flat = Flatten()(l_pool3)
l_dense = Dense(DENSE_LAYER_OUT_DIM, activation='relu')(l_flat)
preds = Dense(len(dictionary), activation='softmax')(l_dense)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

print("model fitting - simplified convolutional neural network")
model.summary()

#model train
model.fit(new_data, new_labels, validation_data=(x_val, y_val),
          nb_epoch=10, batch_size=128)

# serialize model to YAML
model_yaml = model.to_yaml()
with open("model_twitter_sample_2500_epoch.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)
# serialize weights to HDF5
model.save_weights("model_twitter_2500_epoch.h5")
print("Saved model to disk")


