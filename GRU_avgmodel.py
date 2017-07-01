import numpy as np
import pandas as pd
import cPickle
#import cv2
from collections import defaultdict
import re

from bs4 import BeautifulSoup

import sys
import os
import json

os.environ['KERAS_BACKEND']='theano'

from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers.pooling import GlobalAveragePooling1D
from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout, LSTM, GRU, Bidirectional, TimeDistributed
from keras.models import Model, model_from_json

from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializers

from keras import regularizers, constraints

from glove import Corpus, Glove

from nltk import tokenize
import ast
from keras.preprocessing import sequence

from keras import callbacks

import ast
import sys
import cPickle

EMBEDDING_DIM = 100
WORDGRU = EMBEDDING_DIM/2
DROPOUTPER = 0.3



def GRUf(MAX_NB_WORDS, MAX_WORDS, MAX_SENTS, EMBEDDING_DIM, WORDGRU, embedding_matrix, DROPOUTPER):
    wordInputs = Input(shape=(MAX_SENTS+1, MAX_WORDS), name="wordInputs", dtype='float32')

    wordInp = Flatten()(wordInputs)

    wordEmbedding = Embedding(MAX_NB_WORDS, EMBEDDING_DIM, weights=[embedding_matrix], mask_zero=False, trainable=True, name='wordEmbedding')(wordInp) 

    hij = Bidirectional(GRU(WORDGRU, return_sequences=True), name='gru1')(wordEmbedding)

    head = GlobalAveragePooling1D()(hij) 
    
    v6 = Dense(1, activation="sigmoid", name="dense")(head)

    model = Model(inputs=[wordInputs] , outputs=[v6])
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    return model

