import numpy as np
from collections import defaultdict
import re

import sys
import os

os.environ['KERAS_BACKEND']='theano'

import theano
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.models import Model, model_from_json

from keras.layers.pooling import GlobalAveragePooling1D
from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, concatenate, Dropout, LSTM, GRU, Bidirectional, TimeDistributed, Masking
from keras.models import Model, Sequential
from keras.layers.core import Activation, Reshape

from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializers

from keras import regularizers, constraints, optimizers



def fhan3_avg(MAX_NB_WORDS, MAX_WORDS, MAX_SENTS, EMBEDDING_DIM, WORDGRU, embedding_matrix, DROPOUTPER):
	
    wordInputs = Input(shape=(MAX_WORDS,), name="wordInputs", dtype='float32')

    wordEmbedding = Embedding(MAX_NB_WORDS, EMBEDDING_DIM, weights=[embedding_matrix], mask_zero=False, trainable=True, name='wordEmbedding')(wordInputs) 

    hij = Bidirectional(GRU(WORDGRU, return_sequences=True), name='gru1')(wordEmbedding)

    #alpha_its, Si = AttentionLayer(name='att1')(hij)
    wordDrop = Dropout(DROPOUTPER, name='wordDrop')(hij)
    
    word_pool = GlobalAveragePooling1D()(wordDrop)      
    
    wordEncoder = Model(wordInputs, word_pool)
    
    # -----------------------------------------------------------------------------------------------

    docInputs = Input(shape=(None, MAX_WORDS), name='docInputs' ,dtype='float32')

    #sentenceMasking = Masking(mask_value=0.0, name='sentenceMasking')(docInputs)

    sentEncoding = TimeDistributed(wordEncoder, name='sentEncoding')(docInputs) 

    hi = Bidirectional(GRU(WORDGRU, return_sequences=True), merge_mode='concat', name='gru2')(sentEncoding)
   
    #alpha_s, Vb = AttentionLayer(name='att2')(hi)
    sentDrop = Dropout(DROPOUTPER, name='sentDrop')(hi)
    
    sent_pool = GlobalAveragePooling1D()(sentDrop)         

    Vb = Reshape((1, sent_pool._keras_shape[1]))(sent_pool)

    #-----------------------------------------------------------------------------------------------

    headlineInput = Input(shape=(MAX_WORDS,), name='headlineInput',dtype='float32')

    headlineEmb = Embedding(MAX_NB_WORDS, EMBEDDING_DIM, mask_zero=False, name='headlineEmb')(headlineInput)

    #Vb = Masking(mask_value=0.0, name='Vb')(Vb)		
    headlineBodyEmb = concatenate([headlineEmb, Vb], axis=1, name='headlineBodyEmb')

    h3 = Bidirectional(GRU(WORDGRU, return_sequences=True), merge_mode='concat', name='gru3')(headlineBodyEmb)

    #a3, Vn = AttentionLayer(name='att3')(h3)
    
    headDrop = Dropout(DROPOUTPER, name='3Drop')(h3)
    
    head_pool = GlobalAveragePooling1D()(headDrop) 

    v6 = Dense(1, activation="sigmoid", kernel_initializer = 'he_normal', name="dense")(head_pool)
    model = Model(inputs=[docInputs, headlineInput] , outputs=[v6])

    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model, wordEncoder      

	
if __name__ == "__main__":
	model = HAN(20000,10,5,50)					 
