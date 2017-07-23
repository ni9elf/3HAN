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

from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, concatenate, Dropout, LSTM, GRU, Bidirectional, TimeDistributed, Masking
from keras.models import Model, Sequential
from keras.layers.core import Activation, Reshape

from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializers

from keras import regularizers, constraints, optimizers

class AttentionLayer(Layer):
	def __init__(self, **kwargs):
		self.supports_masking = True
		super(AttentionLayer,self).__init__(**kwargs)

	def build(self, input_shape):
		
		#print '\nhi in build attention'
		#print input_shape
	
		self.W = self.add_weight(shape=(input_shape[-1], input_shape[-1], ), name='{}_W'.format(self.name), initializer = 'glorot_uniform', trainable=True)
		self.bw = self.add_weight(shape=(input_shape[-1], ), name='{}_b'.format(self.name), initializer = 'zero', trainable=True)
		self.uw = self.add_weight(shape=(input_shape[-1], ), name='{}_u'.format(self.name), initializer = 'glorot_uniform', trainable=True)
		self.trainable_weights = [self.W, self.bw, self.uw]
		
		#print "\nweights in attention"
		#print self.W._keras_shape
		#print self.bw._keras_shape
		#print self.uw._keras_shape
		super(AttentionLayer,self).build(input_shape)
	
	def compute_mask(self, input, mask):
        	return 2*[None]

	def call(self, x, mask=None):
	
		#print '\nhi in attention'
		#print x._keras_shape
		
		uit = K.dot(x, self.W)
		
		#print '\nuit'
		#print uit._keras_shape
		
		uit += self.bw
		uit = K.tanh(uit)

		ait = K.dot(uit, self.uw)
		a = K.exp(ait)

		# apply mask after the exp. will be re-normalized next
		#print mask
		if mask is not None:
			a *= K.cast(mask, K.floatx())

		a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
		a = K.expand_dims(a)

		#print "in att ", K.shape(a)
			
		weighted_input = x * a
		
		#print weighted_input	
		
		ssi = K.sum(weighted_input, axis=1)
		#print "type ", type(ssi)	
		#print "in att si ", theano.tensor.shape(ssi)
		#1111print "hello"
		return [a, ssi]

	def get_output_shape_for(self, input_shape):
		#print '\nhiiiiiiiiiiiiii'
		return  [(input_shape[0],input_shape[1]), (input_shape[0], input_shape[-1])]

	def compute_output_shape(self, input_shape):
		#print '\nyooooooooooooooooooooo'
		#print input_shape
		return [(input_shape[0],input_shape[1]), (input_shape[0], input_shape[-1])]


def HAN(MAX_NB_WORDS, MAX_WORDS, MAX_SENTS, EMBEDDING_DIM, WORDGRU, embedding_matrix, DROPOUTPER):
	
    wordInputs = Input(shape=(MAX_WORDS,), name="wordInputs", dtype='float32')

    wordEmbedding = Embedding(MAX_NB_WORDS, EMBEDDING_DIM, weights=[embedding_matrix], mask_zero=True, trainable=True, name='wordEmbedding')(wordInputs) 

    hij = Bidirectional(GRU(WORDGRU, return_sequences=True), name='gru1')(wordEmbedding)

    wordDrop = Dropout(DROPOUTPER, name='wordDrop')(hij)
    
    alpha_its, Si = AttentionLayer(name='att1')(wordDrop)
    
    
    wordEncoder = Model(wordInputs, Si)
    # -----------------------------------------------------------------------------------------------

    docInputs = Input(shape=(None, MAX_WORDS), name='docInputs' ,dtype='float32')

    sentenceMasking = Masking(mask_value=0.0, name='sentenceMasking')(docInputs)

    sentEncoding = TimeDistributed(wordEncoder, name='sentEncoding')(sentenceMasking) 

    hi = Bidirectional(GRU(WORDGRU, return_sequences=True), merge_mode='concat', name='gru2')(sentEncoding)
    
    sentDrop = Dropout(DROPOUTPER, name='sentDrop')(hi)
   
    alpha_s, Vb = AttentionLayer(name='att2')(sentDrop)
    
    Vb = Reshape((1, Vb._keras_shape[1]))(Vb)

    #-----------------------------------------------------------------------------------------------

    headlineInput = Input(shape=(MAX_WORDS,), name='headlineInput',dtype='float32')

    headlineEmb = Embedding(MAX_NB_WORDS, EMBEDDING_DIM, mask_zero=True, name='headlineEmb')(headlineInput)

    Vb = Masking(mask_value=0.0, name='Vb')(Vb)		
    
    headlineBodyEmb = concatenate([headlineEmb, Vb], axis=1, name='headlineBodyEmb')

    h3 = Bidirectional(GRU(WORDGRU, return_sequences=True), merge_mode='concat', name='gru3')(headlineBodyEmb)
    
    headDrop = Dropout(DROPOUTPER, name='3Drop')(h3)

    a3, Vn = AttentionLayer(name='att3')(headDrop)
    

    v6 = Dense(1, activation="sigmoid", kernel_initializer = 'he_normal', name="dense")(Vn)
    model = Model(inputs=[docInputs, headlineInput] , outputs=[v6])

    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model, wordEncoder      

	
if __name__ == "__main__":
	model = HAN(20000,10,5,50)					
