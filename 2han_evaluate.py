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

from keras import regularizers, constraints
from keras.models import load_model
import matplotlib.pyplot as plt

import ast
import sys
import cPickle

class AttentionLayer(Layer):
	def __init__(self, **kwargs):
		self.supports_masking = True
		super(AttentionLayer,self).__init__(**kwargs)

	def build(self, input_shape):
		self.W = self.add_weight(shape=(input_shape[-1], input_shape[-1], ), name='{}_W'.format(self.name), initializer = 'glorot_uniform', trainable=True)
		self.bw = self.add_weight(shape=(input_shape[-1], ), name='{}_b'.format(self.name), initializer = 'zero', trainable=True)
		self.uw = self.add_weight(shape=(input_shape[-1], ), name='{}_u'.format(self.name), initializer = 'glorot_uniform', trainable=True)
		self.trainable_weights = [self.W, self.bw, self.uw]
		
		super(AttentionLayer,self).build(input_shape)
	
	def compute_mask(self, input, mask):
        	return 2*[None]

	def call(self, x, mask=None):
		uit = K.dot(x, self.W)
		
		uit += self.bw
		uit = K.tanh(uit)
		ait = K.dot(uit, self.uw)
		a = K.exp(ait)

		if mask is not None:
			a *= K.cast(mask, K.floatx())

		a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
		a = K.expand_dims(a)
			
		weighted_input = x * a
		
		
		ssi = K.sum(weighted_input, axis=1)
		return [a, ssi]

	def get_output_shape_for(self, input_shape):
		return  [(input_shape[0],input_shape[1]), (input_shape[0], input_shape[-1])]

	def compute_output_shape(self, input_shape):
		return [(input_shape[0],input_shape[1]), (input_shape[0], input_shape[-1])]
		
#load the model - assume we ran the code and saved the final model as 3HAN.h5
model = load_model("SavedModels/2checkpoint-10-0.9681.hdf5", custom_objects={"AttentionLayer":AttentionLayer})
#model1 = load_model("wordEncoder.h5", custom_objects={"AttentionLayer":AttentionLayer})

test_articles = cPickle.load(open('Dataset/test_articles_seq.p', 'rb')).tolist()
test_headlines = cPickle.load(open('Dataset/test_headlines_seq.p', 'rb')).tolist()
test_y = cPickle.load(open('Dataset/test_y.p', 'rb')).tolist()

'''
test_articles = test_articles[:10]
test_headlines = test_headlines[:10]
test_y = test_y[:10]
'''

for i in range(0, len(test_headlines)):
    test_articles[i].insert(0, test_headlines[i])

test_articles = np.array(test_articles)
test_headlines = np.array(test_headlines)
test_y = np.array(test_y)

print ("test")
print (test_articles.shape)
print (test_headlines.shape)

print model.evaluate(test_articles, test_y, batch_size=32)
