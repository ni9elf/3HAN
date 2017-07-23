import csv
import sys
import re
import os
import numpy as np

import _pickle as cPickle

from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import RandomForestClassifier


from sklearn.model_selection import StratifiedKFold

from nltk.corpus import stopwords

from sklearn import metrics

#--------------------------------------------------------------------------------------------------------------------------------

print ('loading files...')
train_articles1 = cPickle.load(open('Dataset/train_articles1.p', 'rb')).tolist()
train_articles2 = cPickle.load(open('Dataset/train_articles2.p', 'rb')).tolist()
train_articles3 = cPickle.load(open('Dataset/train_articles3.p', 'rb')).tolist()
train_articles4 = cPickle.load(open('Dataset/train_articles4.p', 'rb')).tolist()


train_headlines = cPickle.load(open('Dataset/train_headlines.p', 'rb')).tolist()
train_y = cPickle.load(open('Dataset/train_y.p', 'rb')).tolist()

val_articles = cPickle.load(open('Dataset/val_articles.p', 'rb')).tolist()
val_headlines = cPickle.load(open('Dataset/val_headlines.p', 'rb')).tolist()
val_y = cPickle.load(open('Dataset/val_y.p','rb')).tolist()

train_articles = []
train_articles = train_articles1 + train_articles2 + train_articles3 + train_articles4 + val_articles
train_headlines = train_headlines + val_headlines
train_y = train_y + val_y

'''
train_articles = train_articles[:10]
train_headlines = train_headlines[:10]
train_y = train_y[:10]
'''

# --------------------------------------------------------------------------------------------------------------------------------
majority = -1
val = sum(train_y)
if (val >= (len(train_y)/2.0)):
    #genuine
    majority = 1
else:
    majority = 0
    
test_articles = cPickle.load(open('Dataset/test_articles.p', 'rb'))
test_headlines = cPickle.load(open('Dataset/test_headlines.p', 'rb'))
test_y = cPickle.load(open('Dataset/test_y.p', 'rb'))

pred = [majority for _ in range(len(test_y))]

print (sum(abs(test_y - pred)) / len(test_y))        
        
