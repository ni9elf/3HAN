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



print ('removing stop words...')
stops = set(stopwords.words("english"))                  
print (stops)
    
train_headlines = [[w for w in line if not w in stops] for line in train_headlines]  

train_headlines = [ ( " ".join(line)) for line in train_headlines]


train_articles = [[[w for w in line if not w in stops] for line in art] for art in train_articles]

train_articles = [[ ( " ".join(line)) for line in art] for art in train_articles]

i = 0
for head in train_headlines:
	train_articles[i].insert(0, head)
	i += 1
	


#articles = fake_articles + genuine_articles

train_articles = [( " ".join(line)) for line in train_articles]		

print (len(train_articles))



train_articles = np.asarray(train_articles)
train_y = np.asarray(train_y)




#-------------------------------------------------------------------------------------------------------------------------------------

print ('Run BOW...')

cvscores = []
for i in range(1):
    
    print ('Shuffling....')
    #shuffling training data
    p = np.random.permutation(len(train_articles))
    train_articles = train_articles[p]
    train_y = train_y[p]
    del p

    print ('Shuffling done...')
    # create model
    
    #TODO decide on max_features

    vectorizer = CountVectorizer(max_features = 50000, ngram_range = (1, 5))
                             
    print ('vectorize done...')
    train_data_features = vectorizer.fit_transform(train_articles)
   
    print ('classifier...')
    #TODO decide on number of trees
    model = LR()
    #model = RandomForestClassifier(n_estimators = 100) 
    
    del train_articles
    del train_headlines    

    print ('fitting ...')
    # Fit the model
    model = model.fit(train_data_features, train_y)

    del train_data_features
    del train_y 
    
    print ('load test...')

    test_articles = cPickle.load(open('Dataset/test_articles.p', 'rb'))
    test_headlines = cPickle.load(open('Dataset/test_headlines.p', 'rb'))
    test_y = cPickle.load(open('Dataset/test_y.p', 'rb'))
    
    '''
    test_articles = test_articles[:10]
    test_headlines = test_headlines[:10]
    test_y = test_y[:10]
    '''
    
    test_headlines = [[w for w in line if not w in stops] for line in test_headlines] 
    test_headlines = [ ( " ".join(line)) for line in test_headlines]
    
    test_articles = [[[w for w in line if not w in stops] for line in art] for art in test_articles]
    test_articles = [[ ( " ".join(line)) for line in art] for art in test_articles]
    
    i = 0
    for head in test_headlines:
	    test_articles[i].insert(0, head)
	    i += 1
    
    test_articles = [( " ".join(line)) for line in test_articles]	
    

    print (len(test_articles))

   
    
    test_articles = np.asarray(test_articles)
    test_y = np.asarray(test_y)    

    test_data_features = vectorizer.transform(test_articles)
    
    del test_articles
    del test_headlines

    print ('predict on test...')
    # evaluate the model
    pred = model.predict(test_data_features)



    pscore = metrics.accuracy_score(test_y, pred)
    
    print ("\n")
    #print (i)
    print (pscore)
    cvscores.append(pscore * 100)
	
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
