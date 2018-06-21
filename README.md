This repository contains the code for the models, baselines and visualizations used in the paper '3HAN: A Deep Neural Network for Fake News Detection'. Sneha Singhania, Nigel Fernandez, and Shrisha Rao In The 24th International Conference On Neural Information Processing (ICONIP) 2017.

***Abstract***:
The proliferation of fake news has possibly led to a mis-informed public opinion. We propose 3HAN, a three level hierarchical attention network, which creates a news vector, a representation of news articles to detect fake news. Our model (i) interprets the structure of an article as a three level hierarchy of words, sentences and headline, (ii) treats the headline as a distinctive feature and (iii) uses attention to differentiate elements of an hierarchy consistent with their interactions when constructing the news vector. Experiments on our large real world data set demonstrate the effectiveness of news vector in giving state-of- the-art performance. Visualizing the attention layers provides important parts of an article which is helpful in further human fact-checking.

<p align="center"> 
<img src="assets/architecture.png">
</p>

***Models***:
The models folder contains code for all the models run in the paper. These include: <br/>
3HAN-Ave: Vectors are composed using global average pooling <br/>
3HAN-Max: Vectors are composed using global max pooling <br/>
3HAN: Our proposed model with attention mechanism <br/>
3HAN+PT: Variant of 3HAN to include pre-training <br/>

***Baselines***:
The baselines folder contains code for all the baselines 3HAN was compared against in the paper. These include: <br/>

Word Count Based Models: <br/>

Majority uses the heuristic of taking the majority label in the training set as the assigning label to every point in the test set. <br/>
Bag-of-words and its TF-IDF constructs a vocabulary of the most frequent 50,000 words. The count of these words is used as features. The TF-IDF count is used as features in the other model variant. <br/>
Bag-of-ngrams and its TF-IDF uses the count of the 50,000 most frequent ngrams (n <= 5). The features are formed in the same way as above. <br/>
SVM+Bigrams uses the count of the 50,000 most frequent bigrams as features with an SVM classifier.

Neural Network Models: <br/>

GloVe-Avg flattens the article text to a word level granularity as a sequence of words. The GloVe embeddings of all words are averaged to form the feature vector. <br/>
GRU treats the article text as a sequence of words. A GRU with an annotation dimension of 300 is run on the sequence of GloVe word embeddings. The hidden annotation after the last time step is used as the feature vector. <br/>
GRU-Avg runs a GRU on the sequence of word embeddings and returns all hidden annotations at each time step. The average of these hidden annotations is used as the feature vector. <br/>
HAN and Variants include HAN-Ave, Han-Max and HAN. HAN uses a two level hierarchical attention network. HAN-Ave and Han-Max replaces the attention mechanism with average and max pooling for composition respectively. Since the code is not officially released we use our own implementation.

***Requirements:*** <br/>

Keras 2.0.4 <br/>
Theano 0.9.0 <br/>
Python 3.4.3
