import os
import pandas as pd
import numpy as np
import nltk
import gensim
import ast
import sklearn
from gensim import corpora, models, similarities    
from nltk.tokenize import RegexpTokenizer



ads = pd.read_csv('ads.csv',encoding='iso-8859-1')
search = pd.read_csv('search.csv',encoding='iso-8859-1')

print(len(ads))
len(search)

def vectorize(row):
    cell=row[0]
    vector=[]
    if cell is not None:
        cell = ast.literal_eval(cell)
        assert(type(cell)==dict)
    
        vector=list(cell.values())
    if row[1] is not None:
        vector.append(row[1])
        
    return " ".join(vector).lower()



ads = ads[['Params','Title']]
ads['Vector'] = ads.apply(vectorize,axis=1)
ads = pd.concat([ads for _ in range(10)])
ads_corpus = ads['Vector'].tolist()
print('ads')
print(len(ads_corpus))

search = search[['SearchParams','SearchQuery']]
search['Vector'] = search.apply(vectorize,axis=1)
search_corpus = search['Vector'].tolist()
print('\nsearch')
print(len(search_corpus))

import nltk
nltk.download('stopwords')
nltk.download('punkt')

from nltk.corpus import stopwords
import string
stop = stopwords.words('english') + list(string.punctuation) + ["''", "'s"]

ads_corp= [ [i for i in nltk.word_tokenize(sent.lower()) if i not in stop] for sent in ads_corpus]
search_corp= [ [i for i in nltk.word_tokenize(sent.lower()) if i not in stop] for sent in search_corpus]

ads_corp = ads_corp + search_corp
print(ads_corp[:10])

ads_model = gensim.models.Word2Vec(ads_corp, min_count=1, size = 32)

words = np.array(sorted(set([x for p in ads_corp for x in p])))
print(len(words))
print(np.where(words=='accessories')[0][0])
arrs = np.array([np.array(sklearn.preprocessing.normalize(ads_model.wv.__getitem__([word]), norm='l2', axis=1, copy=True, return_norm=False)[0]) for word in words])

print(arrs[:10])

np.save('arrs.npy', arrs)

np.save('words.npy', words)
