# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 20:05:13 2021

@author: Mathew Pazhur
"""

import pickle
import pandas as pd
import nltk
from gensim.models.doc2vec import TaggedDocument
pd.options.mode.chained_assignment = None  # default='warn'


def tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            if len(word) < 2:
                continue
            tokens.append(word.lower())
    return tokens

def get_vectors(model, tagged_docs):
   sents = tagged_docs.values
   targets, regressors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in sents])
   return targets, regressors
   

data=[input("Enter description : ")]
df =pd.DataFrame(data, columns = ['Description'])






model = pickle.load(open('group-model.sav', 'rb'))
modellog = pickle.load(open('group-model logreg.sav', 'rb'))

modelritm = pickle.load(open('finalized_model-ritmdump.sav', 'rb'))
modellogritm = pickle.load(open('finalized_model-ritmdump logreg.sav', 'rb'))


test2_tagged = df.apply(
    lambda r: TaggedDocument(words=tokenize_text(r['Description']), tags=['']), axis=1)


# incident 

# y_test2, X_test2 = get_vectors(model, test2_tagged)

# y_pred2 = modellog.predict(X_test2)

# y_pred3 = modellog.predict_proba(X_test2)


#ritm

y_test2, X_test2 = get_vectors(model, test2_tagged)

y_pred2 = modellog.predict(X_test2)

y_pred3 = modellog.predict_proba(X_test2)

prob_list=[]
for x in range(3):
    prob_list.append(y_pred3[0][x])
    
max_prob = max(prob_list)

c=-1

for x in prob_list:
    c+=1
    if(max_prob==x):
        categ=c
        
if(max_prob>0.50):
    if(categ==0):
        print('Category : 1')
    elif(categ==1):
        print('Category : 2')
    elif(categ==2):
        print('Category : 3')

print('------------------')
print(y_pred2)
print(y_pred3)
