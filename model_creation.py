import pandas as pd

import re
from nltk.tokenize import word_tokenize

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords 
from sklearn.utils import resample

from tqdm.notebook import tqdm_notebook
tqdm_notebook.pandas()
from gensim.models import Doc2Vec
from sklearn import utils
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from gensim.models.doc2vec import TaggedDocument
import seaborn as sns
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import nltk
import multiprocessing
cores = multiprocessing.cpu_count()
import pickle

def upsample(df_majority,subcategory,count):
    
    df_minority = df[df.Subcategory==subcategory]

    
    df_minority_upsampled = resample(df_minority, 
                                replace=True,     # sample with replacement
                                n_samples=count,    # to match majority class
                                random_state=123) # reproducible results

    df_upsampled = pd.concat([df_majority, df_minority_upsampled])
    return df_upsampled

def cleanText(text):
    text = BeautifulSoup(text, "lxml").text
    text = re.sub(r'\|\|\|', r' ', text) 
    text = re.sub(r'http\S+', r'<URL>', text)
    text = text.lower()
    text = text.replace('x', '')
    return text

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

df = pd.read_csv('ritmdump.csv')


df = df[['Description','Subcategory']]
df = df[pd.notnull(df['Description'])]
df = df[pd.notnull(df['Subcategory'])]

#Removing unwanted Data

df=df[df["Subcategory"].isin(['REQUEST A', 'REQUEST B', 'REQUEST C', 'REQUEST D', 'REQUEST E'])]

df=df[~df.Description.str.contains('''stuff that is not required in our data''')]

#Sampling 

df_majority = df[df.Subcategory=='REQUEST A']

df_upsampled=upsample(df_majority,'REQUEST B',250)

df_upsampled=upsample(df_upsampled,'REQUEST C',250)

df_upsampled=upsample(df_upsampled,'REQUEST D',280)

df_upsampled=upsample(df_upsampled,'REQUEST E',400)

df=df_upsampled


#Preprocessing

X=df['Description']
stop_words = set(stopwords.words('english'))


X = X.reset_index(drop=True)

documents = []
stemmer = WordNetLemmatizer()
doc_cleaned_string=''


for sen in range(0, len(X)):  
        
        # Remove all the special characters
        document = re.sub(r'\W', ' ', str(X[sen]))
        # print(document)
        # print(1)
        document = document.replace('_',' ')
        # print(document)
        # document = document.replace('_',' ')
        # print(2)
        #Remove Whitespaces
        #document = document.strip()
    
        # remove all single characters
        document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
        # print(document)
        # print(3)
        
        # Remove single characters from the start
        document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) 
        # print(document)
        # print(4)
        
        # Substituting multiple spaces with single space
        document = re.sub(r'\s+', ' ', document, flags=re.I)
        # print(document)
        # print(5)
        
        # Removing prefixed 'b'
        document = re.sub(r'^b\s+', '', document)
        # print(document)
        # print(6)
        # break
        # Converting to Lowercase
        document = document.lower()
        
        # Removing underscores
        # 
                
        # print(document)
       
        
        # Lemmatization
        # document = document.split()

        
    
        # document = [stemmer.lemmatize(word) for word in document]
        
        # document = ' '.join(document)
        
        # print(document)
        # break
        
        #Tokenizing sentence
        doc_word_tokens = word_tokenize(document)
        
        #Removing stopwords
        doc_cleaned_list = [w for w in doc_word_tokens if not w in stop_words] 
        
        #Creating cleaned string from cleaned list
        for x in doc_cleaned_list:
            doc_cleaned_string=doc_cleaned_string+x+' '
         
        #Adding string to list
        documents.append(doc_cleaned_string)
        
        doc_cleaned_string=''


df['Description']=documents
df.index=range(df.shape[0])

cnt_pro = df['Subcategory'].value_counts()

plt.figure(figsize=(12,4))
sns.barplot(cnt_pro.index, cnt_pro.values, alpha=0.8)
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Subcategory', fontsize=12)
plt.xticks(rotation=90)
plt.show();

df['Description'] = df['Description'].apply(cleanText)


train, test = train_test_split(df, test_size=0.3, random_state=42)
train_tagged = train.apply(
    lambda r: TaggedDocument(words=tokenize_text(r['Description']), tags=[r.Subcategory]), axis=1)
test_tagged = test.apply(
    lambda r: TaggedDocument(words=tokenize_text(r['Description']), tags=[r.Subcategory]), axis=1)

train_tagged.values[30]

model_dbow = Doc2Vec(dm=1, vector_size=300, negative=5, hs=0, min_count=2, sample = 0, workers=cores)
model_dbow.build_vocab([x for x in tqdm_notebook(train_tagged.values)])

for epoch in range(30):
    model_dbow.train(utils.shuffle([x for x in tqdm_notebook(train_tagged.values)]), total_examples=len(train_tagged.values), epochs=1)
    model_dbow.alpha -= 0.002
    model_dbow.min_alpha = model_dbow.alpha
    
def vec_for_learning(model, tagged_docs):
    sents = tagged_docs.values
    targets, regressors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in sents])
    return targets, regressors

# y_train, X_train = vec_for_learning(model_dbow, train_tagged)
# y_test, X_test = vec_for_learning(model_dbow, test_tagged)

logreg = LogisticRegression(n_jobs=1, solver='saga', C=1e5, max_iter=10000)
# logreg.fit(X_train, y_train)
# y_pred = logreg.predict(X_test)
from sklearn.metrics import accuracy_score, f1_score
# print('Testing accuracy %s' % accuracy_score(y_test, y_pred))
# print('Testing F1 score: {}'.format(f1_score(y_test, y_pred, average='weighted')))

model_dmm = Doc2Vec(dm=1, dm_mean=1, vector_size=300, window=10, negative=5, min_count=1, workers=5, alpha=0.065, min_alpha=0.065)
model_dmm.build_vocab([x for x in tqdm_notebook(train_tagged.values)])

# %%time
for epoch in range(30):
    model_dmm.train(utils.shuffle([x for x in tqdm_notebook(train_tagged.values)]), total_examples=len(train_tagged.values), epochs=1)
    model_dmm.alpha -= 0.002
    model_dmm.min_alpha = model_dmm.alpha

# y_train, X_train = vec_for_learning(model_dmm, train_tagged)
# y_test, X_test = vec_for_learning(model_dmm, test_tagged)
# logreg.fit(X_train, y_train)
# y_pred = logreg.predict(X_test)
# print('Testing accuracy %s' % accuracy_score(y_test, y_pred))
# print('Testing F1 score: {}'.format(f1_score(y_test, y_pred, average='weighted')))

from gensim.test.test_doc2vec import ConcatenatedDoc2Vec
new_model = ConcatenatedDoc2Vec([model_dbow, model_dmm])

   
y_train, X_train = get_vectors(new_model, train_tagged)
y_test, X_test = get_vectors(new_model, test_tagged)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print('Testing accuracy %s' % accuracy_score(y_test, y_pred))
print('Testing F1 score: {}'.format(f1_score(y_test, y_pred, average='weighted')))

filename = 'subcat-ritm-model.sav'
pickle.dump(new_model, open(filename, 'wb'))

filename = 'subcat-ritm-model logreg.sav'
pickle.dump(logreg, open(filename, 'wb'))




# test2=test
# test2=test2.iloc[-1:,:]
# test2.Description.iloc[0]=''
# test2_tagged = test2.apply(
#     lambda r: TaggedDocument(words=tokenize_text(r['Description']), tags=['']), axis=1)

# y_test2, X_test2 = get_vectors(model, test2_tagged)
# y_pred2 = modellog.predict(X_test2)
# print(y_pred2)
