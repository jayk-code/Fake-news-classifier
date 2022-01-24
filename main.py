#!/usr/bin/env python
# coding: utf-8

# In[49]:


import matplotlib.pyplot as plt 
from sklearn import metrics
from sklearn.metrics import confusion_matrix

import seaborn as sns
import pickle

import numpy as np # linear algebra
import pandas as pd #data processing

import os
import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')


# In[50]:


data_0 = pd.read_csv('news_train.csv')


# In[51]:


print(data_0.isnull().sum())


# In[52]:


a = data_0[data_0['text'].isnull()]
print(len(a))


# In[53]:


b = a[a['author'].isnull()]
print(len(b))


# In[54]:


data_0 = data_0.fillna(' ')


# In[8]:


data_0['total']=data_0['title']+' '+ data_0['author']+ ' ' +  data_0['text']


# In[9]:


x = []
y = []
for i in range(len(data_0['total'])):
    a = data_0['total'][i].split()
    if len(a) > 50:
        x_1 = data_0['total'][i]
        y_1 = data_0['label'][i]
        x.append(x_1)
        y.append(y_1)


# In[10]:


data = pd.DataFrame(columns = ['total', 'label'])


# In[11]:


data['total'] = x
data['label'] = y


# In[12]:


data.shape


# In[13]:


from nltk.corpus import stopwords


# In[14]:


STOPWORDS = set(stopwords.words('english'))


# In[15]:


contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not",

                           "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",

                           "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",

                           "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would",

                           "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",

                           "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam",

                           "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have",

                           "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock",

                           "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",

                           "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is",

                           "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as",

                           "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would",

                           "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have",

                           "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have",

                           "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are",

                           "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",

                           "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",

                           "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",

                           "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have",

                           "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",

                           "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",

                           "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",

                           "you're": "you are", "you've": "you have"}


# In[16]:


from nltk.stem.wordnet import WordNetLemmatizer


# In[17]:


for i in range(len(data['total'])):
    j = data['total'][i].split()
    a = []
    for k in j:
        l = str(k)
        a.append(l)
    data['total'][i] = ' '.join(a)


# In[18]:


lemmatizer=WordNetLemmatizer()
stop_words = set(stopwords.words('english')) 
def text_cleaner(text):
    newString = text.lower()
    #newString = BeautifulSoup(newString, "lxml").text
    newString = re.sub(r'\([^)]*\)', '', newString)
    newString = re.sub('"','', newString)
    newString = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in newString.split(" ")])    
    newString = re.sub(r"'s\b","",newString)
    newString = re.sub("[^a-zA-Z]", " ", newString) 
    tokens = [w for w in newString.split() if not w in stop_words]
    long_words=[]
    for i in tokens:
        if len(i)>=3:                  #removing short word
            long_words.append(lemmatizer.lemmatize(i))   
    return (" ".join(long_words)).strip()

cleaned_total = []
for t in data['total']:
    cleaned_total.append(text_cleaner(t))
    

    
    
data['cleaned_total'] = cleaned_total


# In[19]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data['cleaned_total'], data['label'], random_state=10)


# In[20]:


y_train.value_counts()


# In[21]:


from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


# In[27]:


vectorizer = TfidfVectorizer( stop_words='english',
                            decode_error='strict',
                            analyzer='word',
                            ngram_range=(1, 2),
                            max_features=5500
                            #max_df=0.5 # Verwendet im ML-Kurs unter Preprocessing                   
                            )


# In[28]:


tfidf_train = vectorizer.fit_transform(X_train) 
tfidf_test = vectorizer.transform(X_test)


# In[30]:


pickle.dump(vectorizer, open('transformer.pkl','wb'))


# In[31]:


X_train = tfidf_train
X_test = tfidf_test


# In[32]:


from sklearn.svm import SVC
SVC = SVC()
SVC.fit(X_train, y_train)
pred = SVC.predict(X_test)
print('Accuracy of SVC  classifier on training set: {:.2f}'
     .format(SVC.score(X_train, y_train)))
print('Accuracy of SVC classifier on test set: {:.2f}'
     .format(SVC.score(X_test, y_test)))
cm = confusion_matrix(y_test, pred)
print(cm)
print('Precision of SVC classifier is : ' + str(cm[0,0] / (cm[0,0] + cm[0,1])))


# In[33]:


import pickle


# In[34]:


pickle.dump(SVC, open('model.pkl','wb'))


# In[ ]:




