#!/usr/bin/env python
# coding: utf-8

# ### Import Libraries

# In[1]:


import pickle
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
import nltk
nltk.download('stopwords')


# ### Data Preprocessing

# In[2]:


def preprocess_tweet(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower())
    text = text+' '.join(emoticons).replace('-', '') 
    return text


# In[3]:


tqdm.pandas()
df = pd.read_csv('suicidal_data1.csv',sep=',')
print(df.head())
print(df['tweet'])
df['tweet'] = df['tweet'].progress_apply(preprocess_tweet)


# In[4]:


from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()
def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]


# In[5]:


from nltk.corpus import stopwords
stop = stopwords.words('english')


# In[6]:


[w for w in tokenizer_porter('a runner likes running and runs a lot') if w not in stop]


# In[7]:


def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\(|D|P)',text.lower())
    text = re.sub('[\W]+', ' ', text.lower())
    text += ' '.join(emoticons).replace('-', '')
    tokenized = [w for w in tokenizer_porter(text) if w not in stop]
    return tokenized


# ### Using the Hashing Vectorizer

# In[8]:


from sklearn.feature_extraction.text import HashingVectorizer
vect = HashingVectorizer(decode_error='ignore', n_features=2**21, 
                         preprocessor=None,tokenizer=tokenizer)


# ### Building the Model

# In[9]:


from sklearn.linear_model import SGDClassifier
clf = SGDClassifier(loss='log', random_state=1)


# In[10]:


X = df["tweet"].to_list()
y = df['label']


# In[11]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,
                                                 y,
                                                 test_size=0.20,
                                                 random_state=0)


# In[12]:


X_train = vect.transform(X_train)
X_test = vect.transform(X_test)


# In[13]:


classes = np.array([0, 1])
clf.partial_fit(X_train, y_train,classes=classes)


# In[14]:


print('Accuracy: %.3f' % clf.score(X_test, y_test))


# In[15]:


clf = clf.partial_fit(X_test, y_test)


# ### Testing and making Predictions

# In[16]:


label = {0:'negative', 1:'positive'}
example = ["I'll kill myself am tired of living depressed and alone"]
X = vect.transform(example)
print('Prediction: %s\nProbability: %.2f%%'
      %(label[clf.predict(X)[0]],np.max(clf.predict_proba(X))*100))


# In[17]:


label = {0:'negative', 1:'positive'}
example = ["It's such a hot day, I'd like to have ice cream and visit the park"]
X = vect.transform(example)
print('Prediction: %s\nProbability: %.2f%%'
      %(label[clf.predict(X)[0]],np.max(clf.predict_proba(X))*100))


# In[ ]:




