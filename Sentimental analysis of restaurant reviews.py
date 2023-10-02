#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[4]:


data=pd.read_csv('Restaurant_Reviews.tsv',delimiter='\t',quoting=3)


# In[6]:


data.shape


# In[8]:


data.columns


# In[9]:


data.head()


# In[10]:


data.info


# In[13]:


import nltk #Natural Language Toolkit used for NLP
import re   #Used to work with regular expressions
nltk.download('stopwords')    #nltk corpus is massive dump of all kind of natural language dataset
from nltk.corpus import stopwords   #stopword is commonly used word(such as 'a','an','the','in') that a search engine has been programmed to ignore
from nltk.stem.porter import PorterStemmer    #Porter Stemmer is used for data mining and Information Retrieval


# In[15]:


#cleaning the reviews
corpus=[]
for i in range(0,1000):
    #Cleaning special characters from the review
    review=re.sub(pattern='[^a-zA-Z]',repl=' ',string=data['Review'][i])
    #converting entire review into lower case
    review=review.lower()
    #Tokenizing the review by words
    review_words=review.split()
    #Stemming the words
    ps=PorterStemmer()
    review=[ps.stem(word) for word in review_words]
    #joining the stemmed words
    review=' '.join(review)
    #Creating a corpus
    corpus.append(review)
    
    


# In[16]:


corpus[:1500]


# In[17]:


#Creating the bag of words model
from sklearn.feature_extraction.text import CountVectorizer   #CountVectorizer convers text to numerical data
cv=CountVectorizer(max_features=1500)
X=cv.fit_transform(corpus).toarray()
y=data.iloc[:,1].values


# In[19]:


#Split Data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=0)
#Random state sends seed to random generator, so that train-test split is always deterministic. If you don't set seed, it's different each time


# In[20]:


X_train.shape,X_test.shape,y_train.shape,y_test.shape


# In[21]:


from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(random_state=0)
classifier.fit(X_train,y_train)



# In[22]:


#Predicting test set results
y_pred=classifier.predict(X_test)
y_pred


# In[24]:


#Accuracy, Precision and Recall
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score

score1=accuracy_score(y_test,y_pred)
score2=precision_score(y_test,y_pred)
score3=recall_score(y_test,y_pred)

print("Scores")
print("Accuracy Score is {}%".format(round(score1*100,2)))
print("Precision Score is {}%".format(round(score2*100,2)))


# In[25]:


#Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)


# In[26]:


cm


# In[30]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

plt.figure(figsize=(10,6))
sns.heatmap(cm, annot=True, cmap='YlGnBu', xticklabels=['Negative','Positive'], yticklabels=['Negative','Positive'])
plt.xlabel("Predicted Values")
plt.ylabel("Actual Label")


# In[36]:


#HyperParameter tuning the Naive Bayes Classifier
best_accuracy=0.0
alpha_val=0.0
for i in np.arange(0.1,1.1,0.1):
    temp_classifier=RandomForestClassifier(random_state=0)
    temp_classifier.fit(X_train,y_train)
    temp_y_pred= temp_classifier.predict(X_test)
    score=accuracy_score(y_test,temp_y_pred)
    print("Accuracy score for alpha={} is: {}%".format(round(i,1),round(score*100,2)))
    if score>best_accuracy:
        best_accuracy=score
        alpha_val=i
print("-------------------------------------------")
print("The best accuracy score is {}% for alpha value as {}".format(round(best_accuracy*100,2),round(alpha_val,1)))


# In[37]:


classifier=RandomForestClassifier(random_state=0)
classifier.fit(X_train,y_train)


# In[38]:


def predict_sentiment(sample_review):
    sample_review=re.sub(pattern='[^a-zA-Z]',repl=' ',string=sample_review)
    sample_review=sample_review.lower()
    sample_review_words=sample_review.split()
    ps=PorterStemmer()
    final_review=[ps.stem(word) for word in sample_review_words]
    final_review=' '.join(final_review)
    
    temp=cv.transform([final_review]).toarray()
    return classifier.predict(temp)
    


# In[44]:


samplereview=str(input())
if(predict_sentiment(samplereview)):
    print("This is a positive review")
else:
    print("This is a negative review")


# In[ ]:




