#!/usr/bin/env python
# coding: utf-8

# In[2]:


https://github.com/edyoda/data-science-complete-tutorial/blob/master/Data/house_rental_data.csv.txt



Data cleaning & getting rid of irrelevant information before clustering
Finding the optimal value of k
Storing cluster to which the house belongs along with the data


# In[3]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from sklearn.datasets import make_blobs
import seaborn as sns
import math
print("done")


# In[4]:


df=pd.read_csv('data/House_Rental_Dataset.csv')
df


# In[5]:


df.isnull().sum()


# In[6]:


df.head()


# In[11]:


df=df.drop('Unnamed: 0',axis=1)
df


# In[12]:


X=df.iloc[:, :-1].values
y=df.iloc[:, 4].values


# In[13]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20)


# In[14]:


from sklearn.neighbors import KNeighborsClassifier
training_accuracy = []
test_accuracy=[]

neighbors=range(1,11)
for number_of_neighbors in neighbors:
    KNN=KNeighborsClassifier(n_neighbors=number_of_neighbors)
    KNN.fit(X_train,y_train)
    training_accuracy.append(KNN.score(X_train,y_train))
    test_accuracy.append(KNN.score(X_test,y_test))


# In[16]:


plt.plot(neighbors,training_accuracy,label="training_accuracy")
plt.plot(neighbors,test_accuracy,label="test_accuracy")
print("K=3")


# In[28]:


plt.scatter(df['TotalFloor'],df['Bathroom'],df['Bedroom'],df['Price'])


# In[31]:


from sklearn.cluster import KMeans


# In[36]:


k=KMeans(n_clusters=3)
k


# In[43]:


y_predict =k.fit_predict(df[['TotalFloor','Bathroom','Bedroom','Price']])
y_predict


# In[45]:


df['cluster']=y_predict


# In[46]:


df1=df[df.cluster==0]
df2=df[df.cluster==1]
df3=df[df.cluster==2]

plt.scatter(df1.Price,df1['TotalFloor'],color='green')
plt.scatter(df2.Price,df2['Bathroom'],color='red')
plt.scatter(df3.Price,df3['Bedroom'],color='orange')

plt.xlabel('Price')
plt.ylabel('bbl')
plt.legend()


# In[ ]:



