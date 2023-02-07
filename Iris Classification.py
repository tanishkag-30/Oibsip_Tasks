#!/usr/bin/env python
# coding: utf-8

# In[20]:


pip install manager


# In[22]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[23]:


# load the csv data
df = pd.read_csv('Downloads/Iris.csv')
df.head()


# In[24]:


# delete a column
df = df.drop(columns = ['Id'])
df.head()


# In[25]:


# to display stats about data
df.describe()


# In[26]:


# to display stats about data
df.describe()


# In[27]:


# to display no. of samples on each class
df['Species'].value_counts()


# In[28]:


# check for null values
df.isnull().sum()


# In[29]:


# histograms
df['SepalLengthCm'].hist()


# In[30]:


df['SepalWidthCm'].hist()


# In[31]:


df['PetalLengthCm'].hist()


# In[32]:


df['PetalWidthCm'].hist()


# In[35]:


# create list of colors and class labels
colors = ['yellow', 'red', 'green']
species = ['Iris-virginica', 'Iris-versicolor', 'Iris-setosa']


# In[36]:


for i in range(3):
    # filter data on each class
    x = df[df['Species'] == species[i]]
    # plot the scatter plot
    plt.scatter(x['SepalLengthCm'], x['SepalWidthCm'], c = colors[i], label=species[i])
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.legend()


# In[37]:


for i in range(3):
    # filter data on each class
    x = df[df['Species'] == species[i]]
    # plot the scatter plot
    plt.scatter(x['PetalLengthCm'], x['PetalWidthCm'], c = colors[i], label=species[i])
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.legend()


# In[38]:


for i in range(3):
    # filter data on each class
    x = df[df['Species'] == species[i]]
    # plot the scatter plot
    plt.scatter(x['SepalLengthCm'], x['PetalLengthCm'], c = colors[i], label=species[i])
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.legend()


# In[39]:


for i in range(3):
    # filter data on each class
    x = df[df['Species'] == species[i]]
    # plot the scatter plot
    plt.scatter(x['SepalWidthCm'], x['PetalWidthCm'], c = colors[i], label=species[i])
plt.xlabel("Sepal Width")
plt.ylabel("Petal Width")
plt.legend()


# In[40]:


# display the correlation matrix
df.corr()


# In[41]:


corr = df.corr()
# plot the heat map
fig, ax = plt.subplots(figsize=(5,4))
sns.heatmap(corr, annot=True, ax=ax, cmap = 'coolwarm')


# In[42]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
# transform the string labels to integer
df['Species'] = le.fit_transform(df['Species'])
df.head()


# In[116]:


from sklearn.model_selection import train_test_split
## train - 70%
## test - 30%

# input data
X = df.drop(columns=['Species'])
# output data
Y = df['Species']
# split the data for train and test
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30)


# In[117]:


# logistic regression 
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()


# In[118]:


# model training
model.fit(x_train, y_train)


# In[119]:


# print metric to get performance
print("Accuracy: ",model.score(x_test, y_test) * 100)


# In[122]:


# knn - k-nearest neighbours
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier()
model.fit(x_train, y_train)

# print metric to get performance
print("Accuracy: ",model.score(x_test, y_test) * 100)


# In[125]:


# decision tree
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()
model.fit(x_train, y_train)

# print metric to get performance
print("Accuracy: ",model.score(x_test, y_test) * 100)


# In[ ]:




