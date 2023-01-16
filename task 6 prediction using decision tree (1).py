#!/usr/bin/env python
# coding: utf-8

# ## Author :Shubhada Mangesh Usapkar
# ## Problem Statement
# ## ● Create the Decision Tree classifier and visualize it graphically.
# 
# ###  ● The purpose is if we feed any new data to this classifier, it would be able to predict the right class accordingly.

# ## Data Collection

# In[1]:


# Import all libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import train_test_split
from sklearn import tree

from sklearn.preprocessing import LabelEncoder


# In[3]:


# Import Dataset

iris = pd.read_csv('C:/Users/Viraj Vijay Samant/Downloads/Iris (2).csv')
iris.head()


# In[4]:


iris.columns


# In[5]:


iris.shape


# In[6]:


iris.isnull().sum()


# In[7]:


iris.info()


# In[8]:


iris.describe()


# ## Data Preprocessing

# In[9]:


iris.drop(['Id'], axis = 'columns',inplace = True)
iris.head()


# In[10]:


# Convert text values into numeric
label_enco = LabelEncoder()


# In[11]:


iris['Species'] = label_enco.fit_transform(iris['Species'])


# In[12]:


x = iris.drop(['Species'],axis = 'columns')
y = iris['Species']


# ## Bulid Model

# In[13]:


model = tree.DecisionTreeClassifier()


# In[14]:


model.fit(x, y)


# ## Visualize the Decision Tree

# In[15]:


tree.plot_tree(model)


# In[16]:


feature_cols = ['SepalLengthCm','SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
class_names = ['setosa', 'versicolor', 'virginica']

fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
tree.plot_tree(model,
               feature_names = feature_cols, 
               class_names=class_names,
               filled = True);
fig.savefig('TreeDia.png')


# ## thanku

# In[ ]:




