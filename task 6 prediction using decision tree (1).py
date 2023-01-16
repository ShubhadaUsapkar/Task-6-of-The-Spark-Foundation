
# ## Author :Shubhada Mangesh Usapkar
# ## Problem Statement
# ## ● Create the Decision Tree classifier and visualize it graphically.
# 
# ###  ● The purpose is if we feed any new data to this classifier, it would be able to predict the right class accordingly.

# ## Data Collection
# Import all libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
# Import Dataset
iris = pd.read_csv('C:/Users/Viraj Vijay Samant/Downloads/Iris (2).csv')
iris.head()
iris.columns
iris.shape
iris.isnull().sum()
iris.info()
iris.describe()
# ## Data Preprocessing
iris.drop(['Id'], axis = 'columns',inplace = True)
iris.head()
# Convert text values into numeric
label_enco = LabelEncoder()
iris['Species'] = label_enco.fit_transform(iris['Species'])
x = iris.drop(['Species'],axis = 'columns')
y = iris['Species']
# ## Bulid Model
model = tree.DecisionTreeClassifier()
model.fit(x, y)
## Visualize the Decision Tree
tree.plot_tree(model)
feature_cols = ['SepalLengthCm','SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
class_names = ['setosa', 'versicolor', 'virginica']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
tree.plot_tree(model,
               feature_names = feature_cols, 
               class_names=class_names,
               filled = True);
fig.savefig('TreeDia.png')
