#!/usr/bin/env python
# coding: utf-8

# # This is a supervised learning problem in which we are using logistic regression algorithm for classification.

# In[4]:


#1. Import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[5]:


#2. Import the dataset
dataset = pd.read_csv('Wine.csv')

# X has input value of [all the rows , all the columns except the last column i.e. segment in this dataset]
X = dataset.iloc[:, :-1].values

# y will store all the labels i.e. all the rows and only the last column.
y = dataset.iloc[:, -1].values


# In[6]:


X


# In[7]:


y


# In[8]:


#3. Split the dataset into test set and training set.
from sklearn.model_selection import train_test_split

#here below 0.2 means 20% of dataset will go as test dataset and remaining 80% will be training dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state =0)


# In[9]:


y_test


# In[10]:


#4. Feature scaling - it is applied only on training dataset

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[11]:


#5. Applying PCA
# PCA is applied to reduce the number of input dimensions.

from sklearn.decomposition import PCA

# Below, 2 is the number of dimensions you want your data to be i.e. now there will be only 2 columns of data.
pca = PCA(n_components = 2)

X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)


# In[12]:


#6. Train logistic regression

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train,y_train)

#here training is sucessfully completed of logistic regression


# In[13]:


#7. Confusion matrix

from sklearn.metrics import confusion_matrix
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test , y_pred)


# In[14]:


cm


# In[15]:


#8. Accuracy of the model

from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)


# In[41]:


#9. Visualizing training set result

from matplotlib.colors import ListedColormap
X_set , y_set = X_train , y_train

#it finds the min value in zeroth column 
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() -1, stop = X_set[: , 0].max() +1, step = 0.01), 
                     np.arange(start = X_set[:, 1].min() -1, stop = X_set[:, 1].max() +1, step = 0.01))

#plotting result of all data
plt.contourf(X1,X2,classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),alpha = 0.75,cmap = ListedColormap(('red','green','blue')))

#For plotting training data i.e scatter dots
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red','green','blue'))(i), label = j)
    
#for labelling 
    plt.title("Logistic Regression after PCA - Training Data")
plt.xlabel("PC1")
plt.ylabel("PC2")

#for tiny box in upright corner
plt.legend()
plt.show()


# In[19]:


X1


# In[40]:


#9. Visualizing test set result

from matplotlib.colors import ListedColormap
X_set , y_set = X_test , y_test

#it finds the min value in zeroth column 
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() -1, stop = X_set[: , 0].max() +1, step = 0.01), 
                     np.arange(start = X_set[:, 1].min() -1, stop = X_set[:, 1].max() +1, step = 0.01))

#plotting result of all data
plt.contourf(X1,X2,classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),alpha = 0.75,cmap = ListedColormap(('red','green','blue')))

#For plotting training data i.e scatter dots
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red','green','blue'))(i), label = j)
    
#for labelling 
    plt.title("Logistic Regression after PCA - Test Data")
plt.xlabel("PC1")
plt.ylabel("PC2")

#for tiny box in upright corner
plt.legend()
plt.show()

