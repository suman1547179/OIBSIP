#!/usr/bin/env python
# coding: utf-8

# In[39]:


from IPython.display import Image
Image(url='https://editor.analyticsvidhya.com/uploads/51518iris%20img1.png', width=850)



# In[4]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


get_ipython().system('pip install seaborn')


# In[5]:


iris = pd.read_csv("C:\\Users\\sumanpaul\\Downloads\\internships\\OASISINFOBYTE\\TASK-1\\Iris.csv")
iris.head()


# In[6]:


# Rename the complex columns name
iris= iris.rename(columns={'SepalLengthCm':'Sepal_Length',
                           'SepalWidthCm':'Sepal_Width',
                           'PetalLengthCm':'Petal_Length',
                           'PetalWidthCm':'Petal_Width'})
     


# In[7]:


iris.head()
     

    


# In[8]:


# checking null values
iris.isnull().sum()


# In[9]:


# checking if the data is biased or not
iris ['Species'].value_counts()
     


# In[10]:


# checking statistical features
iris.describe()


# In[11]:


sns.FacetGrid(iris, hue="Species",height=6).map(plt.scatter,"Petal_Length","Sepal_Width").add_legend()


# In[12]:


# visualize the whole dataset
sns.pairplot(iris[['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width','Species']], hue="Species",diag_kind='kde')


# In[13]:


# Separate features and target
data=iris.values


# In[14]:


# slicing the matrices
X=data[:,0:4]
Y=data[:,5]


# In[15]:


print(X.shape)
print(X)
     


# In[16]:


print(Y.shape)
print(Y)


# In[19]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X,Y, test_size=0.2)
    


# In[18]:


get_ipython().system('pip install scikit-learn')


# In[20]:


print(X_train.shape)
print(X_train)


# In[21]:


print(y_test.shape)
print(y_test)


# In[22]:


print(X_test.shape)
print(X_test)


# In[23]:


print(y_train.shape)
print(y_train)


# In[24]:


from sklearn.svm import SVC

model_svc=SVC()
model_svc.fit(X_train,y_train)
    


# In[25]:


prediction1 = model_svc.predict(X_test)

#calculate the accuracy
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, prediction1))


# In[26]:


flower_mapping = {'Iris-setosa':0,'Iris-versicolor':1,'Iris-virginica':2}
iris['Species']=iris['Species'].map(flower_mapping)
     


# In[27]:


iris.head()


# In[28]:


iris.tail()


# In[29]:


# preparing inputs and outputs
X=iris [['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width']].values
y= iris[['Species']].values


# In[30]:


from sklearn.linear_model import LogisticRegression
model= LogisticRegression()
model.fit(X,y)


# In[31]:


model.score(X,y)
     


# In[32]:


expected = y
predicted = model.predict(X)
predicted
     


# In[33]:


from sklearn import metrics


# In[34]:


print(metrics.classification_report(expected, predicted))


# In[35]:


print(metrics.confusion_matrix(expected, predicted))


# In[36]:


from sklearn.tree import DecisionTreeClassifier
model_DTC = DecisionTreeClassifier()
model_DTC.fit(X_train, y_train)


# In[37]:


prediction3= model_svc.predict(X_test)

#calculate the accuracy
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, prediction3))


# In[38]:


# New data for prediction
X_new = np.array([[3, 2, 1, 0.2], [4.9, 2.2, 3.8, 1.1], [5.3, 2.5, 4.6, 1.9]])

# Predicting the sizes of the iris flowers
predicted_sizes = model.predict(X_new)

# Output the predicted sizes
print(predicted_sizes)


# In[ ]:




