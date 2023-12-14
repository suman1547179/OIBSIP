#!/usr/bin/env python
# coding: utf-8

# #### Name : SUMAN PAUL KAMBHAMPATI
# ##### Data Science Intern  
# ##### Oasis Infobyte - October P-1
# ##### Task 5 - Sales Prediction using Python
# 
# 

# In[1]:


#importing necessary libraries
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
#importing libraries for visualisation
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns


# In[6]:


#importing Data
data_frame = pd.read_csv("C:\\Users\\sumanpaul\\Downloads\\internships\\OASISINFOBYTE\\TASK-5\\Advertising.csv", index_col = 0)


# ####  Performing descriptive analysis. Understand the variables and their corresponding values.

# In[7]:


# Understanding the dimensions of data
data_frame.shape


# In[8]:


# Understanding the Data Variables
data_frame.info()


# In[9]:


data_frame.columns


# ###### The company spent their budget for differnt products on 3 advertising medias such as TV, Radio, Newspaper and the corresponding sales for each product

# In[10]:


# Show the top 5 Rows of data
data_frame.head()


# In[11]:


# Performing Descriptive Analysis
data_frame.describe().T


# In[12]:


# Check for Duplicated Entries
data_frame.duplicated().sum()


# #### Outlier Analysis

# In[13]:


# Reset the index
data_frame = data_frame.reset_index(drop=True)

# Create the boxplot
fig, axs = plt.subplots(1, 1)
plt1 = sns.boxplot(data=data_frame['TV'], ax=axs)
plt.show()


# In[14]:


fig,axs=plt.subplots(1,1)
plt1=sns.boxplot(data_frame['Radio'],ax=axs)


# In[15]:


fig,axs=plt.subplots(1,1)
plt1=sns.boxplot(data_frame['Newspaper'],ax=axs)


# In[16]:


fig,axs=plt.subplots(1,1)
plt1=sns.boxplot(data_frame['Sales'],ax=axs)


# #### Data Visualization

# ###### Data Visualization helps to  show how the budget spent on each advertising media affect the sales of products

# In[17]:


#Scatter plot is used find the distribution of effects of each advertising media against Target Sales variable
plt.figure(figsize=(6,4))
sns.scatterplot(data=data_frame,x=data_frame['TV'],y=data_frame['Sales'])


# In[18]:


plt.figure(figsize=(6,4))
sns.scatterplot(data=data_frame,x=data_frame['Radio'],y=data_frame['Sales'])


# In[19]:


plt.figure(figsize=(6,4))
sns.scatterplot(data=data_frame,x=data_frame['Newspaper'],y=data_frame['Sales'])


# ##### * It is seen that TV data set is more linear as compared to other 2 variables .

# #### Heat Map

# In[20]:


# find correlation between variables in data set for plotting heatmap
df_corr=data_frame.corr()


# In[21]:


plt.figure(figsize=(10,6))
sns.heatmap(df_corr,annot=True,cmap="BuPu")
plt.show()


# ##### * We can see that TV variable has highest correlation value with the target Sales variable

# #### Building the Forecasting Model

# In[22]:


from sklearn.model_selection import train_test_split


# In[23]:


#First step in building the forecasting model is to identify the Feature(Input) variables and Target (Output) variable
features = data_frame[['TV', 'Radio', 'Newspaper']]
target = data_frame[['Sales']]


# #####  * Splitting data for training and testing the model

# In[24]:


# Splitting data for training the model and testing the model
# train size taken as 0.8
X_train, X_test, y_train, y_test = train_test_split(features, target, train_size = .8)
# Dimensions of Train and Test Data sets
print('Train set of features: ', X_train.shape)
print('Test set of features: ', X_test.shape)
print('Target for train: ', y_train.shape)
print('Target for test: ', y_test.shape)


# ### Learn the model on train data

# In[25]:


from sklearn.linear_model import LinearRegression


# In[26]:


# Linear Regression Model ( a Supervised Machine learning Algorithm)
# LR models impose a linear function between predictor and response variables
my_model = LinearRegression()


# In[27]:


# Fitting the model in train data set ie the Linear Regression Model learned from the on Train Data
my_model.fit(X_train, y_train)


# #### Predicting the Sales

# In[28]:


# Predicting the sales from Feature Test values
y_pred = my_model.predict(X_test)
y_pred


# #### Test the model

# In[29]:


from sklearn.metrics import mean_squared_error


# ##### Mean Squared Error

# In[30]:


# Compare the predicted values with the true values
mean_squared_error(y_pred, y_test)


# ##### Coefficient of Determination or R Squared Value (r2)

# In[31]:


from sklearn.metrics import r2_score


# In[32]:


# find Coefficient of Determination or R Squared Value (r2)
r2_score(y_test,y_pred)

