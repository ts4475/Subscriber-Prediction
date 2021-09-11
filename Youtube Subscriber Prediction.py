#!/usr/bin/env python
# coding: utf-8

# # YouTube Subscriber Prediction

# In[295]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor


# In[296]:


data = pd.read_csv('youtube_data.csv')


# In[297]:


print(len(data.index)) # no.rows
print(len(data.columns)) # no.columns


# In[298]:


data.head(10)


# In[299]:


data.info()


# In[300]:


data.describe()


# # PRE PROCESSING

# In[301]:


def preprocess_inputs(df):
    df = df.copy()
    
    # Droping unused columns(video and video title)
    df = df.drop(['Video', 'Video title'], axis=1)
    
    # Drop row with missing target value
    missing_target_row = df[df['Subscribers'].isna()].index
    df = df.drop(missing_target_row, axis=0).reset_index(drop=True)
    
    # Extract date features
    df['Video publish time'] = pd.to_datetime(df['Video publish time'])
    df['Video month'] = df['Video publish time'].apply(lambda x: x.month) #separating the month
    df['Video day'] = df['Video publish time'].apply(lambda x: x.day) #separating the day
    df = df.drop('Video publish time', axis=1) #after extracting , droppping it
    
    # Convert durations to seconds
    df['Average view duration'] = pd.to_datetime(df['Average view duration']).apply(lambda x: (x.minute * 60) + x.second)
    
    # Split df into X and y
    df['Subscribers'] = df['Subscribers'].astype(int) #conveting into int datatype
    y = df['Subscribers'] #dependent variable
    
    X = df.drop('Subscribers', axis=1) #independent variables(removing dependant variable from independent variables)
    
    return X, y


# In[302]:


X, y = preprocess_inputs(data)


# In[303]:


X


# In[304]:


y


# In[305]:


sns.pairplot(data,x_vars=['Shares','Watch time (hours)'],y_vars='Subscribers',height=5,aspect=1,hue ='Subscribers',palette='Dark2')


# In[306]:


sns.pairplot(data,x_vars=['Likes (vs. dislikes) (%)','Impressions'],y_vars='Average percentage viewed (%)',height=5,aspect=1,hue ='Subscribers',palette='RdBu')


# #   TRAINING / VALIDATION

# In[307]:


rmses = []
r2s = []

kf = KFold(n_splits=3)

for train_idx, test_idx in kf.split(X):
    X_train = X.iloc[train_idx, :]
    X_test = X.iloc[test_idx, :]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]
    
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    rmse = np.sqrt(np.mean((y_test - y_pred)**2)) #root mean square error
    rmses.append(rmse)
    
    r2 = 1 - (np.sum((y_test - y_pred)**2) / np.sum((y_test - y_test.mean())**2))
    r2s.append(r2)


# ## RESULTS

# In[308]:


print("     RMSE: {:.2f}".format(np.mean(rmses)))
print("R^2 Score: {:.5f}".format(np.mean(r2s)))


# In[309]:


plt.figure(figsize=(10, 10))
plt.scatter(x=y_pred, y=y_test,color='red',alpha=0.7)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xlim(0, 10)
plt.ylim(0, 10)
plt.title("Actual vs. Predicted Values")
plt.show()


# In[ ]:




