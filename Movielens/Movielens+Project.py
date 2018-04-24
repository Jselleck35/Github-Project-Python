
# coding: utf-8

# In[10]:


import pandas as pd
import numpy as np


# In[20]:


movies_dataset = pd.read_table('C:\\Data Science with Python\\Project\\movies.dat', sep='::',
                               names=['MovieID','Title', 'Genres'])


# In[22]:


movies_dataset.shape


# In[23]:


movies_dataset.head()


# In[24]:


users_dataset = pd.read_table('C:\\Data Science with Python\\Project\\users.dat', sep='::',
                              names=['UserID', 'Gender','Age','Occupation', 'Zip-code'])


# In[25]:


users_dataset.head()


# In[28]:


ratings_dataset = pd.read_table('C:\\Data Science with Python\\Project\\ratings.dat', sep='::',
                                names=['UserID', 'MovieID', 'Rating', 'Timestamp'])


# In[29]:


ratings_dataset.head()


# In[37]:


import matplotlib.pyplot as plt
from matplotlib import style
get_ipython().magic(u'matplotlib inline')
import seaborn as sns


# In[181]:


# show age distribution
sns.set()
plt.hist(users_dataset['Age'], bins=12)
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Age Distribution of Users')
plt.show


# In[149]:


# show rating distribution
plt.hist(ratings_dataset['Rating'])
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.title('Overall Rating by Users')
plt.show


# In[151]:


df_movieID=pd.merge(ratings_dataset,movies_dataset, on='MovieID')


# In[155]:


df_toystory = df_movieID[df_movieID['Title']=='Toy Story (1995)']
df_toystory


# In[159]:


# show user rating of toy story
plt.hist(df_toystory['Rating'])
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.title('User Ratings of Toy Story')
plt.show


# In[157]:


df_toystoryusers=pd.merge(df_toystory,users_dataset, on='UserID')


# In[161]:


# show age viewership of toystory
plt.hist(df_toystoryusers['Age'], bins=12)
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Toy Story viewership by Age')
plt.show


# In[168]:


df_all = pd.merge(df_movieID, users_dataset)


# In[163]:


# shows the top 25 movies by viewership rating
Movieviews = df_movieID.groupby('Title').size().sort_values(ascending=False).head(25)
Movieviews


# In[123]:


usergrp = ratings_dataset.groupby('UserID')
user2696 = usergrp.get_group(2696)
user2696


# In[164]:


# show rating data by user 2696
plt.hist(user2696['Rating'])
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.title('Rating data of User 2696')
plt.show


# In[197]:


df_userandrating=pd.merge(ratings_dataset,users_dataset, on='UserID').head(500)
df_userandrating


# In[198]:


x_feature = df_userandrating[['MovieID','Occupation','Age']]


# In[199]:


x_feature.head()


# In[200]:


y_target = df_userandrating[['Rating']]


# In[201]:


x_feature.shape


# In[202]:


y_target.shape


# In[203]:


from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_feature, y_target,random_state=1)


# In[204]:


print x_train.shape
print y_train.shape
print x_test.shape
print y_test.shape


# In[205]:


from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(x_train,y_train)


# In[223]:


print linreg.intercept_
print linreg.coef_


# In[207]:


y_pred = linreg.predict(x_test)
y_pred


# In[208]:


from sklearn import metrics


# In[209]:


print 'MSE value is %.2f '% np.sqrt(metrics.mean_squared_error(y_test,y_pred))


# In[233]:


x_test.hist()
plt.show()


# In[230]:


x_train.hist()
plt.show()

