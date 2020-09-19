#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[9]:


dataset = pd.read_csv('D:\student_scores.csv')


# In[10]:


dataset.shape


# In[11]:


dataset.head()


# In[12]:


dataset.describe()


# In[14]:


dataset.plot(x='Hours',y='Scores',style='o')
plt.title('hours vs percentage')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
plt.show()


# In[19]:


X=dataset.iloc[:, :-1].values
Y=dataset.iloc[:, 1].values

 from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# In[21]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# In[22]:


print(regressor.intercept_)


# In[23]:


print(regressor.coef_)


# In[24]:


y_pred = regressor.predict(X_test)


# In[25]:


df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df


# In[26]:


from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[31]:


hours = float(input(' Enter the hours of study:- '))
predicted = regressor.predict([[hours]])
print(" If a student studies for = {}".format(hours),"hours, then his/her predicted score is = {}%".format(predicted))


# 

# In[ ]:





# In[ ]:




