#!/usr/bin/env python
# coding: utf-8

# In[40]:


import pandas as pd
import matplotlib as plt
import seaborn as sns
import numpy as np


# In[8]:


data = pd.read_csv('C:\\Users\\user\\Downloads\\Advertising.csv',index_col=0)
data


# In[9]:


data.head()


# In[10]:


data.Newspaper


# In[8]:


data.tail()


# In[10]:


data.describe()


# In[11]:


data.info()


# In[11]:


corrdata = data.corr()
corrdata                    # seeing the correlation 


# In[ ]:


# prefer the lower triangulare matrix excluding diagonal. Here Sale to tv is 0.78 , sale to radio is 0.57 and sale to newspaper is 0.228.


# In[14]:


sns.heatmap(corrdata)


# In[5]:


sns.pairplot(data,x_vars = ['','TV','Radio','Newspaper'], y_vars=['Sales'] , height=7)


# In[13]:


x= data[['TV','Radio','Newspaper']]
x


# In[14]:


y = data[['Sales']]
y


# In[15]:


from sklearn.model_selection import train_test_split


# In[16]:


X_train, X_test, Y_train, Y_test =train_test_split(x, y, random_state = 11)

# to divide data in 4 parts two for testing and two for training. 150 rows for training and remaining 50 for testing


# In[17]:


data.head(10)


# In[18]:


X_train


# In[19]:


Y_train


# In[20]:


print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)


# In[21]:


from sklearn.linear_model import LinearRegression


# In[22]:


lin = LinearRegression()   # y=mx+c[y=beta0+beta1(TV$)+beta2(radio$)+beta3(newspaper$)] so for multiple regression we need Y depends on x1,x2 and x3.


# In[23]:


lin.fit(X_train, Y_train)


# In[24]:


print(lin.intercept_)     # value of beta0


# In[25]:


print(lin.coef_)    # value of beta1,beta2 and beta3


# In[26]:


b0 = 3.444
b1 = 0.042
b2=0.194
b3=-0.009


# In[27]:


y_pred = lin.predict(X_test)


# In[28]:


y_pred = pd.DataFrame(y_pred)


# In[30]:


y_pred.head(10)


# In[34]:


Y_test.head(10)   # Actual Y


# In[ ]:


# we need predicted values minus actual to find erors take root mean square of that.


# In[35]:


from sklearn import metrics


# In[42]:


ME = metrics.mean_squared_error(Y_test,y_pred)
ME


# In[37]:


import math


# In[43]:


np.sqrt(ME)


# # OLS METHOD

# In[44]:


import statsmodels.api as st


# In[45]:


x= data[['TV','Radio','Newspaper']]
x


# In[46]:


y = data[['Sales']]
y


# In[47]:


model =st.OLS(y,x).fit()


# In[48]:


model.summary()


# In[ ]:


# from above we see that newspaper t value is very less and therefore its not helping us. for some values in newspaper the output will come negative.Also its error is high so we exclude newspaper


# In[49]:


x= data[['TV','Radio']]
y = data[['Sales']]
model =st.OLS(y,x).fit()
model.summary()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




