#!/usr/bin/env python
# coding: utf-8

# ## Multiple Linear Regression

# Bağımlı değişken bu kez 1 ' den fazla girdiye bağlı. Bu değişken katsayılarını yine RSS kullanarak bulacağız.OLS kullanacağız bunun için statsmodel kullanacağız bunun için.Datamız yine advertisinig datası olacak
# 

# In[1]:


#Kütüphaneleri import edelim
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[4]:


#Datayı import edelim
df=pd.read_csv("Advertising.csv",index_col=0)


# In[5]:


df.head()


# In[6]:


X=df["TV"]
y=df["sales"]


# In[7]:


# İşlem yapabilmek için x ve y ' yi boyutlandıralım'
X=X.values.reshape(-1,1)
y=y.values.reshape(-1,1)


# In[8]:


#Datayı test train olarak ayırabilmek için model kütüphanesini import edelim
from sklearn.model_selection import train_test_split


# In[10]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=100)


# In[11]:


import statsmodels.api as sm


# In[16]:


X_train_ols=sm.add_constant(X_train)
# Bo lar da hesaplansın diye katsayı verdik OLS hesaplamasında bu gerekli


# In[15]:


X_train_ols


# In[17]:


sm_model=sm.OLS(y_train,X_train_ols)
# OLS önce y değerini alır


# In[19]:


sonuc=sm_model.fit()


# In[20]:


print(sonuc.summary())


# In[22]:


# R2 doğruluk değeri yani TV değişkeni tek başına modeli %61 oranında açıklıyor.
#X1 std error yani her kat sayıda 0.003 hata oluyor demek 
#t dağılımı ile bulduğumuz katsayı hatamızın kaç katı diye kontrol ederiz
#p de bize hangi değişkenin ne kadar önemli olduğunu  verecek 0.05 den büyük olursa o değişkeni atabileceğiz


# In[ ]:




