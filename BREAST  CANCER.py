#!/usr/bin/env python
# coding: utf-8

# BREAST CANCER CLASSIFICATION
# 
# Predicting if the cancer is malignant or benign. 
# 30 features.
# 

# # IMPORTING DATA

# In[6]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[5]:


from sklearn.datasets import load_breast_cancer


# In[4]:


cancer = load_breast_cancer()


# In[7]:


cancer


# In[9]:


cancer.keys()


# In[10]:


print(cancer['DESCR'])


# In[11]:


print(cancer['target'])


# In[13]:


print(cancer['target_names'])


# In[14]:


print(cancer['feature_names'])


# In[15]:


cancer['data'].shape


# In[18]:


df_cancer = pd.DataFrame(np.c_[cancer['data'],cancer['target']],columns = np.append(cancer['feature_names'],['target']))


# In[19]:


df_cancer.head()


# In[ ]:





# # VISUALIZING DATA

# In[22]:


sns.pairplot(df_cancer, hue='target',vars = ['mean radius','mean texture','mean area','mean perimeter','mean smoothness'])


# In[24]:


sns.countplot(df_cancer['target'])


# In[25]:


sns.scatterplot(x= 'mean area', y = 'mean smoothness', hue ='target', data=df_cancer)


# In[29]:


plt.figure(figsize = (20,10))
sns.heatmap(df_cancer.corr(), annot = True)


# In[ ]:





# # MODEL TRAINING

# In[30]:


X = df_cancer.drop(['target'], axis =1)


# In[31]:


X


# In[32]:


y = df_cancer['target']


# In[33]:


y


# In[35]:


from sklearn.model_selection import train_test_split


# In[36]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=5)


# In[37]:


X_train


# In[38]:


X_test


# In[39]:


y_train


# In[40]:


y_test


# In[41]:


from sklearn.svm import SVC


# In[42]:


from sklearn.metrics import classification_report, confusion_matrix


# In[43]:


svc_model = SVC()


# In[44]:


svc_model.fit(X_train, y_train)


# In[ ]:





# # EVALUATING THE MODEL

# In[45]:


y_predict = svc_model.predict(X_test)


# In[46]:


y_predict


# In[47]:


cm = confusion_matrix(y_test, y_predict)


# In[48]:


sns.heatmap(cm, annot = True)


# # IMPROVING THE MODEL

# In[49]:


min_train = X_train.min()


# In[50]:


range_train = (X_train - min_train).max()


# In[51]:


X_train_scaled = (X_train - min_train)/range_train


# In[53]:


sns.scatterplot(x=X_train['mean area'], y=X_train['mean smoothness'], hue = y_train)


# In[55]:


min_test = X_test.min()
range_test = (X_test - min_test).max()
X_test_scaled = (X_test - min_test)/range_test


# In[57]:


svc_model.fit(X_train_scaled, y_train)


# In[58]:


y_predict = svc_model.predict(X_test_scaled)


# In[ ]:


cm = confusion_matrix(y_test, y_predict)


# In[59]:


sns.heatmap(cm, annot = True)


# In[60]:


print(classification_report(y_test, y_predict))


# In[ ]:





# # IMPROVING MODEL PART 2

# In[61]:


param_grid = {'C':[0.1,1,10,100], 'gamma': [1, 0.1, 0.01, 0.001 ], 'kernel': ['rbf'] }


# In[62]:


from sklearn.model_selection import GridSearchCV


# In[63]:


grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 4)


# In[65]:


grid.fit(X_train_scaled, y_train)


# In[66]:


grid.best_params_


# In[67]:


grid_predictions = grid.predict(X_test_scaled)


# In[68]:


cm = confusion_matrix(y_test, grid_predictions)


# In[69]:


sns.heatmap(cm, annot=True)


# In[70]:


print(classification_report(y_test, grid_predictions))


# In[ ]:





# In[ ]:





# In[ ]:




