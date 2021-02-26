#!/usr/bin/env python
# coding: utf-8

# ##INTRODUCTION
# 
# Lending Companies work by analyzing the financial history of their loan applicants, and choosing whether or not the applicant is too risky to be given a loan. If the applicant is not, the applicant then determines the terms of the loan. To acquire these applicants, companies can organically recieve them through their websites/apps, often with the help of advertisement campaigns.
# Other times lending companies partner with peer to peer(P2P) lending marketplaces, in order to acquire leads of possible applicants. Some example marketpleces include Upstart, Lending Tree and Lending Club.In this project we are going to assess the quality of the leads of our comapny recieves from these marketplaces.
# 
# 1. Market: The target audience is the set of loan applicants who reached out through an intermediary markerplace.
# 
# 2.Product: A loan.
# 
# 3.Goal: Develop a model to predict for quality applicants. In this case study, quality applicants are those who reach a key part of the loan application process.

# In this case study, we will be working for a fintech company that specializes on loans. It offers low APR loans to applicants based on theor financial habits, as almost all lending companies do. This company has partnered with a P2P lending marketplace that provides real time needs. The number of conversions from these leads are satisfactory.
# 
# The company tasks you with creating a model that predicts whether or not these leads will complete the electronic signaturephase of the loan application. The company seeks to leverage this model to identify less quality applicants(eg those who are not responding to the onboarding process), and experiment with giving them different boarding screens.

# In[25]:


import pandas as pd
from pandas import Series,DataFrame


# In[36]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

import random
import time

random.seed(100)


# In[27]:


dataset = pd.read_csv('P39-Financial-Data.csv')


# In[28]:


dataset.head()


# In[29]:


dataset.columns


# In[30]:


dataset.describe()


# In[31]:


#Cleaning the data
dataset.isna().any()




# In[32]:


dataset2 = dataset.drop(columns= ['entry_id', 'pay_schedule', 'e_signed'])


# In[33]:


fig = plt.figure(figsize=(15,12))

plt.suptitle('Histograms of Numerical Columns', fontsize = 20)
#Since we want to plot every feature in one single plot, we re going to iterate every feature
for i in range(dataset2.shape[1]):
    #This will iterate every feature.Shape gives the dimensions of the dataframe and he first item gives the number of columns
    #and since python does not include all the columns we are adding 1 to it. 
    plt.subplot(6,3,i+1)
    #Here in subplot, we are going to tell python number of images in the plot.'i' is given to tell what we are 
    #working on the moment
    f = plt.gca()
    #gca() command cleans up everything
    f.set_title(dataset2.columns.values[i])
    #It will title each feature
    
    vals = np.size(dataset2.iloc[:, i].unique())
    if vals >= 100:
        vals = 100
    #It will tell python how many bins will be in each feature. [:, i-1 ] will query the entire column
    
    plt.hist(dataset2.iloc[:, i], bins = vals, color= '#3F5D7D')
    
plt.tight_layout(rect=[0, 0.03, 1, 0.95])


# In[34]:


#CORRELATION PLOT

dataset2.corrwith(dataset.e_signed).plot.bar(
    figsize = (20,10), title = "Correlation with e_signed", fontsize = 15, rot = 45)


# In[ ]:





# In[35]:


sn.set(style="white", font_scale=2)

corr = dataset2.corr()

mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(18,15))
f.suptitle('Correlation Matrix', fontsize=40)

cmap = sn.diverging_palette(220,10, as_cmap=True)

sn.heatmap(corr, mask=mask, cmap=cmap, vmax=3, center=0,
          square=True, linewidth = .5, cbar_kws={"shrink": .5})


# In[ ]:


#Feature Engineeringabs


# In[37]:


dataset = dataset.drop(columns = ['months_employed'])


# In[40]:


dataset['personal_account_months'] = (dataset.personal_account_m + (dataset.personal_account_y * 12))


# In[42]:


dataset[['personal_account_m' , 'personal_account_y', 'personal_account_months' ]].head()


# In[55]:


dataset = dataset.drop(columns= ['personal_account_m','personal_account_y'])


# In[48]:


#One Hot Encoding

dataset = pd.get_dummies(dataset)


# In[49]:


dataset.columns


# In[51]:


dataset = dataset.drop(columns = ['pay_schedule_semi-monthly'])


# In[57]:


# Removing extra columns

response = dataset["e_signed"]

users = dataset["entry_id"]

dataset = dataset.drop(columns = ['e_signed','entry_id'])


# In[59]:


dataset2


# In[62]:


#Splitting into train - test split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dataset,
                                                   response,
                                                   test_size = 0.2,
                                                   random_state = 0)


# In[63]:


#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScalar()
X_train2 = pd.DataFrame(sc_X.fit_transform(X_train))
X_test2 = pd.DataFrame(sc_X.transform(X_test))
X_train2.columns = X_train.columns.values
X_test2.columns = X_test.columns.values
X_train2.index = X_train.index.values
X_test2.index = X_test.index.values
X_train = X_train2
X_test = X_test2


# In[64]:


#Logistic Regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0, penalty = 'l1')
classifier.fit(X_train, y_train)

#Predicting Test Set
y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score,recall_score, f1_score
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)


results = pd.DataFrame([['Linear Regression (lasso)', acc, prec, f1, rec]],
                      columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1_score'])


# In[65]:


#SVM(linear)
from sklearn.svm import SVC
classifier = SVC(random_state = 0, kernel = 'linear')
classifier.fit(X_train, y_train)

#Predicting Test Set
y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score,recall_score, f1_score
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)


model_results = pd.DataFrame([['SVM (linear)', acc, prec, f1, rec]],
                      columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1_score'])

results = results.append(model_results, ignore_index = True)


# In[ ]:


#Rbf kernel
from sklearn.svm import SVC
classifier = SVC(random_state = 0, kernel = 'rbf')
classifier.fit(X_train, y_train)

#Predicting Test Set
y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score,recall_score, f1_score
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)


model_results = pd.DataFrame([['SVM (rbf)', acc, prec, f1, rec]],
                      columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1_score'])

results = results.append(model_results, ignore_index = True)


# In[ ]:


results


# In[ ]:


#Random Forest
from sklearn.ensemble import RandomForestClassifier
classifier = SVC(random_state = 0, n_estimator=100,
                criterion='entropy')
classifier.fit(X_train, y_train)

#Predicting Test Set
y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score,recall_score, f1_score
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)


model_results = pd.DataFrame([['Random Forest (n=100)', acc, prec, f1, rec]],
                      columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1_score'])

results = results.append(model_results, ignore_index = True)

results


# In[ ]:


#K-Fold cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train,  y = y_train, cv = 10)

print("Random Forest Classifier Accuracy: %0.2f (+/-%0.2f)" %(accuracies.mean(), accuracies.std() * 2))

results


# In[ ]:


parameters = {"max_depth": [3, None],
              "max_features":[1,5,10],
             "min_samples_split":[2,5,10],
             "min_samples_leaf":[1,5,10],
             "bootstrap":[True, False],
             "criterion":["entropy"]}

from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(estimator=classifier,
                          param_grid = parameters,
                          scoring = "accuracy",
                          cv=10,
                          n_jobs = -1)

t0 = time.time()
grid_search = grid_search.fit(X_train, y_train)
t1 = time.time()
print("Took %0.2f seconds" % (t1 - t0))

rf_best_accuracy = grid_search.best_score_
rf__best_parameters = grid_search.best_params_
rf_best_accuracy, rf_best_parameters


# In[ ]:


#Grid Search 2 and Model Conclusion Left


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




