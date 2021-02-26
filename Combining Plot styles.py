#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from numpy.random import randn


# In[2]:


import scipy as stats


# In[3]:


import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


dataset = randn(100)


# In[5]:


#distplot helps us to combine plots
sns.distplot(dataset, bins=25)


# In[6]:


sns.distplot(dataset, bins=25, rug=True, hist=False)


# In[10]:


sns.distplot(dataset, bins=25,
            kde_kws={'color':'indianred','label':'KDE PLOT'},
            hist_kws={'color':'blue', 'label':'HIST'})


# In[11]:


from pandas import Series

ser1 = Series(dataset, name='My_data')

ser1


# In[12]:


sns.distplot(ser1, bins=25)


# In[ ]:





# In[37]:


# Let's create two distributions
data1 = randn(100)
data2 = randn(100) +2  # Off set the mean


# In[47]:


# Now we can create a box plot
sns.boxplot([data1, data2])


# In[42]:


# Notice how the previous plot had outlier points, we can include those with the "whiskers"
sns.boxplot([data1,data2],whis=np.inf)


# In[24]:


# WE can also set horizontal by setting vertical to false
sns.boxplot([data1,data2],whis=np.inf, vert = False)


# In[30]:


# Let's create an example where a box plot doesn't give the whole picture

# Normal Distribution
data1 = stats.norm(0,5).rvs(100)

# Two gamma distributions concatenated together (Second one is inverted)
data2 = np.concatenate([stats.gamma(5).rvs(50)-1,
                        -1*stats.gamma(5).rvs(50)])

# Box plot them
sns.boxplot([data1,data2],whis=np.inf)


# In[36]:


# From the above plots, you may think that the distributions are fairly similar
# But lets check out what a violin plot reveals
sns.violinplot(x=[data1, data2])


# In[32]:


# We can also change the bandwidth of the kernel used for the density fit of the violin plots if desired
sns.violinplot(data2,bw=0.01)


# In[33]:


# Much like a rug plot, we can also include the individual points, or sticks
sns.violinplot(data1,inner="stick")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




