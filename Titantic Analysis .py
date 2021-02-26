#!/usr/bin/env python
# coding: utf-8

# Titanic Dataset 
# 
# In this Dataset we will analyse the Titanic Dataset.

# In[3]:


#Using Pandas to open the Dataset

import pandas as pd
from pandas import Series, DataFrame

#Set up Titanic csv file as a DataFrame
titanic_df = pd.read_csv('train.csv')


# In[4]:


#Taking a preview of the data
titanic_df.head()


# In[5]:


titanic_df.info()


# Looking at the dataset, we will try to answer the these questions:
# 
# 
# 1.) Who were the passengers on the Titanic? (Ages,Gender,Class,..etc)
# 2.) What deck were the passengers on and how does that relate to their class?
# 3.) Where did the passengers come from?
# 4.) Who was alone and who was with family?
# 
# Then we'll dig deeper, with a broader question:
# 
# 5.) What factors helped someone survive the sinking?

# In[6]:


# Let's import what we'll need for the analysis and visualization
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[15]:


# Let's first check gender
sns.catplot(x='Sex',kind='count',data=titanic_df)


# Looing at the Bar Plot of Male and Female, we can see that there are half as many females as males. 

# In[17]:


# Now let's seperate the genders by classes, remember we can use the 'hue' arguement here!
sns.catplot(x='Pclass',kind='count',data=titanic_df,hue='Sex')


# Visualizing the male and female ratio by class, we can see that there are more males in the third class and the ratio of male-female is not proportional in third class than in 1st and 2nd class.
# 
# It would be interesting to split the into male, female, and children. We will do this by considering anyone under 16 as children and apply a function to create a new column.

# In[18]:


def male_female_child(passenger):
    #Take the Age and Sex
    age,sex = passenger
    #Compare the age, otherwise leave the sex
    if age < 16:
        return 'child'
    else:
        return sex
    
#We will define a new column called 'person', remember to specify axis=1 for columns and not index

titanic_df['person'] = titanic_df[['Age','Sex']].apply(male_female_child, axis =1)


# In[19]:


titanic_df[0:10]


# In[20]:


# Let's try the factorplot again!
sns.catplot(x='Pclass', kind='count',data=titanic_df,hue='person')


# In[27]:


# Getting a distribution of ages
titanic_df['Age'].hist(bins=70)


# In[29]:


#Getting a quick comparison of male, female, child
titanic_df['person'].value_counts()


# In[33]:


fig = sns.FacetGrid(titanic_df, hue='Sex', aspect=4)

fig.map(sns.kdeplot,'Age',shade=True)

# Set the x max limit by the oldest passenger
oldest = titanic_df['Age'].max()

#Since we know no one can be negative years old set the x lower limit at 0
fig.set(xlim=(0,oldest))

#Finally add a legend
fig.add_legend()


# In[35]:


fig = sns.FacetGrid(titanic_df, hue='person', aspect=4)

fig.map(sns.kdeplot,'Age',shade=True)

# Set the x max limit by the oldest passenger
oldest = titanic_df['Age'].max()

#Since we know no one can be negative years old set the x lower limit at 0
fig.set(xlim=(0,oldest))

#Finally add a legend
fig.add_legend()


# In[36]:


fig = sns.FacetGrid(titanic_df, hue='Pclass', aspect=4)

fig.map(sns.kdeplot,'Age',shade=True)

# Set the x max limit by the oldest passenger
oldest = titanic_df['Age'].max()

#Since we know no one can be negative years old set the x lower limit at 0
fig.set(xlim=(0,oldest))

#Finally add a legend
fig.add_legend()


# In[38]:


titanic_df.head()


# In[39]:


# First we'll drop the NaN values and create a new object, deck
deck = titanic_df['Cabin'].dropna()


# In[40]:


deck.head()


# In[45]:


#We will try to grab that letter with a simple for loop

#Set empty list
levels = []

#Loop to grab first letter
for level in deck:
    levels.append(level[0])
    
#Reset DataFrame and use factor plot
cabin_df = DataFrame(levels)
cabin_df.columns = ['Cabin']
sns.catplot(x='Cabin',kind='count',data=cabin_df,palette='winter_d')


# In[46]:


# Redefine cabin_df as everything but where the row was equal to 'T'
cabin_df = cabin_df[cabin_df.Cabin != 'T']
#Replot
sns.catplot(x='Cabin',kind='count',data=cabin_df,palette='summer')


# Now after analysing the deck, we will see where did the passengers come from

# In[49]:


sns.catplot(x='Embarked',kind='count',data=titanic_df,hue='Pclass',row_order=['C','Q','S'])


# An interesting find here is that in Queenstown, almost all the passengers that boarded there were 3rd class. It would be intersting to look at the economics of that town in that time period for further investigation.
# 
# Now let's take a look at the 4th question:
# 
# 4.) Who was alone and who was with family?

# In[50]:


# Let's start by adding a new column to define alone

# We'll add the parent/child column with the sibsp column
titanic_df['Alone'] =  titanic_df.Parch + titanic_df.SibSp
titanic_df['Alone']


# In[54]:


# Look for >0 or ==0 to set alone status
titanic_df['Alone'].loc[titanic_df['Alone'] >0] = 'With Family'
titanic_df['Alone'].loc[titanic_df['Alone'] == 0] = 'Alone'

# Note it's okay to ignore an  error that sometimes pops up here. For more info check out this link
url_info = 'http://stackoverflow.com/questions/20625582/how-to-deal-with-this-pandas-warning'


# In[55]:


titanic_df.head()


# In[56]:


# Now let's get a simple visualization!
sns.catplot(x='Alone',kind='count',data=titanic_df,palette='Blues')


# In[57]:


# Let's start by creating a new column for legibility purposes through mapping (Lec 36)
titanic_df["Survivor"] = titanic_df.Survived.map({0: "no", 1: "yes"})

# Let's just get a quick overall view of survied vs died. 
sns.catplot(x='Survivor',kind='count',data=titanic_df,palette='Set1')


# Looking at the graph, it looks that more people died than survived. The ratio is almost double.

# In[65]:


# Let's use a factor plot again, but now considering class
sns.factorplot(x='Pclass',y='Survived',data=titanic_df)


# In[68]:


# Let's use a factor plot again, but now considering class and gender
sns.factorplot(x='Pclass',y='Survived',hue='person',data=titanic_df)


# In[94]:


# Let's use a linear plot on age versus survival
sns.lmplot('Age','Survived',data=titanic_df)


# In[95]:


# Let's use a linear plot on age versus survival
sns.lineplot('Age','Survived',data=titanic_df)


# In[98]:


# Let's use a linear plot on age versus survival using hue for class seperation
generations=[10,20,40,60,80]
sns.lmplot('Age','Survived',hue='Pclass',data=titanic_df,palette='winter',x_bins=generations)


# In[102]:


sns.lmplot('Age','Survived',hue='Sex',data=titanic_df,palette='winter',x_bins=generations)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




