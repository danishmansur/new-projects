#!/usr/bin/env python
# coding: utf-8

# In[18]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import math
import matplotlib.pyplot as plt


# In[19]:


def dataset_reader(file_name):
    """
    @file: str, name of dataset file to read
    @does: read the dataset from the drive and convert it to numpy matrix
    @return: numpy matrix
    """
    
    return np.array(pd.read_csv(file_name, header=None), dtype=np.float64)


# In[20]:


df = pd.read_csv('housing.csv')
df.head()


# EDA

# In[21]:


df.describe()


# In[22]:


df.corr()
#You dont want to work with the data that has high collinearity. If the correlation is 70 % + or -,then i think you dont use 
#both of the features.You use either of the features.


# In[27]:


class LinearRegression:

    def __init__(self, X, y, learningrate, tolerance, maxIteration=50000, error = 'rmse', gd = False):
        self.X = X
        self.y = y
        self.learningrate = learningrate
        self.tolerance = tolerance
        self.maxIteration = maxIteration
        self.error = error      
        self.gd = gd
        #The parameters which are there do not belong to the class. In order for them to belong to the class, we add a self
        #to the class. 
        
        
        
    def splitToTrainTest(self):     
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, 
                                                            test_size = 0.3, 
                                                            random_state = 0)
        return X_train, X_test, y_train, y_test
        
    def add_x0(self, X):
        """
        @X: numpy matrix, dataset
        @does: add x0 to data
        @return: numpy matrix
        """
        return np.column_stack([np.ones([X.shape[0], 1]), X])   
    # we add bias so that i will not start from the origin. Also we add bias after splitting data into training and 
    #test dataset
    
    def normalize(self, X):
        """
        @X: numpy matrix, dataset
        @does: normalize X data using z-score and then add x0
        @return: numpy matrix, float, float
        """
        mean = np.mean(X, 0) # get mean of each column. Zero here is to show column.
        std = np.std(X, 0) # get std of each column
        X_norm = (X - mean) / std
        X_norm = self.add_x0(X_norm)
        return X_norm, mean, std 
    
    def normalizeTestData(self, X, train_mean, train_std, biasTerm = True):
        """
        @X: numpy matrix, dataset
        @does: normalize X testing data using mean and deviation of training data
        @return: numpy matrix
        """
        X_norm = (X - train_mean) / train_std 
        X_norm = self.add_x0(X_norm)
        return X_norm 
    
    #Which method to use. We can use gradient descent if the size is large than a threshold. We can use normal equation 
    #if the matrix is invertible. We can check if the matrix is full rank.If it is not full rank, we cannot invert 
    #Also if te number of records is less than 
    #features, then the matrix is low rank and does not have a unique solution. It will have multiple solution.
    
    def rank(self, X, eps=1e-12):
        u, s, vh = np.linalg.svd(X)
        return len([x for x in s if abs(x) > eps])
    
    #Whenever the matrix is square, we do eigen value decomposition, whenever the matrix is not square, we do singular value 
    #decomposition.

    def checkMatrix(self, X):
        X_rank = np.linalg.matrix_rank(X)
        if X_rank == min(X.shape[0], X.shape[1]):
            self.fullRank = True # solution will be unique
            print('data is fullrank')
        else:
            self.fullRank = False # Solution will not be unique
            print('data is not fullrank')
    
    def checkInvertibility(self, X):
        if X.shape[0] < X.shape[1]:
            self.lowRank = True
            print('data is lowrank')
        else:
            self.lowRank = False
            print('data is not lowrank')

    def closedFormSolution(self, X, y):
        """
        @X: numpy matrix, dataset
        @y: numpy array, output value
        @does: solve the regression using closed form solution
        @return: numpy array, float
        """
        w = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
        return w
    
    def gradient_descent(self, X, y):
        """
        @X: numpy matrix, dataset
        @w: numpy array, weights
        @y: numpy array, output value
        @learningrate: float, learning rate for cross validation
        @tolerance: float, threshold for the tolerance limit
        @maxIteration: int, maximum number of iteration for gradient descent
        @does: implement gradient descent to calculate w
        @return: numpy array, float
        """
        error_sequence = []

        last = float('inf')

        for i in tqdm(range(self.maxIteration)):
            self.w = self.w - self.learningrate * self.cost_derivatives(X, y)
            if self.error=='rmse':
                cur = self.rmse(X, y)
            else:
                cur = self.sse(X, y)
            diff = last - cur
            last = cur
            error_sequence.append(cur)
            if diff < self.tolerance:
                print("The model stopped - no further improvment")
                break

        self.plot_rmse(error_sequence)
        return

    def predict(self, X):
        """
        @X: numpy matrix, dataset
        @w: numpy array, weights
        @does: predict y_hat using X and w
        @return: numpy array
        """
        return X.dot(self.w)
    
    def sse(self, X, y):
        """
        @X: numpy matrix, dataset
        @w: numpy array, weights
        @y: numpy array, output value
        @does: sum of squared errors
        @return: float
        """
        y_hat = self.predict(X)
        return ((y_hat - y) ** 2).sum() # ||X^TQ-Y||2
    
    def rmse(self, X, y):
        """
        @X: numpy matrix, dataset
        @w: numpy array, weights
        @y: numpy array, output value
        @does: root mean squared error
        @return: float
        """
        return math.sqrt(self.sse(X, y) / y.size)   
    
    def cost_function(self, X, y):
        """
        @X: numpy matrix, dataset
        @w: numpy array, weights
        @y: numpy array, output value
        @does: cost function of regression
        @return: float
        """
        return self.sse(X, y) / 2
    
    def cost_derivatives(self, X, y):
        """
        @X: numpy matrix, dataset
        @w: numpy array, weights
        @y: numpy array, output value
        @does: derivative vector of the cost function
        @return: numpy array, float
        """
        y_hat = self.predict(X)
        return (y_hat - y).dot(X) # 2X^TX-2X^TY   
    
    def plot_rmse(self, error_sequence):
        """
        @X: error_sequence, vector of rmse
        @does: Plots the error function
        @return: plot
        """
        # Data for plotting
        s = np.array(error_sequence)
        t = np.arange(s.size)

        fig, ax = plt.subplots()
        ax.plot(t, s)

        ax.set(xlabel='iterations', ylabel=self.error,
               title='{} trend'.format(self.error))
        ax.grid()

        plt.legend(bbox_to_anchor=(1.05,1), loc=2, shadow=True)
        plt.show()
        
    # Run the model 
    def run_model(self):
        """
        @dataset: numpy matrix, dataset
        @learningrate: float, learning rate for cross validation
        @tolerance: float, threshold for the tolerance limit
        @folds: int, maximum number of folds
        @does: k fold cross validation
        @return: numpy array, float
        """
        self.X_train, self.X_test, self.y_train, self.y_test = self.splitToTrainTest()
        
        # Normalize the data
        self.X_train, self.mean, self.std = self.normalize(self.X_train)
        self.X_test = self.normalizeTestData(self.X_test, self.mean, self.std)

        self.checkMatrix(self.X_train)
        self.checkInvertibility(self.X_train)
        
        if self.fullRank and not self.lowRank and self.X_train.shape[0] < 10000 and not self.gd:
            # Have closed form solution
            print('Solving using closed form solution (Normal equation)')
            self.w = self.closedFormSolution(self.X_train, self.y_train)
        
        else: # Solve by gradient descent        
            # initiate the w 
            print('Solving using gradient descent')
            self.w = np.ones(self.X_train.shape[1], dtype=np.float64) * 0
            self.gradient_descent(self.X_train, self.y_train)

        print(self.w)

        if self.error == 'rmse':
            error_train = self.rmse(self.X_train, self.y_train)
            error_test = self.rmse(self.X_test, self.y_test)
        else:
            error_train = self.sse(self.X_train, self.y_train)
            error_test = self.sse(self.X_test, self.y_test)

        print('{} error for training data:'.format(self.error))
        print(error_train)

        print('{} error for testing data:'.format(self.error))
        print(error_test)


# In[28]:


regression = LinearRegression(df.values[:, 0:-1], df.values[:, -1],
                              learningrate=0.00001,
                              tolerance=0.0000001, 
                              gd = True, 
                             error='sse')


# In[29]:


regression.run_model()


# In[30]:


regression.predict(regression.X_test)


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





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




