#PCA.py
"""
Grant White
11/15/20

Principle Component Analysis.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from keras.datasets import fashion_mnist
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import random


class my_PCA:
    """
    Principal Component Analysis
    """
    def __init__(self, X, s=None):
        """
        X : (n,d) array, data points
                rows are data points
                columns are features
        s : (int) optional, number of principle components to compute
        """
        #save variables
        self.X = X
        if s is None:
            self.s = len(X[0])    #d
        else:
            self.s = s
            
        #variance (SVD)
        _, Sig, Vt = np.linalg.svd(X, full_matricies=False)
        self.var = Sig[:s]**2
        self.Vt = Vt[:s]
        
    def transform(self, x):
        """
        returns principal components
        """
        return x @ self.Vt.T
    
    def project(self, x):
        """
        returns data points projected onto the principal axes
        """
        a = self.transform(x)
        return a @ self.Vt


def MCA(Y):
    """
    Multiple Component Analysis.
    Takes a categorical dataset and performs MCA on it so that PCA can properly
    be used on it.
    Parameters:
        Y : (n,d) array, categorical data points
            rows are data points
            columns are features
    """
    return Y / np.mean(Y, axis=0) - 1


def scree_bc():
    """
    Plots the scree plot of the amount of variance explained by each component
    of the breast cancer dataset after principle component analysis.
    """
    #get data
    BC = load_breast_cancer()
    BC_X = BC.data
    BC_y = BC.target

    #center data
    BC_X -= np.mean(BC_X, axis=0)

    #PCA
    BC_n = len(BC_X[0])
    BC_pca = PCA(BC_n)
    BC_pca.fit(BC_X)
    BC_var = BC_pca.explained_variance_ratio_

    #plot
    plt.plot(BC_var, label='variance explained by each component')
    plt.plot(np.cumsum(BC_var), label='cumulative variance explained')
    plt.title('Variance Explained by Components')
    plt.xlabel('number of components')
    plt.ylabel('percent of variance explained')
    plt.legend()
    plt.show()


def scree_fash():
    """
    Plots the scree plot of the amount of variance explained by each component
    of the fashion dataset after principle component analysis.
    """
    #get data
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    #format data
    input_dim = 784    #28*28
    x_train = x_train.reshape(60000, input_dim)
    x_test = x_test.reshape(10000, input_dim)
    x_train = x_train/255
    x_test = x_test/255
    
    #center data
    mu = np.mean(x_train, axis=0)
    x_train -= mu
    x_test -= mu

    #PCA
    fashion_pca = PCA(784)
    fashion_pca.fit(x_train)
    var = fashion_pca.explained_variance_ratio_

    #plot
    plt.plot(var, label='variance explained by each component')
    plt.plot(np.cumsum(var), label='cumulative variance explained')
    plt.title('Variance Explained by Components')
    plt.xlabel('number of components')
    plt.ylabel('percent of variance explained')
    plt.legend()
    plt.show()


def plot_2_bc():
    """
    Plots the first two components of PCA of the breast cancer dataset. As you
    can see, just using two components does a good job at separating the
    different classes.
    """
    #get data
    BC = load_breast_cancer()
    BC_X = BC.data
    BC_y = BC.target

    #center data
    BC_X -= np.mean(BC_X, axis=0)

    #PCA
    BC_n = len(BC_X[0])
    BC_pca = PCA(BC_n)
    BC_pca.fit(BC_X)

    #dimension reduction
    BC_points = BC_pca.transform(BC_X)

    #plot
    plt.scatter(BC_points[:,0], BC_points[:,1], s=1, c=BC_y)
    plt.title('Breast Cancer')
    plt.xlabel('first component')
    plt.ylabel('second component')
    plt.show()


def plot_2_fash():
    """
    Plots the first two components of PCA of the fashion dataset. As you can
    see, just using two components does a good job at separating the different
    classes.

    """
    #get data
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    #format data
    input_dim = 784    #28*28
    x_train = x_train.reshape(60000, input_dim)
    x_test = x_test.reshape(10000, input_dim)
    x_train = x_train/255
    x_test = x_test/255
    
    #center data
    mu = np.mean(x_train, axis=0)
    x_train -= mu
    x_test -= mu

    #PCA
    fashion_pca = PCA(784)
    fashion_pca.fit(x_train)

    #dimension reduction
    fashion_points = fashion_pca.transform(x_train)

    #plot
    plt.scatter(fashion_points[:,0], fashion_points[:,1], s=0.5, c=y_train)
    plt.title('Clothing')
    plt.xlabel('first component')
    plt.ylabel('second component')
    plt.show()


def show_90():
    """
    Finds the minimum amount of principle components necessary to account for
    90% of the variance. Orthogonally projects the data onto the subspace
    spanned by those principal axes. Plots a random selection of 20 of the
    resulting images along with the original for the fashion dataset.
    """
    

    #get data
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    #format data
    input_dim = 784    #28*28
    x_train = x_train.reshape(60000, input_dim)
    x_test = x_test.reshape(10000, input_dim)
    x_train = x_train/255
    x_test = x_test/255

    #center data
    mu = np.mean(x_train, axis=0)
    x_train -= mu
    x_test -= mu

    #PCA
    fashion_pca = PCA(784)
    fashion_pca.fit(x_train)
    var = fashion_pca.explained_variance_ratio_

    #find how many components are needed to show 90% of the variance
    cum_var = np.cumsum(var)
    indx = 0
    for i, val in enumerate(cum_var):
        if val > .9:
            indx = i
            break
            
    #PCA
    fashion_pca = PCA(indx)
    fashion_pca.fit(x_train)

    for j in range(10):
        i = random.randrange(60000)
        img = x_train[i]
        proj = fashion_pca.transform([img])
        new_img = fashion_pca.inverse_transform(proj)
        
        #un-center
        img += mu
        new_img += mu
                
        ax1 = plt.subplot(5,4,1+(2*j))
        ax1.imshow(img.reshape(28,28), cmap='Greys')
        ax1.set_axis_off()
        ax1.set_title('Original', fontsize=4)
        
        ax2 = plt.subplot(5,4,2+(2*j))
        ax2.imshow(new_img.reshape(28,28), cmap='Greys')
        ax2.set_axis_off()
        ax2.set_title('{} of 784 Components'.format(indx), fontsize=4)
    
    plt.tight_layout()
    plt.show()
