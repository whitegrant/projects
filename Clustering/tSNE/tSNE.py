"""
Grant White
11/30/20

This program compares PCA to tSNE on the MNIST Fashion dataset.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from keras.datasets import fashion_mnist
from matplotlib import pyplot as plt

#load data
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

#subset of data (dataset is very large)
num = 6000
idx = np.random.choice(np.arange(len(x_train)), num, replace=False)
fash_X = x_train[idx]
fash_Y = y_train[idx]

#reshape data
input_dim = 784    #28*28
fash_X = fash_X.reshape(num, input_dim)
fash_X = fash_X/255

#PCA
fash_pca = PCA(2).fit_transform(fash_X)

#t-SNE
fash_tsne = TSNE().fit_transform(fash_X)

#plot
classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
           'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
plt.subplot(121)    #PCA plot
scat1 = plt.scatter(fash_pca[:,0], fash_pca[:,1], s=2, c=fash_Y)
plt.legend(loc=4, prop={'size': 5}, handles=scat1.legend_elements()[0], labels=classes)
plt.title('PCA')
plt.axis('off')
plt.subplot(122)    #tSNE plot
scat2 = plt.scatter(fash_tsne[:,0], fash_tsne[:,1], s=2, c=fash_Y)
plt.title('tSNE')
plt.axis('off')
plt.show()
