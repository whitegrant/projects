#handwriting_recognition.py
"""
Grant White
October 24, 2019
"""

import numpy as np
from k_nearest_neighbors import KNeighborsClassifier


def predict(n_neighbors, filename="mnist_subset.npz"):
    """Extract the data from the given file. Load a KNeighborsClassifier with
    the training data and the corresponding labels. Use the classifier to
    predict labels for the test data. Return the classification accuracy, the
    percentage of predictions that match the test labels.

    Parameters:
        n_neighbors (int): the number of neighbors to use for classification.
        filename (str): the name of the data file. Should be an npz file with
            keys 'X_train', 'y_train', 'X_test', and 'y_test'.

    Returns:
        (float): the classification accuracy.
    """
    #initialize
    data = np.load("mnist_subset.npz")
    X_train = data["X_train"].astype(np.float)      #Training data
    y_train = data["y_train"]                       #Training labels
    X_test = data["X_test"].astype(np.float)        #Test data
    y_test = data["y_test"]                         #Test labels

    #train and test
    hand_writting = KNeighborsClassifier(n_neighbors)
    hand_writting.fit(X_train, y_train)
    predictions = [hand_writting.predict(i) for i in X_test]

    #return accuracy
    count = 0
    for i in range(len(y_test)):
        if y_test[i] == predictions[i]:
            count += 1

    return count / len(predictions)
