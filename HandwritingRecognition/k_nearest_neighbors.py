#k_nearest_neighbors.py
"""
Grant White
October 24, 2019
"""

import numpy as np
from scipy import linalg as la
from math import inf
from scipy.spatial import KDTree
from scipy import stats


class KDTNode:
    """A node class for k-dimensional trees.
    Contains an k-dimentional value.
    """
    def __init__(self, x):
        """Constuct a new node with k-dimensional data x.
        x must be of type numpy.ndarray.
        """
        #check data type
        if type(x) is not np.ndarray:
            raise TypeError("Data must be a numpy.ndarray.")
        else:
            self.value = x

        #initialize left, right, and pivot
        self.left = None
        self.right = None
        self.pivot = None


class KDT:
    """A k-dimensional binary tree for solving the nearest neighbor problem.

    Attributes:
        root (KDTNode): the root node of the tree. Like all other nodes in
            the tree, the root has a NumPy array of shape (k,) as its value.
        k (int): the dimension of the data in the tree.
    """
    def __init__(self):
        """Initialize the root and k attributes."""
        self.root = None
        self.k = None

    def find(self, data):
        """Return the node containing the data. If there is no such node in
        the tree, or if the tree is empty, raise a ValueError.
        """
        def _step(current):
            """Recursively step through the tree until finding the node
            containing the data. If there is no such node, raise a ValueError.
            """
            if current is None:                     # Base case 1: dead end.
                raise ValueError(str(data) + " is not in the tree")
            elif np.allclose(data, current.value):
                return current                      # Base case 2: data found!
            elif data[current.pivot] < current.value[current.pivot]:
                return _step(current.left)          # Recursively search left.
            else:
                return _step(current.right)         # Recursively search right.

        # Start the recursive search at the root of the tree.
        return _step(self.root)


    def insert(self, data):
        """Insert a new node containing the specified data.

        Parameters:
            data ((k,) ndarray): a k-dimensional point to insert into the tree.

        Raises:
            ValueError: if data does not have the same dimensions as other
                values in the tree.
        """
        def _step(current_node, new_node):
            """Recursively step through the tree until finding where to
            insert the node.
            """
            #node already exists
            if np.allclose(data, current_node.value):
                raise ValueError("Node already exists in tree.")
            #move left
            elif new_node.value[current_node.pivot] < current_node.value[current_node.pivot]:
                #add node
                if current_node.left is None:
                    current_node.left = new_node
                    #set new pivot
                    new_node.pivot = (current_node.pivot + 1) % self.k
                #move left
                else:
                    return _step(current_node.left, new_node)
            #move right
            else:
                #add node
                if current_node.right is None:
                    current_node.right = new_node
                    #set new pivot
                    new_node.pivot = (current_node.pivot + 1) % self.k
                #move right
                else:
                    return _step(current_node.right, new_node)

        #add node to tree
        if self.root is None:    #empty tree
            self.root = KDTNode(data)
            self.k = len(data)
            self.root.pivot = 0
        elif len(data) != self.k:   #data not right dimension
            raise ValueError("Data is not the right dimention.")
        else:   #add node
            new_node = KDTNode(data)
            _step(self.root, new_node)


    def query(self, z):
        """Find the value in the tree that is nearest to z.

        Parameters:
            z ((k,) ndarray): a k-dimensional target point.

        Returns:
            ((k,) ndarray) the value in the tree that is nearest to z.
            (float) The Euclidean distance from the nearest neighbor to z.
        """
        def KDSearch(current, nearest, distance):
            if current is None:     #base case: dead end
                return nearest, distance
            x = current.value
            i = current.pivot
            if la.norm(x - z) < distance:    #check if current is closer to z than nearest
                nearest = current
                distance = la.norm(x - z)
            if z[i] < x[i]:     #search to the left
                nearest, distance = KDSearch(current.left, nearest, distance)
                if z[i] + distance >= x[i]:     #search to the right if needed
                    nearest, distance = KDSearch(current.right, nearest, distance)
            else:       #search to the right
                nearest, distance = KDSearch(current.right, nearest, distance)
                if z[i] - distance <= x[i]:     #search to the left if needed
                    nearest, distance = KDSearch(current.left, nearest, distance)
            return nearest, distance

        node, distance = KDSearch(self.root, self.root, la.norm(self.root.value - z))
        return node.value, distance


    def __str__(self):
        """String representation: a hierarchical list of nodes and their axes.

        Example:                           'KDT(k=2)
                    [5,5]                   [5 5]   pivot = 0
                    /   \                   [3 2]   pivot = 1
                [3,2]   [8,4]               [8 4]   pivot = 1
                    \       \               [2 6]   pivot = 0
                    [2,6]   [7,5]           [7 5]   pivot = 0'
        """
        #check if tree is empty
        if self.root is None:
            return "Empty KDT"

        #initialize
        nodes, strs = [self.root], []

        #get all nodes
        while nodes:
            current = nodes.pop(0)
            strs.append("{}\tpivot = {}".format(current.value, current.pivot))
            for child in [current.left, current.right]:
                if child:
                    nodes.append(child)

        return "KDT(k={})\n".format(self.k) + "\n".join(strs)


class KNeighborsClassifier:
    """Find the 'k' closest neighbors."""
    def __init__(self, n_neighbors):
        """Constructor.
        Accepts an integer n_neighbors: the number of neighbors to include in the vote.
        """
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        """Create tree and labels.
        Accepts an mxk NumPy array X: the training set.
        Accepts a NumPy array y with m entries: the training labels.
        """
        self.tree = KDTree(X)
        self.labels = y

    def predict(self, z):
        """Find the 'k' closest neighbors.
        Accepts a Numpy array z with k entries: point to compare with.
        """
        #initialize
        distances, indicies = self.tree.query(z, k=self.n_neighbors)

        if self.n_neighbors == 1:   #in case the given n is 1
            indicies = [indicies]
            
        return stats.mode(list(self.labels[i] for i in indicies))[0][0]
