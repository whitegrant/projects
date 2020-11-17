#random_forest.py
"""
Grant White
11/11/20

Implements a random forest by implementing multiple decision trees. It also
creates visualizeations of the individual decision trees, This program is not
optimized for temporal complexity. This can be seen by comparing computation
time between this implementation and scikit-learns's implementation of a random
forest.
"""

import numpy as np
import graphviz
import os
from uuid import uuid4
from time import time
from sklearn.ensemble import RandomForestClassifier


class Question:
    """
    Questions to use in construction and display of Decision Trees.
        Attributes:
            column (int): which column of the data this question asks
            value (int/float): value the question asks about
            feature (str): name of the feature asked about
        Methods:
            match: returns boolean of if a given sample answered T/F
    """
    def __init__(self, column, value, feature_list):
        self.column = column
        self.value = value
        self.feature = feature_list[self.column]
    
    def match(self,sample):
        """
        Returns T/F depending on how the sample answers the question.
            Parameters:
                sample ((n,), ndarray): new sample to classify
            Returns:
                (bool): how the sample compares to the question
        """
        return sample[self.column] >= self.value
        
    def __repr__(self):
        """
        Returns a representation of the question.
        """
        return "Is %s >= %s?" % (self.feature, str(self.value))


def partition(data,question):
    """
    Splits the data into left (true) and right (false)
        Parameters:
            data ((m,n), ndarray): data to partition
            question (Question): question to split on
        Returns:
            left ((j,n), ndarray): portion of the data matching the question
            right ((m-j, n), ndarray): portion of the data NOT matching the question
    """
    #initialize
    left = []
    right = []

    #partition data
    for point in data:
        if question.match(point):
            left.append(point)
        else:
            right.append(point)

    #check if empty
    if len(left) == 0:
        left = None
    else:
        left = np.array(left)
    if len(right) == 0:
        right = None
    else:
        right = np.array(right)

    return left, right
    

def gini(data):
    """
    Returns the Gini impurity of given array of data.
        Parameters:
            data (ndarray): data to examine
        Returns:
            (float): Gini impurity of the data
    """
    #initialize
    N = len(data)

    #single column
    if len(data.shape) == 1:
        f = np.unique(data, return_counts=True)[1] / N
    #multiple columns
    else:
        f = np.unique(data[:,-1], return_counts=True)[1] / N
    
    return 1 - sum(f**2)


def info_gain(left,right,old_info):
    """
    Returns the info gain of a partition of data.
        Parameters:
            left (ndarray): left split of data
            right (ndarray): right split of data
            old (float): Gini impurity of unsplit data
        Returns:
            (float): info gain of the data
    """
    #number of samples
    Nl = len(left)
    Nr = len(right)
    N = Nl + Nr

    return old_info - Nl/N*gini(left) - Nr/N*gini(right)
    

def find_best_split(data, feature_labels, min_samples_leaf=5):
    """
    Finds the optimal split.
        Parameters:
            data (ndarray): data in question
            feature_labels (list of strings): labels for each column of data
            min_samples_leaf (int): minimum number of samples per leaf
        Returns:
            (float): best info gain
            (Question): best question
    """
    #initialize
    max_info = 0
    best_question = None
    
    #iterate through the features
    for i in range(len(feature_labels[:-1])):
        #get info before splitting
        old_info = gini(data)

        #iterate through posible questions based on that feature
        unique = np.unique(data[:,i])
        for val in unique[1:]:
            question = Question(i, val, feature_labels)
            left, right = partition(data, question)

            #check to see if partitions are None
            if left is not None and right is not None:
                #check minimum samples in a leaf
                if len(left) >= min_samples_leaf and len(right) >= min_samples_leaf:
                    info = info_gain(left, right, old_info)

                    #check to see if it's the best
                    if info > max_info:
                        max_info = info
                        best_question = question

    return max_info, best_question


class Leaf:
    """
    Tree leaf node.
        Attribute:
            prediction (dict): dictionary of labels at the leaf
    """
    def __init__(self,data):
        #initialize
        self.prediction = {}

        #get unique labels
        labels, counts = np.unique(data[:,-1], return_counts=True)

        #make dictionary
        for label, count in zip(labels, counts):
            self.prediction[label] = count


class Decision_Node:
    """
    Tree node with a question.
        Attributes:
            question (Question): question associated with node
            left (Decision_Node or Leaf): child branch
            right (Decision_Node or Leaf): child branch
    """
    def __init__(self, question, left_branch, right_branch):
        self.question = question
        self.left = left_branch
        self.right = right_branch


def draw_node(graph, my_tree):
    """
    Helper function for draw_tree.
    """
    node_id = uuid4().hex
    #If it's a leaf, draw an oval and label with the prediction
    if isinstance(my_tree, Leaf):
        graph.node(node_id, shape="oval", label="%s" % my_tree.prediction)
        return node_id
    else: #If it's not a leaf, make a question box
        graph.node(node_id, shape="box", label="%s" % my_tree.question)
        left_id = draw_node(graph, my_tree.left)
        graph.edge(node_id, left_id, label="T")
        right_id = draw_node(graph, my_tree.right)    
        graph.edge(node_id, right_id, label="F")
        return node_id


def draw_tree(my_tree):
    """
    Creates a visualization for a tree.
        Parameters:
            my_tree (Decision_Node (or Leaf)): root of tree
    """
    #remove the files if they already exist
    for file in ['Digraph.gv','Digraph.gv.pdf']:
        if os.path.exists(file):
            os.remove(file)

    #display tree
    graph = graphviz.Digraph(comment="Decision Tree")
    draw_node(graph, my_tree)
    graph.render(view=True) #this saves Digraph.gv and Digraph.gv.pdf


def build_tree(data, feature_names, min_samples_leaf=5, max_depth=4, current_depth=0, random_subset=False):
    """
    Builds a classification tree using the classes Decision_Node and Leaf.
        Parameters:
            data (ndarray): data to build tree from
            feature_names(list or array)
            min_samples_leaf (int): minimum allowed number of samples per leaf
            max_depth (int): maximum allowed depth
            current_depth (int): depth counter
            random_subset (bool): whether or not to train on a random subset of features
        Returns:
            Decision_Node (or Leaf)
    """
    #choose features randomly
    if random_subset:
        #random indicies
        N = len(feature_names) - 1
        n = int(np.sqrt(N))
        indx = np.random.choice(np.arange(N), n, replace=False)
        indx = np.append(indx, -1)    #class label column

        #random subsets
        feature_names = feature_names[indx]
        data = data[:,indx]

    #check depth
    if current_depth == max_depth:    #leaf
        return Leaf(data)
    else:    #split
        info, question = find_best_split(data, feature_names, min_samples_leaf)

    #check split
    if info == 0:    #leaf
        return Leaf(data)
    else:    #decision node
        left, right = partition(data, question)
        return Decision_Node(question,
            build_tree(left, feature_names, min_samples_leaf, max_depth, current_depth + 1),
            build_tree(right, feature_names, min_samples_leaf, max_depth, current_depth + 1))


def predict_tree(sample, my_tree):
    """
    Predicts the label for a sample given a pre-made decision tree.
        Parameters:
            sample (ndarray): a single sample
            my_tree (Decision_Node or Leaf): a decision tree
        Returns:
            Label to be assigned to new sample
    """
    #check to see if at leaf or not
    if type(my_tree) == Leaf:    #reached leaf node
        return max(my_tree.prediction, key=my_tree.prediction.get)
    else:    #continue down tree
        if my_tree.question.match(sample):
            return predict_tree(sample, my_tree.left)
        else:
            return predict_tree(sample, my_tree.right)


def analyze_tree(dataset,my_tree):
    """
    Tests how accurately a tree classifies a dataset.
        Parameters:
            dataset (ndarray): labeled data with the labels in the last column
            tree (Decision_Node or Leaf): a decision tree
        Returns:
            (float): proportion of dataset classified correctly
    """
    #initialize
    num_correct = 0
    N = len(dataset)

    #test each data point
    for point in dataset:
        if predict_tree(point, my_tree) == point[-1]:
            num_correct += 1

    return num_correct / N


def predict_forest(sample, forest):
    """
    Predicts the label for a new sample, given a random forest.
        Parameters:
            sample (ndarray): a single sample
            forest (list): a list of decision trees
        Returns:
            label to be assigned to new sample
    """
    #initialize
    labels = {}

    #predict each tree
    for tree in forest:
        label = predict_tree(sample, tree)
        if label in labels:
            labels[label] += 1
        else:
            labels[label] = 1

    #return majority
    return max(labels, key=labels.get)


def analyze_forest(dataset,forest):
    """
    Tests how accurately a forest classifies a dataset.
        Parameters:
            dataset (ndarray): Llbeled data with the labels in the last column
            forest (list): list of decision trees
        Returns:
            (float): proportion of dataset classified correctly
    """
    #initialize
    num_correct = 0
    N = len(dataset)

    #test each data point
    for point in dataset:
        if predict_forest(point, forest) == point[-1]:
            num_correct += 1

    return num_correct / N


def build_forest(dataset, feature_names, min_samples_leaf=5, max_depth=4, num_tree=64):
    """
    Builds a random forest of decision trees.
        Parameters:
            dataset (ndarray): data to build forest from
            num_tree: number of trees to have in the forest
    """
    return [build_tree(dataset, feature_names, min_samples_leaf, max_depth,
            random_subset=True) for _ in range(num_tree)]


def compare():
    """
    Compares the compile time and accuracy of my implementation of a random
    forest to scikit-learn's implementation. The data used is data on parkinson's
    disease. It trains on 250
        Returns:
            (my accuracy in a 5-tree forest, my accuracy in a 5-tree forest)
            (scikit's accuracy in a 5-tree forest, scikit's accuracy in a 5-tree forest)
            (my accuracy in a 32-tree forest, my accuracy in a 32-tree forest)
            (scikit's accuracy in a 32-tree forest, scikit's accuracy in a 32-tree forest)
    """
    #get data
    data = np.loadtxt('parkinsons.csv', delimiter=',')[:,1:]
    features = np.loadtxt('parkinsons_features.csv', delimiter=',', dtype=np.str)[1:]
    shuffled = np.random.permutation(data)
    train = shuffled[:250]
    test = shuffled[250:300]

    #constants
    min_samples_leaf = 15
    max_depth = 4

    #train/test my random forest
    start = time()
    my_frst = build_forest(train, features, min_samples_leaf, num_tree=5)
    accuracy = analyze_forest(test, my_frst)
    tot_time = time() - start
    my_5 = (accuracy, tot_time)

    #train/test scikit-learn's random forest
    start = time()
    sk_frst = RandomForestClassifier(n_estimators=5, max_depth=max_depth, min_samples_leaf=min_samples_leaf)
    sk_frst.fit(train[:,:-1], train[:,-1])
    accuracy = sk_frst.score(test[:,:-1], test[:,-1])
    tot_time = time() - start
    sk_5 = (accuracy, tot_time)

    #train/test my random forest
    start = time()
    my_frst = build_forest(train, features, min_samples_leaf, num_tree=32)
    accuracy = analyze_forest(test, my_frst)
    tot_time = time() - start
    my_32 = (accuracy, tot_time)

    #train/test scikit-learn's random forest
    start = time()
    sk_frst = RandomForestClassifier(n_estimators=32, max_depth=max_depth, min_samples_leaf=min_samples_leaf)
    sk_frst.fit(train[:,:-1], train[:,-1])
    accuracy = sk_frst.score(test[:,:-1], test[:,-1])
    tot_time = time() - start
    sk_32 = (accuracy, tot_time)

    return my_5, sk_5, my_32, sk_32
