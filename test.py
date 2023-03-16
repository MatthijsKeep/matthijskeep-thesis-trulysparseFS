from sklearn import svm
from utils.load_data import load_fashion_mnist_data, load_cifar10_data, load_madelon_data, load_mnist_data

import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

# train SVM classifier with the selected features
def svm_test(train_X_new, train_y, test_X, test_y):
    clf = svm.SVC()
    clf.fit(train_X_new, train_y)
    return float(clf.score(test_X, test_y))

def load_in_data(data='madelon'):
    if data == 'madelon':
        return load_madelon_data()
    elif data == 'mnist':
        return load_mnist_data(50000, 10000)
    elif data == 'fashion_mnist':
        return load_fashion_mnist_data(50000, 10000)
    else:
        raise ValueError('Dataset not found')

    
if __name__ == '__main__':
    # load in data 
    train_X, train_y, test_X, test_y = load_in_data(data='madelon')
    
    # if the labels are one-hot encoded, convert to single integer
    if len(train_y.shape) > 1:
        train_y = np.argmax(train_y, axis=1)
        test_y = np.argmax(test_y, axis=1)
    svm_acc = svm_test(train_X, train_y, test_X, test_y)

    # print out the results 
    print(f"SVM Accuracy: {svm_acc}")
    