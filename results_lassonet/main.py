import os
import sys
from xmlrpc.client import boolean
sys.path.append(os.getcwd())
import time
import logging
import copy
import shutil

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np

from dst import dst_FS
import models
from argparser import get_parser

if not os.path.exists('./models'): os.mkdir('./models')
if not os.path.exists('./logs'): os.mkdir('./logs')
logger = None

##### NEW IMPORTS BY MATTHIJS #####
# 
import copy
import pprint
import time
from typing_extensions import TypeAlias
import wandb

wandb.login(key="43d952ea50348fd7b9abbc1ab7d0b787571e8918", timeout=300)

print(os.getcwd())

from lassonet import LassoNetClassifierCV, plot_path

from sklearn import svm

# train SVM classifier with the selected features
def svm_test(train_X_new, train_y, test_X, test_y):
    clf = svm.SVC()
    clf.fit(train_X_new, train_y)
    return float(clf.score(test_X, test_y))

import numpy as np

import matplotlib.pyplot as plt

from results_lassonet.fastr_utils.load_data import *

def get_data(dataset, **kwargs):
    """
    Function to load the data from the dataset.

    :param dataset: (string) Name of the dataset

    :return x_train: (array) Training input
    :return y_train: (array) Correct training output
    :return x_test: (array) Test input
    :return y_test: (array) Correct test output

    """

    if dataset == 'FashionMnist':
        x_train, y_train, x_test, y_test, x_val, y_val = load_fashion_mnist_data(50000, 10000)
    elif dataset == 'mnist':
        x_train, y_train, x_test, y_test, x_val, y_val = load_mnist_data(50000, 10000)
    elif dataset == 'madelon':
        x_train, y_train, x_test, y_test, x_val, y_val = load_madelon_data()
    elif dataset == 'usps':
        x_train, y_train, x_test, y_test = load_usps()
        x_val, y_val = x_test, y_test # None for now
    elif dataset == 'coil':
        x_train, y_train, x_test, y_test = load_coil()
        x_val, y_val = x_test, y_test # None for now
    elif dataset == 'isolet':
        x_train, y_train, x_test, y_test = load_isolet()
        x_val, y_val = x_test, y_test # None for now
    elif dataset == 'har':
        x_train, y_train, x_test, y_test = load_har()
        x_val, y_val = x_test, y_test # None for now
    elif dataset == 'smk':
        x_train, y_train, x_test, y_test = load_smk()
        x_val, y_val = x_test, y_test # None for now
    elif dataset == 'gla':
        x_train, y_train, x_test, y_test = load_gla()
        x_val, y_val = x_test, y_test # None for now
    elif dataset == 'synthetic':
        x_train, y_train, x_test, y_test, x_val, y_val = load_synthetic(n_samples = kwargs['n_samples'],
                                                                        n_features = kwargs['n_features'],
                                                                        n_classes = kwargs['n_classes'],
                                                                        n_informative = kwargs['n_informative'],
                                                                        n_redundant = kwargs['n_redundant'],
                                                                        n_clusters_per_class = kwargs['n_clusters_per_class'],)
    elif dataset == 'synthetic1':
        print("Loading synthetic1")
        x_train, y_train, x_test, y_test, x_val, y_val = load_synthetic(n_samples = 1000,
                                                                        n_features = 2500,
                                                                        n_classes = 2,
                                                                        n_informative = 20,
                                                                        n_redundant = 0,
                                                                        n_clusters_per_class = 16)
    elif dataset == 'synthetic2':
        print("Loading synthetic2")
        x_train, y_train, x_test, y_test, x_val, y_val = load_synthetic(n_samples = 500,
                                                                        n_features = 2500,
                                                                        n_classes = 2,
                                                                        n_informative = 20,
                                                                        n_redundant = 0,
                                                                        n_clusters_per_class = 16)
    elif dataset == 'synthetic3':
        print("Loading synthetic3")
        x_train, y_train, x_test, y_test, x_val, y_val = load_synthetic(n_samples = 500,
                                                                        n_features = 5000,
                                                                        n_classes = 2,
                                                                        n_informative = 20,
                                                                        n_redundant = 0,
                                                                        n_clusters_per_class = 16)
        
    elif dataset == 'synthetic4':
        print("Loading synthetic4")
        x_train, y_train, x_test, y_test, x_val, y_val = load_synthetic(n_samples = 500,
                                                                        n_features = 10000,
                                                                        n_classes = 2,
                                                                        n_informative = 20,
                                                                        n_redundant = 0,
                                                                        n_clusters_per_class = 16)
    elif dataset == 'synthetic5':
        print("Loading synthetic5")
        x_train, y_train, x_test, y_test, x_val, y_val = load_synthetic(n_samples = 100,
                                                                        n_features = 10000,
                                                                        n_classes = 2,
                                                                        n_informative = 20,
                                                                        n_redundant = 0,
                                                                        n_clusters_per_class = 16)
        
        
    else:
        raise ValueError("Unknown dataset")
    return x_train, y_train, x_test, y_test, x_val, y_val



def lassonet_fs(data, K):
    """
    Return new train and test data with K features selected using the lassonet algorithm

    Args:

    """
    # Load in data
    train_X, train_y, test_X, test_y, _, _= get_data(data)
    # make labels from one-hot encoding to single integer
    train_y = np.argmax(train_y, axis=1)
    test_y = np.argmax(test_y, axis=1)
    model = LassoNetClassifierCV(M=K,
    hidden_dims=(10,),
    verbose=0,
    )
    path = model.path(train_X, train_y)

    # Get the indices of the top K features
    print(model.get_params())
    top_k_indices = np.argsort(model.feature_importances_.numpy())[::-1][:K]
    # Get the top K features
    train_X_new = train_X[:, top_k_indices]
    test_X_new = test_X[:, top_k_indices]

    # Train SVM classifier with the selected features
    return train_X_new, train_y, test_X_new, test_y, top_k_indices


def main(config):
    # print(args)
    config = wandb.config
    print(config)
    print("---------------------")
    print("Starting LassoNet")

    if config.data in ["synthetic1", "synthetic2", "synthetic3", "synthetic4", "synthetic5", "madelon"]:
        print(f"Since data is in {['synthetic1', 'synthetic2', 'synthetic3', 'synthetic4', 'synthetic5', 'madelon']}, we will use K = 20")
        K = 20
    else:
        K = 50

    train_X_new, train_y, test_X_new, test_y, top_k_indices = lassonet_fs(config.data, K)

    # Train SVM classifier with the selected features
    svm_acc = svm_test(train_X_new, train_y, test_X_new, test_y)

    if config.data in ["synthetic1", "synthetic2", "synthetic3", "synthetic4", "synthetic5"]:
        # check how many of the selected indices are in the first 20
        print(top_k_indices)
        informative_indices = np.arange(20)
        pct_correct = len(np.intersect1d(top_k_indices, informative_indices)) / len(informative_indices)
        print(f"Percentage of correct indices: {pct_correct}")
        wandb.summary["pct_correct"] = pct_correct
        wandb.summary["svm_acc"] = svm_acc
        wandb.log({"pct_correct": pct_correct, "svm_acc": svm_acc})
        return pct_correct, svm_acc
    else:
        wandb.summary["svm_acc"] = svm_acc
        wandb.log({"svm_acc": svm_acc})
        return svm_acc



    
if __name__ == '__main__': 
    

    sweep_config = {
    'method': 'grid',
    'metric': {
        'name': 'accuracy_topk',
        'goal': 'maximize'
    },
    'early_terminate': {
        'type': 'hyperband',
        'min_iter': 50
    },
    'parameters': {
        'data': {
            'distribution': 'categorical',
            'values': ['synthetic1', 'synthetic2', 'synthetic3', 'synthetic4', 'synthetic5', 'mnist', 'madelon', 'smk', 'gla', 'usps', 'coil', 'isolet', 'har']
        },
        }
    }

    pprint.pprint(sweep_config)

    def run_wast(config=None):
        with wandb.init(config=config):
            main(config)
    
    sweep_id = wandb.sweep(sweep_config, project="results_lassonet")
    wandb.agent(sweep_id, function=run_wast)
    
