import os
import sys
from xmlrpc.client import boolean
sys.path.append(os.getcwd())

import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np

from sklearn.model_selection import train_test_split

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

from concrete_autoencoder import ConcreteAutoencoderFeatureSelector

from keras.utils import to_categorical
from keras.layers import Dense, Dropout, LeakyReLU
import numpy as np

from sklearn import svm

# train SVM classifier with the selected features
def svm_test(train_X_new, train_y, test_X, test_y):
    clf = svm.SVC()
    clf.fit(train_X_new, train_y)
    return float(clf.score(test_X, test_y))

import numpy as np

import matplotlib.pyplot as plt

from fastr_utils.load_data import *

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
        x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, random_state=42, stratify=y_test)
    elif dataset == 'mnist':
        x_train, y_train, x_test, y_test, x_val, y_val = load_mnist_data(50000, 10000)
        x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, random_state=42, stratify=y_test)
    elif dataset == 'madelon':
        x_train, y_train, x_test, y_test, x_val, y_val = load_madelon_data()
        x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, random_state=42, stratify=y_test)
    elif dataset == 'usps':
        x_train, y_train, x_test, y_test = load_usps()
        x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, random_state=42, stratify=y_test)
    elif dataset == 'coil':
        x_train, y_train, x_test, y_test = load_coil()
        x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, random_state=42, stratify=y_test)
    elif dataset == 'isolet':
        x_train, y_train, x_test, y_test = load_isolet()
        x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, random_state=42, stratify=y_test)
    elif dataset == 'har':
        x_train, y_train, x_test, y_test = load_har()
        x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, random_state=42, stratify=y_test)
    elif dataset == 'smk':
        x_train, y_train, x_test, y_test = load_smk()
        x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, random_state=42, stratify=y_test)
    elif dataset == 'gla':
        x_train, y_train, x_test, y_test = load_gla()
        x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, random_state=42, stratify=y_test)
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



def cae_fs(config, output_classes, K):
    """
    Return new train and test data with K features selected using the CAE algorithm
    """

    x_train, y_train, x_test, y_test, x_val, y_val = get_data(config.data)

    x_train = np.reshape(x_train, (len(x_train), -1))
    x_test = np.reshape(x_test, (len(x_test), -1))
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)

    output_dim = x_train.shape[1]

    def decoder(x):
        x = Dense(output_dim)(x)
        x = LeakyReLU(0.2)(x)
        x = Dropout(0.1)(x)
        x = Dense(int(1.5*output_dim))(x)
        x = LeakyReLU(0.2)(x)
        x = Dropout(0.1)(x)
        x = Dense(output_dim)(x)
        return x

    selector = ConcreteAutoencoderFeatureSelector(K = K, output_function = decoder, num_epochs = 200)

    selector.fit(x_train, x_train, x_test, x_test)
    # Train SVM classifier with the selected features

    top_k_indices = selector.get_support(indices = True)

    train_X_new = x_train[:, top_k_indices]
    test_X_new = x_test[:, top_k_indices]

    train_y = np.argmax(y_train, axis = 1)
    test_y = np.argmax(y_test, axis = 1)
    return train_X_new, train_y, test_X_new, test_y, top_k_indices


def main(config):
    # print(args)
    config = wandb.config
    print(config)
    print("---------------------")
    print("Starting CAE")

    if config.data in ["synthetic1", "synthetic2", "synthetic3", "synthetic4", "synthetic5", "madelon"]:
        print(f"Since data is in {['synthetic1', 'synthetic2', 'synthetic3', 'synthetic4', 'synthetic5', 'madelon']}, we will use K = 20")
        K = 20
    else:
        K = 50

    if config.data in ["synthetic1", "synthetic2", "synthetic3", "synthetic4", "synthetic5", "madelon", "smk"]:
        output_classes = 2
    elif config.data in ["gla"]:
        output_classes = 4
    elif config.data in ["har"]:
        output_classes = 6
    elif config.data in ["mnist", "usps"]:
        output_classes = 10
    elif config.data in ["coil"]:
        output_classes = 20
    elif config.data in ["isolet"]:
        output_classes = 26
    else:
        raise ValueError("Unknown dataset, maybe check for typos")
    
    train_X_new, train_y, test_X_new, test_y, top_k_indices = cae_fs(config, output_classes, K)



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

    sweep_id = wandb.sweep(sweep_config, project="results_cae")
    wandb.agent(sweep_id, function=run_wast)
