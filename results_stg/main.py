import os
import sys
from xmlrpc.client import boolean

sys.path.append(os.getcwd())

import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np

from sklearn.model_selection import train_test_split

if not os.path.exists("./models"):
    os.mkdir("./models")
if not os.path.exists("./logs"):
    os.mkdir("./logs")
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

import stg


from sklearn import svm


# train SVM classifier with the selected features
def svm_test(train_X_new, train_y, test_X, test_y):
    clf = svm.SVC()
    clf.fit(train_X_new, train_y)
    return float(clf.score(test_X, test_y))


import numpy as np

import matplotlib.pyplot as plt

from results_stg.fastr_utils.load_data import *


def get_data(dataset, **kwargs):
    """
    Function to load the data from the dataset.

    :param dataset: (string) Name of the dataset

    :return x_train: (array) Training input
    :return y_train: (array) Correct training output
    :return x_test: (array) Test input
    :return y_test: (array) Correct test output

    """

    if dataset == "FashionMnist":
        x_train, y_train, x_test, y_test, x_val, y_val = load_fashion_mnist_data(
            50000, 10000
        )
        x_test, x_val, y_test, y_val = train_test_split(
            x_test, y_test, test_size=0.5, random_state=42, stratify=y_test
        )
    elif dataset == "mnist":
        x_train, y_train, x_test, y_test, x_val, y_val = load_mnist_data(50000, 10000)
        x_test, x_val, y_test, y_val = train_test_split(
            x_test, y_test, test_size=0.5, random_state=42, stratify=y_test
        )
    elif dataset == "madelon":
        x_train, y_train, x_test, y_test, x_val, y_val = load_madelon_data()
        x_test, x_val, y_test, y_val = train_test_split(
            x_test, y_test, test_size=0.5, random_state=42, stratify=y_test
        )
    elif dataset == "usps":
        x_train, y_train, x_test, y_test = load_usps()
        x_test, x_val, y_test, y_val = train_test_split(
            x_test, y_test, test_size=0.5, random_state=42, stratify=y_test
        )
    elif dataset == "coil":
        x_train, y_train, x_test, y_test = load_coil()
        x_test, x_val, y_test, y_val = train_test_split(
            x_test, y_test, test_size=0.5, random_state=42, stratify=y_test
        )
    elif dataset == "isolet":
        x_train, y_train, x_test, y_test = load_isolet()
        x_test, x_val, y_test, y_val = train_test_split(
            x_test, y_test, test_size=0.5, random_state=42, stratify=y_test
        )
    elif dataset == "har":
        x_train, y_train, x_test, y_test = load_har()
        x_test, x_val, y_test, y_val = train_test_split(
            x_test, y_test, test_size=0.5, random_state=42, stratify=y_test
        )
    elif dataset == "smk":
        x_train, y_train, x_test, y_test = load_smk()
        x_test, x_val, y_test, y_val = train_test_split(
            x_test, y_test, test_size=0.5, random_state=42, stratify=y_test
        )
    elif dataset == "gla":
        x_train, y_train, x_test, y_test = load_gla()
        x_test, x_val, y_test, y_val = train_test_split(
            x_test, y_test, test_size=0.5, random_state=42, stratify=y_test
        )
    elif dataset == "synthetic":
        x_train, y_train, x_test, y_test, x_val, y_val = load_synthetic(
            n_samples=kwargs["n_samples"],
            n_features=kwargs["n_features"],
            n_classes=kwargs["n_classes"],
            n_informative=kwargs["n_informative"],
            n_redundant=kwargs["n_redundant"],
            n_clusters_per_class=kwargs["n_clusters_per_class"],
        )
    elif dataset == "synthetic1":
        print("Loading synthetic1")
        x_train, y_train, x_test, y_test, x_val, y_val = load_synthetic(
            n_samples=1000,
            n_features=2500,
            n_classes=2,
            n_informative=20,
            n_redundant=0,
            n_clusters_per_class=16,
        )
    elif dataset == "synthetic2":
        print("Loading synthetic2")
        x_train, y_train, x_test, y_test, x_val, y_val = load_synthetic(
            n_samples=500,
            n_features=2500,
            n_classes=2,
            n_informative=20,
            n_redundant=0,
            n_clusters_per_class=16,
        )
    elif dataset == "synthetic3":
        print("Loading synthetic3")
        x_train, y_train, x_test, y_test, x_val, y_val = load_synthetic(
            n_samples=500,
            n_features=5000,
            n_classes=2,
            n_informative=20,
            n_redundant=0,
            n_clusters_per_class=16,
        )

    elif dataset == "synthetic4":
        print("Loading synthetic4")
        x_train, y_train, x_test, y_test, x_val, y_val = load_synthetic(
            n_samples=500,
            n_features=10000,
            n_classes=2,
            n_informative=20,
            n_redundant=0,
            n_clusters_per_class=16,
        )
    elif dataset == "synthetic5":
        print("Loading synthetic5")
        x_train, y_train, x_test, y_test, x_val, y_val = load_synthetic(
            n_samples=100,
            n_features=10000,
            n_classes=2,
            n_informative=20,
            n_redundant=0,
            n_clusters_per_class=16,
        )

    else:
        raise ValueError("Unknown dataset")
    return x_train, y_train, x_test, y_test, x_val, y_val


def stg_fs(data, k=50, output_dim=2):
    """
    Return new train and test data with K features selected using the stg algorithm
    """

    train_X, train_y, test_X, test_y, val_X, val_y = get_data(data)

    #  make labels from one-hot encoding to single integer
    if len(train_y.shape) > 1:
        train_y = np.argmax(train_y, axis=1)
        test_y = np.argmax(test_y, axis=1)
        val_y = np.argmax(val_y, axis=1)

    print(train_y[:3], test_y[:3], val_y[:3])

    print(
        f" Shapes going into STG: train_X: {train_X.shape}, train_y: {train_y.shape}, test_X: {test_X.shape}, test_y: {test_y.shape}"
    )

    # make STG (classification) take K features
    print("before STG ")
    model = stg.STG(
        task_type="classification",
        input_dim=train_X.shape[1],
        output_dim=output_dim,
        hidden_dims=200,
        activation="relu",
        optimizer="SGD",
        learning_rate=0.01,
        batch_size=train_X.shape[0],
        feature_selection=True,
        sigma=0.5,
        lam=0.5,
        random_state=1,
        device="cpu",
    )
    print("fitting STG")
    # If you get an error here, replace collections with collections.abc in the source code
    model.fit(
        X=train_X,
        y=train_y,
        valid_X=val_X,
        valid_y=val_y,
        nr_epochs=500,
        verbose=True,
        shuffle=True,
        print_interval=50,
    )
    print("after STG")
    gates = model.get_gates(mode="prob")
    print(f"gates.shape: {gates.shape}")

    # Get the indices of the top K features
    top_k_indices = np.argsort(gates)[::-1][:k]

    # Get the top K features
    train_X_new = train_X[:, top_k_indices]
    test_X_new = test_X[:, top_k_indices]

    print("The shapes going into the SVM are", train_X_new.shape, test_X_new.shape)
    # Train SVM classifier with the selected features

    return train_X_new, train_y, test_X_new, test_y, top_k_indices


def main(config):
    # print(args)
    config = wandb.config
    print(config)
    print("---------------------")
    print("Starting LassoNet")

    if config.data in [
        "synthetic1",
        "synthetic2",
        "synthetic3",
        "synthetic4",
        "synthetic5",
        "madelon",
    ]:
        print(
            f"Since data is in {['synthetic1', 'synthetic2', 'synthetic3', 'synthetic4', 'synthetic5', 'madelon']}, we will use K = 20"
        )
        K = 20
    else:
        K = 50

    if config.data in [
        "synthetic1",
        "synthetic2",
        "synthetic3",
        "synthetic4",
        "synthetic5",
        "madelon",
        "smk",
    ]:
        output_dim = 2
    elif config.data in ["gla"]:
        output_dim = 4
    elif config.data in ["har"]:
        output_dim = 6
    elif config.data in ["mnist", "usps"]:
        output_dim = 10
    elif config.data in ["coil"]:
        output_dim = 20
    elif config.data in ["isolet"]:
        output_dim = 26
    else:
        raise ValueError("Unknown dataset, maybe check for typos")

    train_X_new, train_y, test_X_new, test_y, top_k_indices = stg_fs(
        config.data, K, output_dim
    )

    # Train SVM classifier with the selected features

    print(train_y)
    print(test_y)
    svm_acc = svm_test(train_X_new, train_y, test_X_new, test_y)

    if config.data in [
        "synthetic1",
        "synthetic2",
        "synthetic3",
        "synthetic4",
        "synthetic5",
    ]:
        # check how many of the selected indices are in the first 20
        print(top_k_indices)
        informative_indices = np.arange(20)
        pct_correct = len(np.intersect1d(top_k_indices, informative_indices)) / len(
            informative_indices
        )
        print(f"Percentage of correct indices: {pct_correct}")
        wandb.summary["pct_correct"] = pct_correct
        wandb.summary["svm_acc"] = svm_acc
        wandb.log({"pct_correct": pct_correct, "svm_acc": svm_acc})
        return pct_correct, svm_acc
    else:
        wandb.summary["svm_acc"] = svm_acc
        wandb.log({"svm_acc": svm_acc})
        return svm_acc


if __name__ == "__main__":
    sweep_config = {
        "method": "grid",
        "metric": {"name": "accuracy_topk", "goal": "maximize"},
        "early_terminate": {"type": "hyperband", "min_iter": 50},
        "parameters": {
            "data": {
                "distribution": "categorical",
                "values": [
                    "synthetic1",
                    "synthetic2",
                    "synthetic3",
                    "synthetic4",
                    "synthetic5",
                    "mnist",
                    "madelon",
                    "smk",
                    "gla",
                    "usps",
                    "coil",
                    "isolet",
                    "har",
                ],
            },
        },
    }

    pprint.pprint(sweep_config)

    def run_wast(config=None):
        with wandb.init(config=config):
            main(config)

    sweep_id = wandb.sweep(sweep_config, project="results_stg")
    wandb.agent(sweep_id, function=run_wast)
