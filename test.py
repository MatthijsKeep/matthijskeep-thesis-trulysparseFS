from lassonet import LassoNetClassifierCV, plot_path
from ll_l21 import proximal_gradient_descent, feature_ranking
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE, SelectFromModel
from sklearn.linear_model import SGDClassifier, MultiTaskElasticNetCV
from stg import STG
from utils.load_data import load_fashion_mnist_data, load_cifar10_data, load_madelon_data, load_mnist_data

from numpy import loadtxt
from xgboost import XGBClassifier

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

def run_feature_selection_baselines(data='madelon', models=None):

    if models is None:
        raise ValueError('No models specified')

    metrics = {model: {"accuracy": []} for model in models}



    for model in models:
        if model == 'MultiTaskElasticNetCV':
            print(f"\n Running MultiTaskElasticNetCV on {data} dataset \n")
            train_X, train_y, test_X, test_y = load_in_data(data)
            # If the data is madelon, take K=20, else K=50 features from model feature importance
            K = 20 if data == 'madelon' else 50

            W, obj, val_gamma = proximal_gradient_descent(train_X, train_y, z=0.1, verbose=True)

            idx = feature_ranking(W)

            # train a classification model with the selected features on the training dataset
            train_X_new = train_X[:, idx[:K]]
            test_X_new = test_X[:, idx[:K]]

            # Train SVM classifier with the selected features
            train_y = np.argmax(train_y, axis=1)
            test_y = np.argmax(test_y, axis=1)
            print(f"The x shapes going into the SVM are {train_X_new.shape} and {test_X_new.shape}")
            print(f"The y shapes going into the SVM are {train_y.shape} and {test_y.shape}")

            acc = svm_test(train_X_new, train_y, test_X_new, test_y)

            metrics['MultiTaskElasticNetCV']['accuracy'].append(acc)
            print('MultiTaskElasticNetCV Accuracy: ', acc)
        if model == 'stochastic_gates':
            print(f"\n Running STG on {data} dataset \n")
            train_X, train_y, test_X, test_y = load_in_data(data)
            # If the data is madelon, take K=20, else K=50 features from model feature importance
            K = 20 if data == 'madelon' else 50

            # make labels from one-hot encoding to single integer
            train_y = np.argmax(train_y, axis=1)
            test_y = np.argmax(test_y, axis=1)

            # make STG (classification) take K features
            model = STG(task_type='classification',input_dim=train_X.shape[1], output_dim=2, hidden_dims=[60, 20], activation='tanh',
                        optimizer='SGD', learning_rate=0.1, batch_size=train_X.shape[0], feature_selection=True, sigma=0.5, 
                        lam=0.5, random_state=1, device='cpu') 
            # If you get an error here, replace collections with collections.abc in the source code
            model.fit(train_X, train_y, nr_epochs=100, verbose=False)
            gates = model.get_gates(mode='prob')
            print(f"gates.shape: {gates.shape}")

            # Get the indices of the top K features
            top_k_indices = np.argsort(gates)[::-1][:K]

            # Get the top K features
            train_X_new = train_X[:, top_k_indices]
            test_X_new = test_X[:, top_k_indices]

            print("The shapes going into the SVM are", train_X_new.shape, test_X_new.shape)
            # Train SVM classifier with the selected features
            acc = svm_test(train_X_new, train_y, test_X_new, test_y)

            metrics['stochastic_gates']['accuracy'].append(acc)
            print('STG Accuracy: ', acc)
        if model == 'LassoNet':
            print(f"\n Running LassoNet on {data} dataset \n")
            train_X, train_y, test_X, test_y = load_in_data(data)

            # If the data is madelon, take K=20, else K=50 features from model feature importance
            K = 20 if data == 'madelon' else 50

            # make labels from one-hot encoding to single integer
            train_y = np.argmax(train_y, axis=1)
            test_y = np.argmax(test_y, axis=1)

            model = LassoNetClassifierCV(M=K,
            hidden_dims=(10,),
            verbose=True,
            dropout=0.2,
            )
            path = model.path(train_X, train_y)

            plot_path(model, path, test_X, test_y)

            plt.savefig("lasso_madelon.png")

            plt.clf()


            # Get the indices of the top K features
            print(model.get_params())
            top_k_indices = np.argsort(model.feature_importances_.numpy())[::-1][:K]
            # Get the top K features
            train_X_new = train_X[:, top_k_indices]
            test_X_new = test_X[:, top_k_indices]

            # Train SVM classifier with the selected features
            acc = svm_test(train_X_new, train_y, test_X_new, test_y)

            metrics['LassoNet']['accuracy'].append(acc)
            print('LassoNet Accuracy: ', acc)

    # Save the metrics
    with open(f'./results/{data}_feature_selection_metrics.json', 'w') as f:
        json.dump(metrics, f)

    print(f"Metrics saved to ./results/{data}_feature_selection_metrics.json")

    # sort the metrics by accuracy
    metrics = dict(
        sorted(
            metrics.items(), key=lambda item: item[1]['accuracy'], reverse=True
        )
    )

    # print the metrics and round to 3 decimal places
    print(f"\n\nMetrics for {data} dataset: \n")
    for model, metric in metrics.items():
        print(f"Model: {model}")
        for metric_name, metric_value in metric.items():
            print(f"{metric_name}: {round(np.mean(metric_value), 3)}")


    
if __name__ == '__main__':
    # run_feature_selection_baselines(data='madelon')
    run_feature_selection_baselines(data='madelon', models=['MultiTaskElasticNetCV', 'stochastic_gates', 'LassoNet'])
    # run_feature_selection_baselines(data='fashion_mnist'