from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE, SelectFromModel
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
        if model == 'xgboost':
            train_X, train_y, test_X, test_y = load_in_data(data)
            print(f"\n Running XGBoost on {data} dataset \n")
            model = XGBClassifier()
            model.fit(train_X, train_y)

            # If the data is madelon, take K=20, else K=50 features from model feature importance
            K = 20 if data == 'madelon' else 50
            # Get the feature importance from the model
            feature_importance = model.feature_importances_
            # Get the indices of the top K features
            top_k_indices = np.argsort(feature_importance)[::-1][:K]
            # Get the top K features
            train_X_new = train_X[:, top_k_indices]
            test_X_new = test_X[:, top_k_indices]

            # make labels from one-hot encoding to single integer
            train_y = np.argmax(train_y, axis=1)
            test_y = np.argmax(test_y, axis=1)
            # Train SVM classifier with the selected features
            acc = svm_test(train_X_new, train_y, test_X_new, test_y)

            metrics['xgboost']['accuracy'].append(acc)
            print('XGBoost Accuracy: ', acc)
        if model == 'f_classif':
            print(f"\n Running f_classif on {data} dataset \n")
            train_X, train_y, test_X, test_y = load_in_data(data)
            # If the data is madelon, take K=20, else K=50 features from model feature importance
            K = 20 if data == 'madelon' else 50

            # make labels from one-hot encoding to single integer
            train_y = np.argmax(train_y, axis=1)
            test_y = np.argmax(test_y, axis=1)

            train_X_new = SelectKBest(f_classif, k=K).fit_transform(train_X, train_y)
            # take the same features from test data
            test_X_new = SelectKBest(f_classif, k=K).fit_transform(test_X, test_y)

            # Train SVM classifier with the selected features
            acc = svm_test(train_X_new, train_y, test_X_new, test_y)

            metrics['f_classif']['accuracy'].append(acc)
            print('f_classif Accuracy: ', acc)
        if model == 'mutual_info_classif':
            print(f"\n Running mutual_info_classif on {data} dataset \n")
            train_X, train_y, test_X, test_y = load_in_data(data)
            # If the data is madelon, take K=20, else K=50 features from model feature importance
            K = 20 if data == 'madelon' else 50

            # make labels from one-hot encoding to single integer
            train_y = np.argmax(train_y, axis=1)
            test_y = np.argmax(test_y, axis=1)

            train_X_new = SelectKBest(mutual_info_classif, k=K).fit_transform(train_X, train_y)
            # take the same features from test data
            test_X_new = SelectKBest(mutual_info_classif, k=K).fit_transform(test_X, test_y)

            # Train SVM classifier with the selected features
            acc = svm_test(train_X_new, train_y, test_X_new, test_y)

            metrics['mutual_info_classif']['accuracy'].append(acc)
            print('mutual_info_classif Accuracy: ', acc)
        if model == 'LinearSVC':
            print(f"\n Running LinearSVC on {data} dataset \n")
            train_X, train_y, test_X, test_y = load_in_data(data)
            # If the data is madelon, take K=20, else K=50 features from model feature importance
            K = 20 if data == 'madelon' else 50

            # make labels from one-hot encoding to single integer
            train_y = np.argmax(train_y, axis=1)
            test_y = np.argmax(test_y, axis=1)

            # LinearSVC with L1-based feature selection, until there are K features left
            lsvc = svm.LinearSVC(C=0.00553, penalty="l1", dual=False).fit(train_X, train_y)
            model = SelectFromModel(lsvc, prefit=True)
            print(f"Number of features before LinearSVC: {train_X.shape[1]}")
            train_X_new = model.transform(train_X)
            test_X_new = model.transform(test_X)
            print(f"Number of features after LinearSVC: {train_X_new.shape[1]}")


            # Train SVM classifier with the selected features
            acc = svm_test(train_X_new, train_y, test_X_new, test_y)

            metrics['LinearSVC']['accuracy'].append(acc)
            print('LinearSVC Accuracy: ', acc)
        if model == 'RandomForestClassifier':
            print(f"\n Running RandomForestClassifier on {data} dataset \n")
            train_X, train_y, test_X, test_y = load_in_data(data)
            # If the data is madelon, take K=20, else K=50 features from model feature importance
            K = 20 if data == 'madelon' else 50

            # make labels from one-hot encoding to single integer
            train_y = np.argmax(train_y, axis=1)
            test_y = np.argmax(test_y, axis=1)

            # LinearSVC with L1-based feature selection, until there are K features left
            clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
            clf.fit(train_X, train_y)

            feature_importance = clf.feature_importances_
            # print(f"Number of features before RandomForestClassifier: {train_X.shape[1]}")
            # Get the top K features
            top_k_indices = np.argsort(feature_importance)[::-1][:K]
            train_X_new = train_X[:, top_k_indices]
            test_X_new = test_X[:, top_k_indices]
            # print(f"Number of features after RandomForestClassifier: {train_X_new.shape[1]}")


            # Train SVM classifier with the selected features
            acc = svm_test(train_X_new, train_y, test_X_new, test_y)

            metrics['RandomForestClassifier']['accuracy'].append(acc)
            print('RandomForestClassifier Accuracy: ', acc)

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

    # print the metrics
    print(f"\n\nMetrics for {data} dataset: \n")
    for model, metric in metrics.items():
        print(f"Model: {model}")
        print(f"Accuracy: {round(metric['accuracy'][0], 3)}")

    

    
if __name__ == '__main__':
    # run_feature_selection_baselines(data='madelon')
    run_feature_selection_baselines(data='madelon', models=['xgboost', 'f_classif', 'mutual_info_classif', 'LinearSVC', 'RandomForestClassifier'])
    # run_feature_selection_baselines(data='fashion_mnist'