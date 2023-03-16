from ll_l21 import proximal_gradient_descent, feature_ranking

from test import load_in_data, svm_test

import numpy as np

def multitask_fs(data, k=50):
    train_X, train_y, test_X, test_y = load_in_data(data)
    # If the data is madelon, take K=20, else K=50 features from model feature importance

    W, obj, val_gamma = proximal_gradient_descent(train_X, train_y, z=0.1, verbose=True)

    idx = feature_ranking(W)

    # train a classification model with the selected features on the training dataset
    train_X_new = train_X[:, idx[:k]]
    test_X_new = test_X[:, idx[:k]]

    # Train SVM classifier with the selected features
    train_y = np.argmax(train_y, axis=1)
    test_y = np.argmax(test_y, axis=1)
    # print(f"The x shapes going into the SVM are {train_X_new.shape} and {test_X_new.shape}")
    # print(f"The y shapes going into the SVM are {train_y.shape} and {test_y.shape}")
    
    return train_X_new, train_y, test_X_new, test_y