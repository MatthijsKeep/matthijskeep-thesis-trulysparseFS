from lassonet import LassoNetClassifierCV, plot_path
from test import load_in_data, svm_test

import numpy as np
import matplotlib.pyplot as plt

def lassonet_fs(data, K):
    """
    Return new train and test data with K features selected using the lassonet algorithm

    Args:

    """
    # Load in data
    train_X, train_y, test_X, test_y = load_in_data(data)
    # make labels from one-hot encoding to single integer
    train_y = np.argmax(train_y, axis=1)
    test_y = np.argmax(test_y, axis=1)

    model = LassoNetClassifierCV(M=K,
    hidden_dims=(10,),
    verbose=False,
    )
    path = model.path(train_X, train_y)

    plot_path(model, path, test_X, test_y)

    plt.savefig(f"lasso_{data}.png")

    plt.clf()

    # Get the indices of the top K features
    print(model.get_params())
    top_k_indices = np.argsort(model.feature_importances_.numpy())[::-1][:K]
    # Get the top K features
    train_X_new = train_X[:, top_k_indices]
    test_X_new = test_X[:, top_k_indices]

    # Train SVM classifier with the selected features
    return train_X_new, train_y, test_X_new, test_y
