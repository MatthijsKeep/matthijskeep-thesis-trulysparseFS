from stg import STG
from test import load_in_data, svm_test

import numpy as np

def stg_fs(data, k=50):
    """
    Return new train and test data with K features selected using the stg algorithm
    """

    train_X, train_y, test_X, test_y = load_in_data(data)

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
    top_k_indices = np.argsort(gates)[::-1][:k]

    # Get the top K features
    train_X_new = train_X[:, top_k_indices]
    test_X_new = test_X[:, top_k_indices]

    print("The shapes going into the SVM are", train_X_new.shape, test_X_new.shape)
    # Train SVM classifier with the selected features

    return train_X_new, train_y, test_X_new, test_y