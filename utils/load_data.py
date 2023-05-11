import numpy as np
import pandas as pd
import os
from concurrent.futures import ProcessPoolExecutor
# from keras.datasets import cifar10, mnist
from keras.utils import to_categorical
from PIL import Image
from scipy.io import loadmat
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import datasets

import os
from matplotlib import dates
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.datasets import fashion_mnist
import torch
import torch.nn.functional as F
import urllib.request as urllib2 
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn import preprocessing 
# from torchvision import datasets, transforms
from PIL import Image



class Error(Exception):
    pass


# Artificial dataset with two classes
def load_madelon_data():
    # Download the data
    x_train = np.loadtxt("data/Madelon/madelon_train.data")
    y_train = np.loadtxt('data/Madelon//madelon_train.labels')
    x_val = np.loadtxt('data/Madelon/madelon_valid.data')
    y_val = np.loadtxt('data/Madelon/madelon_valid.labels')
    x_test = np.loadtxt('data/Madelon/madelon_test.data')

    y_train = np.where(y_train == -1, 0, 1)
    y_val = np.where(y_val == -1, 0, 1)

    xTrainMean = np.mean(x_train, axis=0)
    xTtrainStd = np.std(x_train, axis=0)
    x_train = (x_train - xTrainMean) / xTtrainStd
    x_test = (x_test - xTrainMean) / xTtrainStd
    x_val = (x_val - xTrainMean) / xTtrainStd

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_val = x_val.astype('float32')
    y_train = to_categorical(y_train, 2)
    y_val = to_categorical(y_val, 2)
    y_test = np.zeros((x_test.shape[0], 2))

    return x_train, y_train, x_val, y_val


# The MNIST database of handwritten digits.
def load_mnist_data(n_training_samples, n_testing_samples):

    # read MNIST data
    (x, y), (x_test, y_test) = mnist.load_data()

    y = to_categorical(y, 10)
    y_test = to_categorical(y_test, 10)
    x = x.astype('float32')
    x_test = x_test.astype('float32')

    index_train = np.arange(x.shape[0])
    np.random.shuffle(index_train)

    index_test = np.arange(x_test.shape[0])
    np.random.shuffle(index_test)

    x_train = x[index_train[0:n_training_samples], :]
    y_train = y[index_train[0:n_training_samples], :]

    x_test = x_test[index_test[0:n_testing_samples], :]
    y_test = y_test[index_test[0:n_testing_samples], :]

    # Normalize data
    x_train = x_train / 255.
    x_test = x_test / 255.
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train, x_test = x_train.reshape(x_train.shape[0], 784), x_test.reshape(x_test.shape[0], 784)
    # print(x_train.shape, x_test.shape)
    # print(y_test)
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=42)
    return x_train, y_train, x_test, y_test, x_val, y_val


# Fashion-MNIST is a dataset of Zalando's article images consisting of a training set of 60,000 examples and a test set of 10,000 examples.
# Each example is a 28x28 grayscale image, associated with a label from 10 classes.
def load_fashion_mnist_data(n_training_samples, n_testing_samples):

    data = np.load("data/FashionMNIST/fashion_mnist.npz")

    index_train = np.arange(data["X_train"].shape[0])
    np.random.shuffle(index_train)

    index_test = np.arange(data["X_test"].shape[0])
    np.random.shuffle(index_test)

    x_train = data["X_train"][index_train[0:n_training_samples], :]
    y_train = data["Y_train"][index_train[0:n_training_samples], :]
    x_test = data["X_test"][index_test[0:n_testing_samples], :]
    y_test = data["Y_test"][index_test[0:n_testing_samples], :]

    # Normalize in 0..1
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # Normalize data
    x_train_mean = np.mean(x_train, axis=0)
    x_train_std = np.std(x_train, axis=0)
    x_train = (x_train - x_train_mean) / x_train_std
    x_test = (x_test - x_train_mean) / x_train_std
    
    print(y_test)
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=42)
    return x_train, y_train, x_test, y_test, x_val, y_val


# The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class.
# There are 50000 training images and 10000 test images.
def load_cifar10_data(n_training_samples, n_testing_samples):

    # read CIFAR10 data
    (x, y), (x_test, y_test) = cifar10.load_data()

    y = np_utils.to_categorical(y, 10)
    y_test = np_utils.to_categorical(y_test, 10)
    x = x.astype('float32')
    x_test = x_test.astype('float32')

    index_train = np.arange(x.shape[0])
    np.random.shuffle(index_train)

    index_test = np.arange(x_test.shape[0])
    np.random.shuffle(index_test)

    x_train = x[index_train[0:n_training_samples], :]
    y_train = y[index_train[0:n_training_samples], :]

    x_test = x_test[index_test[0:n_testing_samples], :]
    y_test = y_test[index_test[0:n_testing_samples], :]

    # Normalize data
    x_train_mean = np.mean(x_train, axis=0)
    x_train_std = np.std(x_train, axis=0)
    x_train = (x_train - x_train_mean) / x_train_std
    x_test = (x_test - x_train_mean) / x_train_std

    x_train = x_train.reshape(-1, 32 * 32 * 3)
    x_test = x_test.reshape(-1, 32 * 32 * 3)

    return x_train, y_train, x_test, y_test


# Not flattened version of CIFAR10
def load_cifar10_data_not_flattened(n_training_samples, n_testing_samples):

    # read CIFAR10 data
    (x, y), (x_test, y_test) = cifar10.load_data()

    y = np_utils.to_categorical(y, 10)
    y_test = np_utils.to_categorical(y_test, 10)
    x = x.astype('float32')
    x_test = x_test.astype('float32')

    index_train = np.arange(x.shape[0])
    np.random.shuffle(index_train)

    index_test = np.arange(x_test.shape[0])
    np.random.shuffle(index_test)

    x_train = x[index_train[0:n_training_samples], :]
    y_train = y[index_train[0:n_training_samples], :]

    x_test = x_test[index_test[0:n_testing_samples], :]
    y_test = y_test[index_test[0:n_testing_samples], :]

    # Normalize data
    x_train_mean = np.mean(x_train, axis=0)
    x_train_std = np.std(x_train, axis=0)
    x_train = (x_train - x_train_mean) / x_train_std
    x_test = (x_test - x_train_mean) / x_train_std

    return x_train, y_train, x_test, y_test

# TODO - clean up file

def load_usps():
    # read in UPSP data matrix from data/USPS.mat

    mat = loadmat('./data/USPS.mat')
    X = mat['X']
    y = mat['Y']
    # subtract all values in y by 1 so you have [0, 9] instead of [1, 10]
    y = y - 1
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.20, random_state=42) 

    m = np.mean(train_X)
    std = np.std(train_X)

    # print amount of unique valus in train_y

    train_y = to_categorical(train_y, 10)
    test_y = to_categorical(test_y, 10)
    train_X = train_X.astype('float64')
    test_X = test_X.astype('float64')

    index_train = np.arange(train_X.shape[0])
    np.random.shuffle(index_train)

    index_test = np.arange(test_X.shape[0])
    np.random.shuffle(index_test)

    # random shuffle
    train_X = train_X[index_train, :]
    train_y = train_y[index_train, :]
    test_X = test_X[index_test, :]
    test_y = test_y[index_test, :]

    x_train_mean = np.mean(train_X, axis=0)
    x_train_std = np.std(train_X, axis=0)
    train_X = (train_X - x_train_mean) / x_train_std
    test_X = (test_X - x_train_mean) / x_train_std

    print(test_y)

    return train_X, train_y, test_X, test_y

def load_coil():

    # read in coil data matrix from data/COIL20.mat

    mat = loadmat('./data/COIL20.mat')
    X = mat['X']
    y = mat['Y']

    # subtract all values in y by 1 so you have [0, 19] instead of [1, 20]
    y = y - 1

    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.20, random_state=42)
    
    train_y = to_categorical(train_y, 20)
    test_y = to_categorical(test_y, 20)
    train_X = train_X.astype('float64')
    test_X = test_X.astype('float64')

    index_train = np.arange(train_X.shape[0])
    np.random.shuffle(index_train)

    index_test = np.arange(test_X.shape[0])
    np.random.shuffle(index_test)

    # random shuffle
    train_X = train_X[index_train, :]
    train_y = train_y[index_train, :]

    test_X = test_X[index_test, :]
    test_y = test_y[index_test, :]
    

    x_train_mean = np.mean(train_X, axis=0)
    x_train_std = np.std(train_X, axis=0)

    train_X = (train_X - x_train_mean) / x_train_std
    test_X = (test_X - x_train_mean) / x_train_std

    return train_X, train_y, test_X, test_y

def load_isolet():
    import pandas as pd
    df=pd.read_csv('./data/isolet.csv', sep=',',header=None)
    data = df.values
    X = data[1:,:-1].astype('float')
    y = [int(x.replace('\'','')) for x in data[1:,-1]]
    # subtract all values in y by 1 so you have [0, 25] instead of [1, 26]
    y = np.array(y) - 1

    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.20, random_state=42) 

    train_y = to_categorical(train_y, 26)
    test_y = to_categorical(test_y, 26)

    train_X = train_X.astype('float64')
    test_X = test_X.astype('float64')

    index_train = np.arange(train_X.shape[0])
    np.random.shuffle(index_train)

    index_test = np.arange(test_X.shape[0])
    np.random.shuffle(index_test)

    # random shuffle
    train_X = train_X[index_train, :]
    train_y = train_y[index_train, :]

    test_X = test_X[index_test, :]
    test_y = test_y[index_test, :]

    x_train_mean = np.mean(train_X, axis=0)
    x_train_std = np.std(train_X, axis=0)

    train_X = (train_X - x_train_mean) / x_train_std
    test_X = (test_X - x_train_mean) / x_train_std

    return train_X, train_y, test_X, test_y

def load_har():

    # TODO - check if correct?

    X_train = np.loadtxt('./data/UCI_HAR_Dataset/train/X_train.txt')
    y_train = np.loadtxt('./data/UCI_HAR_Dataset/train/y_train.txt')
    X_test =  np.loadtxt('./data/UCI_HAR_Dataset/test/X_test.txt')
    y_test =  np.loadtxt('./data/UCI_HAR_Dataset/test/y_test.txt')
    # subtract all values in y by 1 so you have [0, 5] instead of [1, 6]
    y_train = y_train - 1
    y_test = y_test - 1

    # to categorical
    y_train = to_categorical(y_train, 6)
    y_test = to_categorical(y_test, 6)

    print(f"Shapes are: X_train: {X_train.shape}, y_train: {y_train.shape}, X_test: {X_test.shape}, y_test: {y_test.shape}")

    return X_train, y_train, X_test, y_test

def load_pcmac():
    mat = loadmat('./data/PCMAC.mat')
    X = mat['X']
    y = mat['Y'] 
    # print(f" Shapes are: X: {X.shape}, y: {y.shape}")
    # print(X[0])
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.20, random_state=42) 

    # print(train_X)

    # print(train_y)

    # subtract all values in y by 1 so you have [0, 1] instead of [1, 2]
    train_y = train_y - 1
    test_y = test_y - 1

    train_y = to_categorical(train_y, 2)
    test_y = to_categorical(test_y, 2)

    train_X = train_X.astype('float64')
    test_X = test_X.astype('float64')

    index_train = np.arange(train_X.shape[0])
    np.random.shuffle(index_train)

    index_test = np.arange(test_X.shape[0])
    np.random.shuffle(index_test)

    # random shuffle
    train_X = train_X[index_train, :]
    train_y = train_y[index_train, :]

    test_X = test_X[index_test, :]
    test_y = test_y[index_test, :]

    # x_train_mean = np.mean(train_X, axis=0)
    # x_train_std = np.std(train_X, axis=0)

    # train_X = (train_X - x_train_mean) / x_train_std
    # test_X = (test_X - x_train_mean) / x_train_std

    return train_X, train_y, test_X, test_y

def load_smk():
    mat = loadmat('./data/SMK-CAN-187.mat')
    X = mat['X']
    y = mat['Y'] 
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.20, random_state=42) 

    # subtract all values in y by 1 so you have [0, 1] instead of [1, 2]
    train_y = train_y - 1
    test_y = test_y - 1


    train_y = to_categorical(train_y, 2)
    test_y = to_categorical(test_y, 2)

    train_X = train_X.astype('float64')
    test_X = test_X.astype('float64')

    index_train = np.arange(train_X.shape[0])
    np.random.shuffle(index_train)

    index_test = np.arange(test_X.shape[0])
    np.random.shuffle(index_test)

    # random shuffle
    train_X = train_X[index_train, :]
    train_y = train_y[index_train, :]

    test_X = test_X[index_test, :]
    test_y = test_y[index_test, :]

    x_train_mean = np.mean(train_X, axis=0)
    x_train_std = np.std(train_X, axis=0)

    train_X = (train_X - x_train_mean) / x_train_std
    test_X = (test_X - x_train_mean) / x_train_std

    return train_X, train_y, test_X, test_y

def load_gla():
    mat = loadmat('./data/GLA-BRA-180.mat', squeeze_me=True)
    X = mat["X"]
    y = mat["Y"]
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.20, random_state=42) 

    # subtract all values in y by 1 so you have [0, 1] instead of [1, 2]
    train_y = train_y - 1
    test_y = test_y - 1

    train_y = to_categorical(train_y, 4)
    test_y = to_categorical(test_y, 4)

    train_X = train_X.astype('float64')
    test_X = test_X.astype('float64')

    index_train = np.arange(train_X.shape[0])
    np.random.shuffle(index_train)

    index_test = np.arange(test_X.shape[0])
    np.random.shuffle(index_test)

    # random shuffle
    train_X = train_X[index_train, :]
    train_y = train_y[index_train, :]

    test_X = test_X[index_test, :]
    test_y = test_y[index_test, :]

    x_train_mean = np.mean(train_X, axis=0)
    x_train_std = np.std(train_X, axis=0)

    train_X = (train_X - x_train_mean) / x_train_std
    test_X = (test_X - x_train_mean) / x_train_std

    print(train_X[0])
    print(train_X.shape)

    return train_X, train_y, test_X, test_y
    
def load_synthetic(n_samples = 200, n_features = 500, n_classes = 2, n_informative = 50 , n_redundant = 0, i=42, n_clusters_per_class=2):
    """
    Function to generate my own, high-dimensional dataset

    Args:
        n_samples (int, optional): Number of samples. Defaults to 200.
        n_features (int, optional): Number of features. Defaults to 500.
        n_classes (int, optional): Number of classes. Defaults to 2.
        n_informative (int, optional): Number of informative features. Defaults to 50.
        n_redundant (int, optional): Number of redundant features. Defaults to 0.
        i (int, optional): Random state. Defaults to 42.
        n_clusters_per_class (int, optional): Number of clusters per class. Defaults to 2. (Increases the difficulty of the problem)

    Returns:
        train_X, train_y, test_X, test_y, val_X, val_y: The train, test and validation sets
    """

    X, y = make_classification(n_samples=n_samples, 
                               n_features=n_features, 
                               n_classes=n_classes, 
                               n_informative=n_informative, 
                               n_redundant=n_redundant,     
                               random_state=i, 
                               n_clusters_per_class=n_clusters_per_class,
                               shuffle=False,
                               class_sep=2,
                               flip_y=0.01)
    # convert y to categorical

    print(f"The informative features are the first {n_informative+n_redundant} features")
    print(f"They are given by: {X[:, :n_informative]}")

    y = to_categorical(y, n_classes)
    # train, test, val split
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.20, random_state=42)
    test_X, val_X, test_y, val_y = train_test_split(test_X, test_y, test_size=0.50, random_state=42)

    # check if shapes are correct. train should be 80% of the data, test and val should be 10% each, allow for rounding
    print(f"Expected train shape: {(n_samples*0.8, n_features)}, actual train shape: {train_X.shape}")
    print(f"Expected test shape: {(n_samples*0.1, n_features)}, actual test shape: {test_X.shape}")
    print(f"Expected val shape: {(n_samples*0.1, n_features)}, actual val shape: {val_X.shape}")

    print(f"shapes are: {train_X.shape}, {train_y.shape}, {test_X.shape}, {test_y.shape}")
    return train_X, train_y, test_X, test_y, val_X, val_y
















# TODO - make sure everything works with the function below (think about n_samples in the mnist and fashionmnist functions)

def get_data(args):
    if args.data == 'mnist':
        train_X, train_y, test_X, test_y = load_mnist() 

    elif args.data == 'FashionMnist':
        train_X, train_y, test_X, test_y = load_FashionMnist() 

    elif args.data == 'madelon':
        train_X, train_y, test_X, test_y = load_madelon() 

    elif args.data == 'coil':
        train_X, train_y, test_X, test_y = load_coil() 

    elif args.data == 'HAR':
        train_X, train_y, test_X, test_y = load_HAR() 

    elif args.data == 'Isolet':
        train_X, train_y, test_X, test_y = load_Isolet()

    elif args.data == 'GLA':
        train_X, train_y, test_X, test_y = load_GLA()

    elif args.data == 'USPS':
        train_X, train_y, test_X, test_y = load_USPS()

    elif args.data == 'SMK':
        train_X, train_y, test_X, test_y = load_SMK()

    elif args.data == 'PCMAC':
        train_X, train_y, test_X, test_y = load_PCMAC()
        
    return train_X, train_y, test_X, test_y

# NOTE - I do not use the function below, but is useful for seeing the correct input and output sizes for each dataset

def get_dataset_loader(args):
    if args.data == 'mnist':
        train_loader, valid_loader, test_loader = get_mnist_dataloaders(args, validation_split=args.valid_split)
        input_size = 28*28
        output_size = 10
    elif args.data == 'FashionMnist':
        train_loader, valid_loader, test_loader = get_FashionMnist_dataloaders(args, validation_split=args.valid_split)
        input_size = 28*28
        output_size = 10
    elif args.data == 'madelon':
        train_loader, valid_loader, test_loader = get_madelon_dataloaders(args, validation_split=args.valid_split)
        input_size = 500 
        output_size = 2   
    elif args.data == 'coil':
        train_loader, valid_loader, test_loader = get_coil_dataloaders(args, validation_split=args.valid_split)
        input_size = 1024
        output_size = 20    
    elif args.data == 'USPS':
        train_loader, valid_loader, test_loader = get_USPS_dataloaders(args, validation_split=args.valid_split)
        input_size = 256
        output_size = 10
    elif args.data == 'HAR':
        train_loader, valid_loader, test_loader = get_HAR_dataloaders(args, validation_split=args.valid_split)
        input_size = 561
        output_size = 6
    elif args.data == 'Isolet':
        train_loader, valid_loader, test_loader = get_Isolet_dataloaders(args, validation_split=args.valid_split)
        input_size = 617
        output_size = 26
    elif args.data == 'PCMAC':
        train_loader, valid_loader, test_loader = get_PCMAC_dataloaders(args, validation_split=args.valid_split)
        input_size = 3289
        output_size = 2
    elif args.data == 'SMK':
        train_loader, valid_loader, test_loader = get_SMK_dataloaders(args, validation_split=args.valid_split)
        input_size = 19993
        output_size = 2
    elif args.data == 'GLA':
        train_loader, valid_loader, test_loader = get_GLA_dataloaders(args, validation_split=args.valid_split)
        input_size = 49151
        output_size = 4