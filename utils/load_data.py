import numpy as np
import pandas as pd
import os
from concurrent.futures import ProcessPoolExecutor
from keras.datasets import cifar10, mnist
from keras.utils import np_utils
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
from torchvision import datasets, transforms
from PIL import Image

# TODO - clean up imports (remove unused ones)


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
    y_train = np_utils.to_categorical(y_train, 2)
    y_val = np_utils.to_categorical(y_val, 2)

    return x_train, y_train, x_val, y_val


# The MNIST database of handwritten digits.
def load_mnist_data(n_training_samples, n_testing_samples):

    # read MNIST data
    (x, y), (x_test, y_test) = mnist.load_data()

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
    x_train = x_train / 255.
    x_test = x_test / 255.
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train, x_test = x_train.reshape(x_train.shape[0], 784), x_test.reshape(x_test.shape[0], 784)
    print(y_test)
    return x_train, y_train, x_test, y_test


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

    return x_train, y_train, x_test, y_test


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

    train_y = np_utils.to_categorical(train_y, 10)
    test_y = np_utils.to_categorical(test_y, 10)
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
    
    train_y = np_utils.to_categorical(train_y, 20)
    test_y = np_utils.to_categorical(test_y, 20)
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

    train_y = np_utils.to_categorical(train_y, 26)
    test_y = np_utils.to_categorical(test_y, 26)

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
    y_train = np_utils.to_categorical(y_train, 6)
    y_test = np_utils.to_categorical(y_test, 6)

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

    train_y = np_utils.to_categorical(train_y, 2)
    test_y = np_utils.to_categorical(test_y, 2)

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


    train_y = np_utils.to_categorical(train_y, 2)
    test_y = np_utils.to_categorical(test_y, 2)

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

    train_y = np_utils.to_categorical(train_y, 4)
    test_y = np_utils.to_categorical(test_y, 4)

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
    
def load_synthetic(n_samples, n_features, n_classes, n_informative, n_redundant, i):
    """
    Function to generate my own, high-dimensional dataset
    """

    X, y = make_classification(n_samples=n_samples, n_features=n_features, n_classes=n_classes, n_informative=n_informative, n_redundant=n_redundant)
    # convert y to categorical

    y = np_utils.to_categorical(y, n_classes)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.6)

    print(f"shapes are: {x_train.shape}, {y_train.shape}, {x_test.shape}, {y_test.shape}")
    return x_train, y_train, x_test, y_test

def get_mnist_dataloaders(args, validation_split=0.0):
    """Creates augmented train, validation, and test data loaders."""
    normalize = transforms.Normalize((0.1307,), (0.3081,))
    transform = transforms.Compose([transforms.ToTensor(),normalize])

    full_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))

    valid_loader = None
    if validation_split > 0.0:
        split = int(np.floor((1.0-validation_split) * len(full_dataset)))
        train_dataset = DatasetSplitter(full_dataset,split_end=split)
        val_dataset = DatasetSplitter(full_dataset,split_start=split)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            args.batch_size,
            num_workers=8,
            pin_memory=True, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(
            val_dataset,
            args.test_batch_size,
            num_workers=2,
            pin_memory=True)
    else:
        train_loader = torch.utils.data.DataLoader(
            full_dataset,
            args.batch_size,
            num_workers=8,
            pin_memory=True, shuffle=True)

    print('Train loader length', len(train_loader))

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        args.test_batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True)

    return train_loader, valid_loader, test_loader

def get_FashionMnist_dataloaders(args, validation_split=0.0):
    """Creates augmented train, validation, and test data loaders."""
    normalize = transforms.Normalize((0.2859,), (0.3530,))
    transform = transforms.Compose([transforms.ToTensor(),normalize])

    full_dataset = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST('./data', train=False, transform=transform)

    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))

    valid_loader = None
    if validation_split > 0.0:
        split = int(np.floor((1.0-validation_split) * len(full_dataset)))
        train_dataset = DatasetSplitter(full_dataset,split_end=split)
        val_dataset = DatasetSplitter(full_dataset,split_start=split)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            args.batch_size,
            num_workers=8,
            pin_memory=True, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(
            val_dataset,
            args.test_batch_size,
            num_workers=2,
            pin_memory=True)
    else:
        train_loader = torch.utils.data.DataLoader(
            full_dataset,
            args.batch_size,
            num_workers=8,
            pin_memory=True, shuffle=True)

    print('Train loader length', len(train_loader))

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        args.test_batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True)

    return train_loader, valid_loader, test_loader

def get_madelon_dataloaders(args, validation_split=0.0):
    m, std = get_m_std_madelon()
    normalize = transforms.Normalize((m,), (std,))
    transform = transforms.Compose([transforms.ToTensor(),normalize])
    train = Madelon('./data', train=True, download=True, transform=transform)
    test = Madelon('./data', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        train,
        args.batch_size,
        num_workers=8,
        pin_memory=True, shuffle=True)

    print('Train loader length', len(train_loader))
    valid_loader = None
    test_loader = torch.utils.data.DataLoader(
        test,
        args.test_batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True)
    return train_loader, valid_loader, test_loader

def get_coil_dataloaders(args, validation_split=0.0):
    m, std = get_m_std_coil()
    normalize = transforms.Normalize((m,), (std,))
    transform = transforms.Compose([transforms.ToTensor(),normalize])
    train = custom_data('coil', './data', train=True, download=True, transform=transform)
    test = custom_data('coil', './data', train=False, download=True, transform=transform)
    train_loader, valid_loader, test_loader = get_loaders(train, test, args.batch_size, args.test_batch_size)
    return train_loader, valid_loader, test_loader

def get_USPS_dataloaders(args, validation_split=0.0):
    m, std = get_m_std_USPS()
    normalize = transforms.Normalize((m,), (std,))
    transform = transforms.Compose([transforms.ToTensor(),normalize])
    train = custom_data('USPS', './data', train=True, download=True, transform=transform)
    test = custom_data('USPS', './data', train=False, download=True, transform=transform)
    train_loader, valid_loader, test_loader = get_loaders(train, test, args.batch_size, args.test_batch_size)
    return train_loader, valid_loader, test_loader

def get_SMK_dataloaders(args, validation_split=0.0):
    m, std = get_m_std_SMK()
    normalize = transforms.Normalize((m,), (std,))
    transform = transforms.Compose([transforms.ToTensor(),normalize])
    train = custom_data('SMK', './data', train=True, download=True, transform=transform)
    test = custom_data('SMK', './data', train=False, download=True, transform=transform)
    train_loader, valid_loader, test_loader = get_loaders(train, test, args.batch_size, args.test_batch_size)
    return train_loader, valid_loader, test_loader

def get_PCMAC_dataloaders(args, validation_split=0.0):
    m, std = get_m_std_PCMAC()
    normalize = transforms.Normalize((m,), (std,))
    transform = transforms.Compose([transforms.ToTensor(),normalize])
    train = custom_data('PCMAC', './data', train=True, download=True, transform=transform)
    test = custom_data('PCMAC', './data', train=False, download=True, transform=transform)
    train_loader, valid_loader, test_loader = get_loaders(train, test, args.batch_size, args.test_batch_size)
    return train_loader, valid_loader, test_loader

def get_HAR_dataloaders(args, validation_split=0.0):
    m, std = get_m_std_HAR()
    normalize = transforms.Normalize((m,), (std,))
    transform = transforms.Compose([transforms.ToTensor(),normalize])
    train = custom_data('HAR', './data', train=True, download=True, transform=transform)
    test = custom_data('HAR', './data', train=False, download=True, transform=transform)
    train_loader, valid_loader, test_loader = get_loaders(train, test, args.batch_size, args.test_batch_size)
    return train_loader, valid_loader, test_loader

def get_Isolet_dataloaders(args, validation_split=0.0):
    m, std = get_m_std_Isolet()
    normalize = transforms.Normalize((m,), (std,))
    transform = transforms.Compose([transforms.ToTensor(),normalize])
    train = custom_data('Isolet', './data', train=True, download=True, transform=transform)
    test = custom_data('Isolet', './data', train=False, download=True, transform=transform)
    train_loader, valid_loader, test_loader = get_loaders(train, test, args.batch_size, args.test_batch_size)
    return train_loader, valid_loader, test_loader

def get_GLA_dataloaders(args, validation_split=0.0):
    m, std = get_m_std_GLA()
    normalize = transforms.Normalize((m,), (std,))
    transform = transforms.Compose([transforms.ToTensor(),normalize])
    train = custom_data('GLA', './data', train=True, download=True, transform=transform)
    test = custom_data('GLA', './data', train=False, download=True, transform=transform)
    train_loader, valid_loader, test_loader = get_loaders(train, test, args.batch_size, args.test_batch_size)
    return train_loader, valid_loader, test_loader

class custom_data(torch.utils.data.Dataset):
    def __init__(self, dataset, root, train=True, download=False, transform=None):
        if dataset == 'Isolet':
            X_train, y_train, X_test, y_test = read_Isolet()
        elif dataset == 'coil':
            X_train, y_train, X_test, y_test = read_coil()
        elif dataset == 'HAR':
            X_train, y_train, X_test, y_test = read_HAR()
        elif dataset == 'GLA':
            X_train, y_train, X_test, y_test = read_GLA()
        elif dataset == 'USPS':
            X_train, y_train, X_test, y_test = read_USPS()
        elif dataset == 'SMK':
            X_train, y_train, X_test, y_test = read_SMK()
        elif dataset == 'PCMAC':
            X_train, y_train, X_test, y_test = read_PCMAC()
        if train:
            self.data = X_train
            self.targets = y_train
        else:
            self.data = X_test
            self.targets = y_test
        self.transform = transform   

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Args: index (int): Index
        Returns: tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(np.array(img))

        return img, target 
       
class Madelon(torch.utils.data.Dataset):
    def __init__(self, root, train=True, download=False, transform=None):
        train_data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/madelon/MADELON/madelon_train.data'
        val_data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/madelon/MADELON/madelon_valid.data'
        train_resp_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/madelon/MADELON/madelon_train.labels'
        val_resp_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/madelon/madelon_valid.labels'
        test_data_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/madelon/MADELON/madelon_test.data'
        if train:
            self.data = np.loadtxt(urllib2.urlopen(train_data_url))
            self.targets = np.loadtxt(urllib2.urlopen(train_resp_url))
        else:
            self.data =  np.loadtxt(urllib2.urlopen(val_data_url))
            self.targets =  np.loadtxt(urllib2.urlopen(val_resp_url))
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Args: index (int): Index
        Returns: tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        # normalize from [-1, 1] to [0, 1]
        target = (target + 1) / 2

        if self.transform is not None:
            img = self.transform(np.array(img))

        return img, target

def load_mnist():

    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    train_X = train_X.reshape((train_X.shape[0],train_X.shape[1]*train_X.shape[2]))
    test_X  = test_X.reshape((test_X.shape[0],test_X.shape[1]*test_X.shape[2]))
    train_X = train_X.astype('float32')
    test_X  = test_X.astype('float32')
    scaler = preprocessing.StandardScaler().fit(train_X)
    train_X = scaler.transform(train_X)
    test_X = scaler.transform(test_X)
    print(f"X_test shape = {str(test_X.shape)}")
    return train_X, train_y, test_X, test_y

def load_FashionMnist():

    (train_X, train_y), (test_X, test_y) = fashion_mnist.load_data()
    train_X = train_X.reshape((train_X.shape[0],train_X.shape[1]*train_X.shape[2]))
    test_X  = test_X.reshape((test_X.shape[0],test_X.shape[1]*test_X.shape[2]))
    train_X = train_X.astype('float32')
    test_X  = test_X.astype('float32')
    scaler = preprocessing.StandardScaler().fit(train_X)
    train_X = scaler.transform(train_X)
    test_X = scaler.transform(test_X)
    print(f"X_test shape = {str(test_X.shape)}")
    return train_X, train_y, test_X, test_y















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