# Authors: Decebal Constantin Mocanu et al.;
# Code associated with SCADS Summer School 2020 tutorial "	Scalable Deep Learning Tutorial"; https://www.scads.de/de/summerschool2020
# This is a pre-alpha free software and was tested in Windows 10 with Python 3.7.6, Numpy 1.17.2, SciPy 1.4.1, Numba 0.48.0

# If you use parts of this code please cite the following article:
# @article{Mocanu2018SET,
#   author  =    {Mocanu, Decebal Constantin and Mocanu, Elena and Stone, Peter and Nguyen, Phuong H. and Gibescu, Madeleine and Liotta, Antonio},
#   journal =    {Nature Communications},
#   title   =    {Scalable Training of Artificial Neural Networks with Adaptive Sparse Connectivity inspired by Network Science},
#   year    =    {2018},
#   doi     =    {10.1038/s41467-018-04316-3}
# }

# If you have space please consider citing also these articles

# @phdthesis{Mocanu2017PhDthesis,
#   title     =    "Network computations in artificial intelligence",
#   author    =    "D.C. Mocanu",
#   year      =    "2017",
#   isbn      =    "978-90-386-4305-2",
#   publisher =    "Eindhoven University of Technology",
# }

# @article{Liu2019onemillion,
#   author  =    {Liu, Shiwei and Mocanu, Decebal Constantin and Mocanu and Ramapuram Matavalam, Amarsagar Reddy and Pei, Yulong Pei and Pechenizkiy, Mykola},
#   journal =    {arXiv:1901.09181},
#   title   =    {Sparse evolutionary Deep Learning with over one million artificial neurons on commodity hardware},
#   year    =    {2019},
# }

# We thank to:
# Thomas Hagebols: for performing a thorough analyze on the performance of SciPy sparse matrix operations
# Ritchie Vink (https://www.ritchievink.com): for making available on Github a nice Python implementation of fully connected MLPs. This SET-MLP implementation was built on top of his MLP code:
#                                             https://github.com/ritchie46/vanilla-machine-learning/blob/master/vanilla_mlp.py

from argparser import get_parser
from numba import njit, prange
from plot import plot_features, plot_importances
from scipy.sparse import lil_matrix
from scipy.sparse import coo_matrix
from scipy.sparse import dok_matrix
from test import svm_test # new 
from utils.nn_functions import *

from utils.load_data import load_fashion_mnist_data, load_cifar10_data, load_madelon_data, load_mnist_data, load_usps, load_coil, load_isolet, load_har, load_smk, load_gla, load_synthetic
from set_mlp_sequential import SET_MLP
from wasap_sgd.train.monitor import Monitor

import copy
import datetime
import json
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pprint
import shutil
import sys
import time
import wandb
wandb.login()


if not os.path.exists('./models'): os.mkdir('./models')
if not os.path.exists('./logs'): os.mkdir('./logs')
logger = None

stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
sys.stderr = stderr


@njit(parallel=True, fastmath=True, cache=True)
def backpropagation_updates_numpy(a, delta, rows, cols, out):
    for i in prange(out.shape[0]):
        s = 0
        for j in range(a.shape[0]):
            s += a[j, rows[i]] * delta[j, cols[i]]
        out[i] = s / a.shape[0]


@njit(fastmath=True, cache=True)
def find_first_pos(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx


@njit(fastmath=True, cache=True)
def find_last_pos(array, value):
    idx = (np.abs(array - value))[::-1].argmin()
    return array.shape[0] - idx


@njit(fastmath=True, cache=True)
def compute_accuracy(activations, y_test):
    correct_classification = 0
    for j in range(y_test.shape[0]):
        if np.argmax(activations[j]) == np.argmax(y_test[j]):
            correct_classification += 1
    return correct_classification / y_test.shape[0]


@njit(fastmath=True, cache=True)
def dropout(x, rate):
    noise_shape = x.shape
    noise = np.random.uniform(0., 1., noise_shape)
    keep_prob = 1. - rate
    scale = np.float32(1 / keep_prob)
    keep_mask = noise >= rate
    return x * scale * keep_mask, keep_mask

def createSparseWeights_II(epsilon,noRows,noCols):
    # generate an Erdos Renyi sparse weights mask
    weights = lil_matrix((noRows, noCols))
    for _ in range(epsilon * (noRows + noCols)):
        weights[np.random.randint(0, noRows), np.random.randint(0, noCols)] = np.float32(np.random.randn()/10)
    print("Create sparse matrix with ", weights.getnnz(), " connections and ",
          (weights.getnnz()/(noRows * noCols))*100, "% density level")
    weights = weights.tocsr()
    return weights

def create_sparse_weights(epsilon, n_rows, n_cols, weight_init):
    # He uniform initialization
    if weight_init == 'he_uniform':
        limit = np.sqrt(6. / float(n_rows))

    # Xavier initialization
    if weight_init == 'xavier':
        limit = np.sqrt(6. / (float(n_rows) + float(n_cols)))

    if weight_init == 'neuron_importance':
        limit = np.sqrt(6. / float(n_rows))
        # TODO: We might want to initialize the weights differently (in a smart way) but I am not sure how to do it yet

    mask_weights = np.random.rand(n_rows, n_cols)
    prob = 1 - (epsilon * (n_rows + n_cols)) / (n_rows * n_cols)  # normal to have 8x connections

    # generate an Erdos Renyi sparse weights mask
    weights = lil_matrix((n_rows, n_cols))
    n_params = np.count_nonzero(mask_weights[mask_weights >= prob])
    weights[mask_weights >= prob] = np.random.uniform(-limit, limit, n_params)
    print("Create sparse matrix with ", weights.getnnz(), " connections and ",
          (weights.getnnz() / (n_rows * n_cols)) * 100, "% density level")
    weights = weights.tocsr()
    return weights

def array_intersect(a, b):
    # this are for array intersection
    n_rows, n_cols = a.shape
    dtype = {
        'names': [f'f{i}' for i in range(n_cols)],
        'formats': n_cols * [a.dtype],
    }
    return np.in1d(a.view(dtype), b.view(dtype))  # boolean return

def select_input_neurons(weights, k):
    """
    Function to select the k most important neurons in a sparse matrix, and returns a sparse matrix with the same shape as weights, but with only the k most important connections.
    Args:   
            weights: csr_matrix
            k: number of neurons to select

    Returns:
            important_neurons_idx: indices of the k most important neurons
            important_neurons: csr_matrix with only the k most important connections
    """

    sum_weights = np.abs(weights).sum(axis=1)  # get the sum of the absolute values of the weights for each neuron
    print(f"The input neuron with the highest weight is: {np.argmax(sum_weights)}")
    important_neurons_idx = np.argsort(sum_weights, axis=0)[::-1][:k]
    # important_neurons = np.abs(copy.deepcopy(network.w[1])).sum(axis=1)

    return important_neurons_idx, sum_weights

def setup_logger(args): 
    global logger
    if logger is None:
        logger = logging.getLogger()
    else:  # wish there was a logger.close()
        for handler in logger.handlers[:]:  # make a copy of the list
            logger.removeHandler(handler)
    args_copy = copy.deepcopy(args)
    # copy to get a clean hash
    # use the same log file hash if iterations or verbose are different
    # these flags do not change the results
    args_copy.iters = 1
    args_copy.verbose = False
    args_copy.log_interval = 1
    log_path = './logs/{0}.log'.format('ae')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt='%(asctime)s: %(message)s', datefmt='%H:%M:%S')
    fh = logging.FileHandler(log_path)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

def print_and_log(msg):
    global logger
    print(msg)
    logger.info(msg)

def evaluate_fs(x_train, x_test, y_train, y_test, selected_features):
    """
    Function to evaluate the feature selection using the input layers' weights.

    :param x_train: (array) Training input
    :param x_test: (array) Test input

    :param y_train: (array) Correct training output
    :param y_test: (array) Correct test output

    :return: (float) Classification accuracy
    """
            # change x_train and x_test to only have the selected features
    # print(selected_features)
    x_train_new = np.squeeze(x_train[:, selected_features])
    x_test_new = np.squeeze(x_test[:, selected_features])
    # change y_train and y_test from one-hot to single label
    y_train_new = np.argmax(y_train, axis=1)
    y_test_new = np.argmax(y_test, axis=1)

    return round(sum(svm_test(x_train_new, y_train_new, x_test_new, y_test_new) for _ in range(5)) / 5, 4)

# TODO - ship this function into load_data
def get_data(dataset):
    """
    Function to load the data from the dataset.

    :param dataset: (string) Name of the dataset

    :return x_train: (array) Training input
    :return y_train: (array) Correct training output
    :return x_test: (array) Test input
    :return y_test: (array) Correct test output

    """

    if dataset == 'FashionMnist':
        x_train, y_train, x_test, y_test = load_fashion_mnist_data(no_training_samples, no_testing_samples)
    elif dataset == 'mnist':
        x_train, y_train, x_test, y_test = load_mnist_data(no_training_samples, no_testing_samples)
    elif dataset == 'madelon':
        x_train, y_train, x_test, y_test = load_madelon_data()
    elif dataset == 'usps':
        x_train, y_train, x_test, y_test = load_usps()
    elif dataset == 'coil':
        x_train, y_train, x_test, y_test = load_coil()
    elif dataset == 'isolet':
        x_train, y_train, x_test, y_test = load_isolet()
    elif dataset == 'har':
        x_train, y_train, x_test, y_test = load_har()
    elif dataset == 'smk':
        x_train, y_train, x_test, y_test = load_smk()
    elif dataset == 'gla':
        x_train, y_train, x_test, y_test = load_gla()
    elif dataset == 'synthetic':
        x_train, y_train, x_test, y_test = load_synthetic(n_samples=400, n_features=1000, n_classes=2, 
                                                          n_informative=50, n_redundant=0, i=42)
    else:
        raise ValueError("Unknown dataset")
    return x_train, y_train, x_test, y_test

class SET_MLP:
    def __init__(self, dimensions, activations, input_pruning, importance_pruning, lamda, epsilon=20, weight_init='neuron_importance', config=None):
        """
        :param dimensions: (tpl/ list) Dimensions of the neural net. (input, hidden layer, output)
        :param activations: (tpl/ list) Activations functions.
        Example of three hidden layer with
        - 3312 input features
        - 3000 hidden neurons
        - 3000 hidden neurons
        - 3000 hidden neurons
        - 5 output classes
        layers -->    [1,        2,     3,     4,     5]
        ----------------------------------------
        dimensions =  (3312,     3000,  3000,  3000,  5)
        activations = (          Relu,  Relu,  Relu,  Sigmoid)
        """

        self.n_layers = len(dimensions)
        self.loss = None
        self.dropout_rate = 0.  # dropout rate
        self.learning_rate = None
        self.momentum = None
        self.weight_decay = None
        self.epsilon = epsilon  # control the sparsity level as discussed in the paper
        self.zeta = None  # the fraction of the weights removed
        self.dimensions = dimensions
        self.weight_init = weight_init
        self.save_filename = ""
        self.input_layer_connections = []
        self.monitor = None
        self.importance_pruning = importance_pruning
        self.input_pruning = input_pruning
        self.lamda = lamda
        self.config = config

        self.training_time = 0
        self.testing_time = 0
        self.evolution_time = 0

        # Weights and biases are initiated by index. For a one hidden layer net you will have a w[1] and w[2]
        self.w = {}
        self.b = {}
        self.pdw = {}
        self.pdd = {}

        # Activations are also initiated by index. For the example we will have activations[2] and activations[3]
        self.activations = {}
        for i in range(len(dimensions) - 1):
            if self.weight_init == 'normal':
                self.w[i + 1] = createSparseWeights_II(self.epsilon, dimensions[i],
                                                       dimensions[i + 1])  # create sparse weight matrices
            else:
                self.w[i + 1] = create_sparse_weights(self.epsilon, dimensions[i], dimensions[i + 1],
                                                      weight_init=self.weight_init)  # create sparse weight matrices
            self.b[i + 1] = np.zeros(dimensions[i + 1], dtype='float32')
            self.activations[i + 2] = activations[i]

        # Added by Matthijs
        self.layer_importances = {} # NOTE (Matthijs): Like this or as a list?
        self._init_layer_importances()

    def _init_layer_importances(self):
        """
        Initialize the layer importances for each layer

        :return: None
        """

        print("Initializing layer importances")
        print("================================")
        # print(f"We are looking at this amount of layers: {self.n_layers}")
        for i in range(1, self.n_layers+1):
            # print(f"Initializing the importances for layer {i}")
            if i == 1: # Input layer
                # print(f"{i} == 1")
                # print(f"The shape of the weight matrix we are looking at is: {self.w[i].shape}")
                self.layer_importances[i] = np.zeros(copy.deepcopy(self.w[i]).shape[0], dtype='float32')
                # print(f"The shape of the input layer is: {self.w[i].shape[0]}, which is layer {i}")
            if i == self.n_layers: # Output layer
                # print(f"{i} == {self.n_layers}")
                # print(f"The shape of the weight matrix we are looking at is: {self.w[i-1].shape}")
                self.layer_importances[i] = np.zeros(copy.deepcopy(self.w[i-1]).shape[1], dtype='float32')
                # print(f"The shape of the output layer is: {self.w[i-1].shape[1]}, which is layer {i}")
            # for the hidden layers all neurons have the same importance
            if i not in [1, self.n_layers]: # All other layers (hidden layers)
                # print(f"{i} not in [1, {self.n_layers}]")
                # print(f"The shape of the weight matrix we are looking at is: {self.w[i].shape}")
                self.layer_importances[i] = np.ones(copy.deepcopy(self.w[i]).shape[0], dtype='float32')
                # print(f"The shape of the hidden layer is: {self.w[i].shape[0]}, which is layer {i}")

        # print(f"The shape of the layer importances is: {self.layer_importances}")          

    def _update_layer_importances(self):
        """
        Update the layer importances for each layer

        :return: None
        """
        start_time = time.time()
        lamda = self.lamda
        temp = np.array(copy.deepcopy(self.input_sum)).reshape(-1)
        
        # update the layer importances for the input layer
        self.layer_importances[1] = self.layer_importances[1] * lamda + temp * (1 - lamda) # NOTE (Matthijs): Only update input layer for now
        # updating the layer importances for the hidden layers
        hidden_importance = False # will test later
        if hidden_importance:
            for i in range(2, self.n_layers):
                temp2 = np.array(np.sum(np.abs(copy.deepcopy(self.w[i])), axis=1)).reshape(-1)
                # print(temp2.shape) # should be (nhidden, ), thus 200, for now
                self.layer_importances[i] = self.layer_importances[i] * lamda + temp2 * (1 - lamda)



        # print(f"Updating the layer importances took {round(time.time() - start_time, 2)} seconds")

    def _feed_forward(self, x, drop=False):
        """
        Execute a forward feed through the network.
        :param x: (array) Batch of input data vectors.
        :return: (tuple) Node outputs and activations per layer. The numbering of the output is equivalent to the layer numbers.
        """
        # w(x) + b
        z = {}

        # activations: f(z)
        a = {1: x}  # First layer has no activations as input. The input x is the input.
        masks = {}

        for i in range(1, self.n_layers):
            z[i + 1] = a[i] @ self.w[i] + self.b[i]
            a[i + 1] = self.activations[i + 1].activation(z[i + 1])
            if drop and i < self.n_layers - 1:
                # apply dropout
                a[i + 1], keep_mask = dropout(a[i + 1], self.dropout_rate)
                masks[i + 1] = keep_mask

        return z, a, masks

    def _back_prop(self, z, a, masks, y_true):
        """
        The input dicts keys represent the layers of the net.
        a = { 1: x,
              2: f(w1(x) + b1)
              3: f(w2(a2) + b2)
              4: f(w3(a3) + b3)
              5: f(w4(a4) + b4)
              }
        :param z: (dict) w(x) + b
        :param a: (dict) f(z)
        :param y_true: (array) One hot encoded truth vector.
        :return:
        """
        keep_prob = 1.
        if self.dropout_rate > 0:
            keep_prob = np.float32(1. - self.dropout_rate)

        # Determine partial derivative and delta for the output layer.
        # delta output layer
        delta = self.loss.delta(y_true, a[self.n_layers])
        # print(delta.shape) # (128,10)
        dw = coo_matrix(self.w[self.n_layers - 1], dtype='float32')
        # compute backpropagation updates
        backpropagation_updates_numpy(a[self.n_layers - 1], delta, dw.row, dw.col, dw.data)

        update_params = {
            self.n_layers - 1: (dw.tocsr(),  np.mean(delta, axis=0))
        }

        # In case of three layer net will iterate over i = 2 and i = 1
        # Determine partial derivative and delta for the rest of the layers.
        # Each iteration requires the delta from the previous layer, propagating backwards.
        for i in reversed(range(2, self.n_layers)):
            # dropout for the backpropagation step
            if keep_prob != 1:
                delta = (delta @ self.w[i].transpose()) * self.activations[i].prime(z[i])
                delta = delta * masks[i]
                delta /= keep_prob
            else:
                delta = (delta @ self.w[i].transpose()) * self.activations[i].prime(z[i])

            dw = coo_matrix(self.w[i - 1], dtype='float32')

            # compute backpropagation updates
            backpropagation_updates_numpy(a[i - 1], delta, dw.row, dw.col, dw.data)

            update_params[i - 1] = (dw.tocsr(),  np.mean(delta, axis=0))
        for k, v in update_params.items():
            self._update_w_b(k, v[0], v[1])

    def _update_w_b(self, index, dw, delta):
        """
        Update weights and biases.
        :param index: (int) Number of the layer
        :param dw: (array) Partial derivatives
        :param delta: (array) Delta error.
        """

        # perform the update with momentum
        if index not in self.pdw:
            self.pdw[index] = - self.learning_rate * dw
            self.pdd[index] = - self.learning_rate * delta
        else:
            self.pdw[index] = self.momentum * self.pdw[index] - self.learning_rate * dw
            self.pdd[index] = self.momentum * self.pdd[index] - self.learning_rate * delta

        self.w[index] += self.pdw[index] - self.weight_decay * self.w[index]
        self.b[index] += self.pdd[index] - self.weight_decay * self.b[index]

    def _input_pruning(self, epoch, i):
        zerow_before = np.count_nonzero(self.input_sum == 0)

        if zerow_before > self.input_sum.shape[0] - (self.config.K * 2):
            print(f"WARNING: No more neurons to prune, since {zerow_before} > {self.input_sum.shape[0] - (self.config.K * 2)}")
            self.input_pruning = False
            # time.sleep(10) 
        zerow_pct = zerow_before / self.input_sum.shape[0] * 100
        print(f"Before pruning in epoch {epoch} The amount of neurons without any incoming weights is: {zerow_before}, \
                which is {zerow_pct}% of the total neurons.")

        start_input_pruning = datetime.datetime.now()

        sum_incoming_weights = np.array(copy.deepcopy(self.input_sum))
    
        curr_percentile = 20 + ((epoch/self.config.epochs) * 70)
        t = np.percentile(sum_incoming_weights, curr_percentile)
        if t == 0:
            # Find a value for the percentile such that you slowly prune the neurons over the epochs until you reach 200 neurons
            # It should be a function of epoch/n_epochs
            curr_percentile = 20 + ((epoch/self.config.epochs) * 70)
            t_2 = np.percentile(sum_incoming_weights, curr_percentile)
            # prune the weights that are lower than 
            print(f"\n NOTE: t is 0, setting it to t_2 : {t_2}, which prunes the {curr_percentile}th percentile of the weights")
            t = t_2
        # print(t)
        # breakpoint()
        sum_incoming_weights = np.where(sum_incoming_weights <= t, 0, sum_incoming_weights)
        # print(sum_incoming_weights)
        ids = np.argwhere(sum_incoming_weights == 0)
        # print("ids", ids.shape)
        # overlay the ids on a 28*28 matrix and set the values to 0
        if self.config.plotting and (self.config.data == 'MNIST' | self.config.data == 'FashionMnist'):
            matrix2828 = np.ones((28,28))
            matrix2828 = matrix2828.flatten()
            matrix2828[ids] = 0
            matrix2828 = matrix2828.reshape((28,28))
            # plot the matrix with the epoch number as title without interruping the training
            plt.imshow(matrix2828, cmap='gray')
            plt.title(f"Epoch {epoch}")
            # save into the pruning directory
            plt.savefig(f"pruning/{epoch}.png")
            plt.close()

        weights = self.w[i].tolil()
        pdw = self.pdw[i].tolil()

        weights[ids[:,0], :] = 0
        pdw[ids[:,0], :] = 0

        self.w[i] = weights.tocsr()
        self.pdw[i] = pdw.tocsr()

        print(f"Input pruning took {datetime.datetime.now() - start_input_pruning} seconds")

    def _importance_pruning(self, epoch, i):
        """
        Function to perform pruning on the weights of the hidden layer.
        Parameters:
        :param epoch: (int) Current epoch
        :param i: (int) Current layer

        Returns:
        None
        """

        sum_incoming_weights = np.abs(copy.deepcopy(self.w[i])).sum(axis=0)

        t = np.percentile(sum_incoming_weights, 20, axis=1)
        sum_incoming_weights = np.where(sum_incoming_weights <= t, 0, sum_incoming_weights)
        
        # print(sum_incoming_weights)
        ids = np.argwhere(sum_incoming_weights == 0)
        # print("ids", ids.shape)
        weights = self.w[i].tolil()
        pdw = self.pdw[i].tolil()

        weights[:, ids[:,1]] = 0
        pdw[:, ids[:,1]] = 0

        self.w[i] = weights.tocsr()
        self.pdw[i] = pdw.tocsr()

    def fit(self, x, y_true, x_test, y_test, loss, epochs, batch_size, eval_epoch, run, metrics, 
            learning_rate=1e-3, momentum=0.9, weight_decay=0.0002, zeta=0.3, dropoutrate=0., 
            testing=True, save_filename="", monitor=False, config=None
            ):
        """
        Train the network.

        :param x: (array) Containing parameters
        :param y_true: (array) Containing one hot encoded labels.

        :return (array) A 2D array of metrics (epochs, 3).
        """
        if x.shape[0] != y_true.shape[0]:
            raise ValueError("Length of x and y arrays don't match")

        self.monitor = Monitor(save_filename=save_filename) if monitor else None

        # Initiate the loss object with the final activation function
        self.loss = loss()
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.zeta = zeta
        self.dropout_rate = dropoutrate
        self.save_filename = save_filename
        if self.config.plotting:
            self.input_layer_connections.append(self.get_core_input_connections())
            np.savez_compressed(self.save_filename + "_input_connections.npz",
                                inputLayerConnections=self.input_layer_connections)

        # 1-d sum of the absolute values of the input weights
        min_loss = 1e9
        max_accuracy_topk = 0
        maximum_accuracy = 0
        early_stopping_counter = 0
        for i in range(epochs):
            # Shuffle the data
            self.input_weights = copy.deepcopy(self.w[1])
            _, self.input_sum = select_input_neurons(self.input_weights, config.K)
            # print(f"The shape of the input layer weights is: {self.input_sum.shape}")
            if i == 0 or i == 1 or i == 5 or i == 10 or i == 25 or i % eval_epoch == 0 or i == epochs:
                start_time = time.time()
                print("In eval loop")
                importances_for_eval = copy.deepcopy(self.input_sum)
                importances_for_eval = pd.DataFrame(importances_for_eval).to_csv(header=False, index=False)

                importances_path = f"importances/importances_{str(config.data)}_{i}_MLP.csv"
                if not os.path.exists(os.path.dirname(importances_path)):
                    os.makedirs(os.path.dirname(importances_path))

                with open(importances_path, 'w') as f:
                    f.write(importances_for_eval)

                        # Round the time to 5 decimals
                        # print(f"Choosing the features before epoch {i} took {round(time.time() - start_time, 5)} seconds")

            seed = np.arange(x.shape[0])
            np.random.shuffle(seed)
            x_ = x[seed]
            y_ = y_true[seed]

            if self.monitor:
                self.monitor.start_monitor()

            # training
            t1 = datetime.datetime.now()

            for j in range(x.shape[0] // batch_size):
                k = j * batch_size
                l = (j + 1) * batch_size
                if config.update_batch == True:
                    # print(config.update_batch) 
                    z, a, masks = self._feed_forward(x_[k:l], True)

                    self._back_prop(z, a, masks,  y_[k:l])

                    self._update_layer_importances()

                    if i < epochs - 1:
                        # self.weights_evolution_I() # this implementation is more didactic, but slow.
                        self.weights_evolution_II(i)  # this implementation has the same behaviour as the one above, but it is much faster.
                else: # epoch update instead of batch update 
                    z, a, masks = self._feed_forward(x_[k:l], True)

                    self._back_prop(z, a, masks,  y_[k:l])
                        # Importance pruning (on the hidden layers)
            if x.shape[0] % batch_size != 0: # if the batch size is not a multiple of the number of samples, take the last batch
                k = (x.shape[0] // batch_size) * batch_size
                l = x.shape[0]
                if config.update_batch == True:
                    z, a, masks = self._feed_forward(x_[k:l], True)

                    self._back_prop(z, a, masks,  y_[k:l])
                    self._update_layer_importances()

                    if i < epochs - 1:
                        # self.weights_evolution_I() # this implementation is more didactic, but slow.
                        self.weights_evolution_II(i)
                else:
                    z, a, masks = self._feed_forward(x_[k:l], True)
                    self._back_prop(z, a, masks,  y_[k:l])


            if self.importance_pruning and i % 5 == 0 and i > 5:
                # print(f"Importance pruning in layer {1}, epoch {i}")
                self._importance_pruning(epoch=i, i=1)

            # Input neuron pruning (on the input layer)
            if self.input_pruning and i > 2:
                # print(f"Input pruning in layer {1}, epoch {i}")
                self._input_pruning(epoch=i, i=1)

            t2 = datetime.datetime.now()

            if self.monitor:
                self.monitor.stop_monitor()

            if not config.update_batch:
                print("Updating layer importances...")
                self._update_layer_importances()
            # print(self.layer_importances[1])

            print("\nSET-MLP Epoch ", i)
            print("Training time: ", t2 - t1)
            self.training_time += (t2 - t1).seconds

            # test model performance on the test data at each epoch
            # this part is useful to understand model performance and can be commented for production settings
            
            if testing:
                t3 = datetime.datetime.now()
                accuracy_test, activations_test = self.predict(x_test, y_test)
                accuracy_train, activations_train = self.predict(x, y_true)
                t4 = datetime.datetime.now()
                maximum_accuracy = max(maximum_accuracy, accuracy_test)
                loss_test = self.loss.loss(y_test, activations_test)
                loss_train = self.loss.loss(y_true, activations_train)
                metrics[run-1, i, 0] = loss_train
                metrics[run-1, i, 1] = loss_test
                metrics[run-1, i, 2] = accuracy_train
                metrics[run-1, i, 3] = accuracy_test
                print("calculated metrics")
                wb_metrics = {
                    'loss_train': loss_train,
                    'loss_test': loss_test,
                    'accuracy_train': accuracy_train,
                    'accuracy_test': accuracy_test,
                }
                wandb.log(wb_metrics)
                # print(metrics)

                print(f"Testing time: {t4 - t3}; \n"
                      f"Loss train: {round(loss_train, 3)}; \n"
                      f"Loss test: {round(loss_test, 3)}; \n"
                      f"Minimum test loss: {round(min_loss, 3)}; \n"
                      f"Accuracy test: {round(accuracy_test, 3)}; \n"
                      f"Maximum accuracy val: {round(maximum_accuracy, 3)} \n"
                      f"max_accuracy_topk: {round(max_accuracy_topk, 3)} \n")
                self.testing_time += (t4 - t3).seconds

                if i % 2 == 0:
                    selected_features, importances = select_input_neurons(copy.deepcopy(self.w[1]), config.K)
                    accuracy_topk = evaluate_fs(x, x_test, y_true, y_test, selected_features)
                    wandb.log({"accuracy_topk": accuracy_topk,
                               "epoch": i})
                    if accuracy_topk > max_accuracy_topk:
                        max_accuracy_topk = accuracy_topk
                        early_stopping_counter = 0

                # If the loss_test does not improve for 25 epochs, stop the training
                if loss_test < min_loss:
                    min_loss = loss_test
                    early_stopping_counter = 0
                print(f"Early stopping counter: {early_stopping_counter}")
                if loss_test > min_loss:
                    early_stopping_counter += 1
                    if early_stopping_counter >= epochs/5:
                        print(f"Early stopping run {run} epoch {i}")
                        # fill metrics with nan
                        metrics[run-1, i:, :] = np.nan
                        # if last run, save metrics
                        if run == self.config.runs - 1:
                            print(metrics[run-1, :, 0])
                            filename = f"results/metrics/metrics_{config.data}_{config.epochs}epochs_batchupdate{config.update_batch}_{self.weight_init}_importancepruning{config.importance_pruning}_inputpruning{config.input_pruning}_zeta{config.zeta}.npy"
                            if os.path.exists(filename):
                                with open(filename, "rb") as f:
                                    metrics_old = np.load(f)
                                metrics = np.concatenate((metrics_old, metrics), axis=0)
                            with open(filename, "wb") as f:
                                np.save(f, metrics)
                        break



            t5 = datetime.datetime.now()
            if i < epochs - 1 and not config.update_batch:  # do not change connectivity pattern after the last epoch
                # self.weights_evolution_I() # this implementation is more didactic, but slow.
                self.weights_evolution_II(i)  # this implementation has the same behaviour as the one above, but it is much faster.
            t6 = datetime.datetime.now()
            print("Weights evolution time ", t6 - t5)

                    # # save performance metrics values in a file
                    # if self.save_filename != "":
                    #     np.savetxt(self.save_filename +".txt", metrics)

                    # if self.save_filename != "" and self.monitor:
                    #     with open(self.save_filename + "_monitor.json", 'w') as file:
                    #         file.write(json.dumps(self.monitor.get_stats(), indent=4, sort_keys=True, default=str))
        # print(self.get_core_input_connections())

        # Save the metrics to a file

        if run+1 == config.runs:
            filename = f"results/metrics/metrics_{config.data}_{config.epochs}epochs_batchupdate{config.update_batch}_{self.weight_init}_importancepruning{config.importance_pruning}_inputpruning{config.input_pruning}_zeta{config.zeta}.npy"
            if os.path.exists(filename):
                with open(filename, "rb") as f:
                    metrics_old = np.load(f)
                metrics = np.concatenate((metrics_old, metrics), axis=0)
            with open(filename, "wb") as f:
                np.save(f, metrics)
                
            self._plot_loss_from_metrics(metrics, run)
        return metrics 

    def _plot_loss_from_metrics(self, metrics, run):
        # plot the train and test loss using metrics array. Somehow it should reflect the early stopping with nan values
        # initialize colors
        # select the last run runs from metrics
        print(metrics.shape)
        metrics = metrics[-max(0, runs):, :, :]
        print(metrics.shape)
        colors = plt.cm.jet(np.linspace(0, 1, run+1))
        # give each run a different color (based on i)
        # dashed line for train loss, solid line for test loss
        # the line should cut off/stop at the nan values
        # no legend
        # it should select the runs backwards from metrics (i.e. the last run should be the first one plotted, and it should only plot the last run runs)
        for i in range(run+1):
            print(f"Amount of nan values in run {i}: {np.count_nonzero(np.isnan(metrics[i, :, 0]))}")
            # print(metrics[i, :, :])
            plt.plot(metrics[i, :, 0], color=colors[i], linestyle="--", alpha=0.2)
            plt.plot(metrics[i, :, 1], color=colors[i], linestyle="-", alpha=0.2)

        # instead of the above, plot the mean and std of the loss for each epoch (result should be shape (epochs, )), but deal with the nan values
        mean_train_loss = np.nanmean(metrics[:, :, 0], axis=0)
        mean_test_loss = np.nanmean(metrics[:, :, 1], axis=0)
        std_train_loss = np.nanstd(metrics[:, :, 0], axis=0)
        std_test_loss = np.nanstd(metrics[:, :, 1], axis=0)

        plt.plot(mean_train_loss, label="Train loss")
        plt.plot(mean_test_loss, label="Test loss")

        plt.fill_between(range(mean_train_loss.shape[0]), mean_train_loss - std_train_loss  , mean_train_loss + std_train_loss, alpha=0.2)
        plt.fill_between(range(mean_test_loss.shape[0]), mean_test_loss - std_test_loss, mean_test_loss + std_test_loss, alpha=0.2)

        plt.legend()

        # do 20 x ticks equally spaced 
        plt.xticks(np.arange(0, self.config.epochs, 20))
        # Fix axis between 0 and 1
        plt.ylim(0, 2)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Train and test loss")
        plot_filename = f"loss_{self.config.data}_{self.config.epochs}ep_bu{self.config.update_batch}_r{run+1}_{self.weight_init}_imp{self.config.importance_pruning}_inp{self.config.input_pruning}_z{self.config.zeta}.png"
        # format to be shorter
        plot_filename = plot_filename.replace("False", "F").replace("True", "T")
        # replace "neuron_importance with ni"
        plot_filename = plot_filename.replace("neuron_importance", "ni")
        plt.savefig(plot_filename)

        return metrics

    def get_core_input_connections(self):
        """
        Returns the number of connections for each input neuron

        :return: an array of size equal to the number of input neurons
        """
        values = np.sort(self.w[1].data)
        first_zero_pos = find_first_pos(values, 0)
        last_zero_pos = find_last_pos(values, 0)

        largest_negative = values[int((1 - self.zeta) * first_zero_pos)]
        smallest_positive = values[
            int(min(values.shape[0] - 1, last_zero_pos + self.zeta * (values.shape[0] - last_zero_pos)))]

        wlil = self.w[1].tolil()
        wdok = dok_matrix((self.dimensions[0], self.dimensions[1]), dtype="float32")

        # remove the weights closest to zero
        keep_connections = 0
        for ik, (row, data) in enumerate(zip(wlil.rows, wlil.data)):
            for jk, val in zip(row, data):
                if (val < largest_negative) or (val > smallest_positive):
                    wdok[ik, jk] = val
                    keep_connections += 1

        return wdok.tocsr().getnnz(axis=1)

    def weights_evolution_I(self):
        # this represents the core of the SET procedure. It removes the weights closest to zero in each layer and add new random weights
        for i in range(1, self.n_layers - 1):

            values = np.sort(self.w[i].data)
            first_zero_pos = find_first_pos(values, 0)
            last_zero_pos = find_last_pos(values, 0)

            largest_negative = values[int((1 - self.zeta) * first_zero_pos)]
            smallest_positive = values[
                int(min(values.shape[0] - 1, last_zero_pos + self.zeta * (values.shape[0] - last_zero_pos)))]

            wlil = self.w[i].tolil()
            pdwlil = self.pdw[i].tolil()
            wdok = dok_matrix((self.dimensions[i - 1], self.dimensions[i]), dtype="float32")
            pdwdok = dok_matrix((self.dimensions[i - 1], self.dimensions[i]), dtype="float32")

            # remove the weights closest to zero
            keep_connections = 0
            for ik, (row, data) in enumerate(zip(wlil.rows, wlil.data)):
                for jk, val in zip(row, data):
                    if (val < largest_negative) or (val > smallest_positive):
                        wdok[ik, jk] = val
                        pdwdok[ik, jk] = pdwlil[ik, jk]
                        keep_connections += 1
            limit = np.sqrt(6. / float(self.dimensions[i]))

            # add new random connections
            for kk in range(self.w[i].data.shape[0] - keep_connections):
                ik = np.random.randint(0, self.dimensions[i - 1])
                jk = np.random.randint(0, self.dimensions[i])
                while (wdok[ik, jk] != 0):
                    ik = np.random.randint(0, self.dimensions[i - 1])
                    jk = np.random.randint(0, self.dimensions[i])
                wdok[ik, jk] = np.random.uniform(-limit, limit)
                pdwdok[ik, jk] = 0

            self.pdw[i] = pdwdok.tocsr()
            self.w[i] = wdok.tocsr()

    def weights_evolution_II(self, epoch=0):
        # this represents the core of the SET procedure. It removes the weights closest to zero in each layer and add new random weights
        # improved running time using numpy routines - Amarsagar Reddy Ramapuram Matavalam (amar@iastate.edu)
        # Every 50 epochs, save a plot of the distribution of the sum of weights of the input layer
        if epoch % 50 == 0 and self.config.plotting:
            self._plot_input_distribution(epoch, copy.deepcopy(self.input_sum))
        for i in range(1, self.n_layers - 1):
            # uncomment line below to stop evolution of dense weights more than 80% non-zeros
            # if self.w[i].count_nonzero() / (self.w[i].get_shape()[0]*self.w[i].get_shape()[1]) < 0.8:
            t_ev_1 = datetime.datetime.now()

            # converting to COO form - Added by Amar
            wcoo = self.w[i].tocoo()
            vals_w = wcoo.data
            rows_w = wcoo.row
            cols_w = wcoo.col

            pdcoo = self.pdw[i].tocoo()
            vals_pd = pdcoo.data
            rows_pd = pdcoo.row
            cols_pd = pdcoo.col
            # print("Number of non zeros in W and PD matrix before evolution in layer",i,[np.size(valsW), np.size(valsPD)])

            values = np.sort(self.w[i].data)
            first_zero_pos = find_first_pos(values, 0)
            last_zero_pos = find_last_pos(values, 0)

            largest_negative = values[int((1-self.zeta) * first_zero_pos)]
            smallest_positive = values[int(min(values.shape[0] - 1, last_zero_pos + self.zeta * (values.shape[0] - last_zero_pos)))]

            # remove the weights (W) closest to zero and modify PD as well 
            vals_w_new = vals_w[(vals_w > smallest_positive) | (vals_w < largest_negative)]
            rows_w_new = rows_w[(vals_w > smallest_positive) | (vals_w < largest_negative)]
            cols_w_new = cols_w[(vals_w > smallest_positive) | (vals_w < largest_negative)]

            new_w_row_col_index = np.stack((rows_w_new, cols_w_new), axis=-1)
            old_pd_row_col_index = np.stack((rows_pd, cols_pd), axis=-1)

            new_pd_row_col_index_flag = array_intersect(old_pd_row_col_index, new_w_row_col_index)  # careful about order

            vals_pd_new = vals_pd[new_pd_row_col_index_flag]
            rows_pd_new = rows_pd[new_pd_row_col_index_flag]
            cols_pd_new = cols_pd[new_pd_row_col_index_flag]

            self.pdw[i] = coo_matrix((vals_pd_new, (rows_pd_new, cols_pd_new)), (self.dimensions[i - 1], self.dimensions[i])).tocsr()

            if i == 1 and self.config.plotting:
                print("Now saving the input layer connections")
                self.input_layer_connections.append(coo_matrix((vals_w_new, (rows_w_new, cols_w_new)),
                                                               (self.dimensions[i - 1], self.dimensions[i])).getnnz(axis=1))
                np.savez_compressed(
                    f"{self.save_filename}_input_connections.npz",
                    inputLayerConnections=self.input_layer_connections,
                )

            # add new random connections # TODO: modify to a smart way of adding new connections
            keep_connections = np.size(rows_w_new)
            length_random = vals_w.shape[0] - keep_connections
            if self.weight_init == 'normal':
                random_vals = np.random.randn(length_random) / 10  # to avoid multiple whiles, can we call 3*rand?
            else:
                if self.weight_init == 'he_uniform':
                    limit = np.sqrt(6. / float(self.dimensions[i - 1]))
                if self.weight_init == 'xavier':
                    limit = np.sqrt(6. / (float(self.dimensions[i - 1]) + float(self.dimensions[i])))
                if self.weight_init == 'neuron_importance':
                    # TODO: FIX using norm dist from kichler
                    limit = np.sqrt(6. / float(self.dimensions[i - 1]))
                random_vals = np.random.uniform(-limit, limit, length_random)


            # adding  (wdok[ik,jk]!=0): condition
            # NOTE (Matthijs): I think here we should add the new connections in a non-random way
            while length_random > 0:
                if self.weight_init == 'neuron_importance':
                    neuron_importance_i = self.layer_importances[i]
                    neuron_importance_j = self.layer_importances[i + 1]
                    neuron_importance_i = neuron_importance_i / np.sum(neuron_importance_i)
                    neuron_importance_j = neuron_importance_j / np.sum(neuron_importance_j)

                    ik = np.random.choice(self.dimensions[i - 1], size=length_random, p=neuron_importance_i).astype('int32')
                    jk = np.random.choice(self.dimensions[i], size=length_random, p=neuron_importance_j).astype('int32')
                else:
                    ik = np.random.randint(0, self.dimensions[i - 1], size=length_random, dtype='int32')
                    jk = np.random.randint(0, self.dimensions[i], size=length_random, dtype='int32')

                random_w_row_col_index = np.stack((ik, jk), axis=-1)
                random_w_row_col_index = np.unique(random_w_row_col_index, axis=0)  # removing duplicates in new rows&cols
                oldW_row_col_index = np.stack((rows_w_new, cols_w_new), axis=-1)

                unique_flag = ~array_intersect(random_w_row_col_index, oldW_row_col_index)  # careful about order & tilda

                ik_new = random_w_row_col_index[unique_flag][:,0]
                jk_new = random_w_row_col_index[unique_flag][:,1]
                # be careful - row size and col size needs to be verified
                rows_w_new = np.append(rows_w_new, ik_new)
                cols_w_new = np.append(cols_w_new, jk_new)

                length_random = vals_w.shape[0]-np.size(rows_w_new) # this will constantly reduce length_random

            # adding all the values along with corresponding row and column indices - Added by Amar
            vals_w_new = np.append(vals_w_new, random_vals) # be careful - we can add to an existing link ?
            # vals_pd_new = np.append(vals_pd_new, zero_vals) # be careful - adding explicit zeros - any reason??
            if vals_w_new.shape[0] != rows_w_new.shape[0]:
                print("not good")
            self.w[i] = coo_matrix((vals_w_new, (rows_w_new, cols_w_new)), (self.dimensions[i-1], self.dimensions[i])).tocsr()

            # print("Number of non zeros in W and PD matrix after evolution in layer",i,[(self.w[i].data.shape[0]), (self.pdw[i].data.shape[0])])

            t_ev_2 = datetime.datetime.now()
            # print("Weights evolution time for layer", i, "is", t_ev_2 - t_ev_1)
            self.evolution_time += (t_ev_2 - t_ev_1).seconds

    def _plot_input_distribution(self, epoch, values):
        plot_time_start = time.time()
        plt.hist(values, bins=100)
        plt.title(f"Distribution of the sum of weights of the input layer in epoch {str(epoch)}")
        # save in a folder called "input_weight_distribution"
        plt.savefig(f"input_weight_distribution/epoch_{str(epoch)}.png")
        plt.close()
        print(f"Plotting the input weight distribution took {time.time() - plot_time_start} seconds.")

    def predict(self, x_test, y_test, batch_size=100):
        """
        Function to predict the output of the network for a given input, and compute the classification accuracy.

        :param x_test: (array) Test input
        :param y_test: (array) Correct test output
        :param batch_size: (int) Batch size (default: 100)

        :return: (float) Classification accuracy
        :return: (array) A 2D array of shape (n_cases, n_classes).
        """

        activations = np.zeros((y_test.shape[0], y_test.shape[1]))
        for j in range(x_test.shape[0] // batch_size):
            k = j * batch_size
            l = (j + 1) * batch_size
            _, a_test, _ = self._feed_forward(x_test[k:l], drop=False)
            activations[k:l] = a_test[self.n_layers]

        j_max = x_test.shape[0] // batch_size
        # add the remaining test cases (after the loop above has run j times)
        if x_test.shape[0] % batch_size != 0:
            k = j_max * batch_size
            l = x_test.shape[0]
            _, a_test, _ = self._feed_forward(x_test[k:l], drop=False)
            activations[k:l] = a_test[self.n_layers]
        
        accuracy = compute_accuracy(activations, y_test)
        return accuracy, activations




if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    print("*******************************************************")
    setup_logger(args)
    print_and_log(args)
    print("*******************************************************")
    sum_training_time = 0
    accuracies = []
    runs = args.runs

    # load data
    no_training_samples = 50000  # max 60000 for Fashion MNIST
    no_testing_samples = 10000  # max 10000 for Fashion MNIST

        # create wandb run

    sweep_config = {
        'method': 'bayes',
        'metric': {
            'name': 'accuracy_topk',
            'goal': 'maximize'
        },
        'early_terminate': {
            'type': 'hyperband',
            'min_iter': 5
        },
        'parameters': {
            'dropout_rate': {
                'min': 0.1,
                'max': 0.5
                },
            'zeta': { 
                'min': 0.4,
                'max': 0.9
                },
            'lamda': {
                'min': 0.8,
                'max': 0.99
                },
            'epsilon': {
                'min': 5,
                'max': 25
                }
        }
    }

    sweep_config["parameters"].update({
        'nhidden': {
            'value': 200},
        'weight_decay': {
            'value': args.weight_decay
        },
        'momentum': {
            'value': args.momentum
        },
        'allrelu_slope': {
            'value': args.allrelu_slope
        },
        'data':{
            'value': "madelon"
        },
        'K': {
            'value': 20
        },
        'runs': {
            'value': args.runs
        },
        'eval_epoch': {
            'value': args.eval_epoch
        },
        'input_pruning':{
            'value': True
        },
        'update_batch':{
            'value': True
        },
        'learning_rate': {
            'value': 1e-3
        },
        'epochs':{
            'value': 10
        },
        'batch_size':{
            'value': 32
        },
        'importance_pruning':{
            'value': True
        },
        'plotting':{
            'value': False
        }
    })

    pprint.pprint(sweep_config)

    # done to here

    def run_exp(config=None):
        sum_training_time = 0
        with wandb.init(config=config):
            config=wandb.config
            np.random.seed(42)
            x_train, y_train, x_test, y_test = get_data(config.data)
            network = SET_MLP((x_train.shape[1], config.nhidden, y_train.shape[1]),
                              (AlternatedLeftReLU(-config.allrelu_slope), Softmax), 
                              input_pruning=config.input_pruning,
                              importance_pruning=config.importance_pruning,
                              epsilon=config.epsilon,
                              config=config,
                              lamda=config.lamda) # One-layer version   
            print(f"Data shapes are: {x_train.shape}, {y_train.shape}, {x_test.shape}, {y_test.shape}")
            metrics = np.zeros((config.runs, config.epochs, 4))
            start_time = time.time()

            network.fit(
                x_train,
                y_train,
                x_test,
                y_test,
                loss=CrossEntropy,
                epochs=config.epochs,
                batch_size=config.batch_size,
                learning_rate=config.learning_rate,
                momentum=config.momentum,
                weight_decay=config.weight_decay,
                zeta=config.zeta,
                dropoutrate=config.dropout_rate,
                testing=True,
                save_filename=f"results/set_mlp_sequential_{x_train.shape[0]}_training_samples_e{config.epsilon}_rand1",
                monitor=True,
                run=1,
                metrics=metrics,
                eval_epoch=config.eval_epoch,
                config=config,
            )        

            selected_features, importances = select_input_neurons(copy.deepcopy(network.w[1]), config.K)
            accuracy_topk = evaluate_fs(x_train, x_test, y_train, y_test, selected_features)
            wandb.summary['accuracy_topk'] = accuracy_topk
            wandb.log({'accuracy_topk': accuracy_topk})

            step_time = time.time() - start_time
            print("\nTotal execution time: ", step_time)
            print("\nTotal training time: ", network.training_time)
            print("\nTotal training time: ", network.training_time)
            print("\nTotal testing time: ", network.testing_time)
            print("\nTotal evolution time: ", network.evolution_time)
            sum_training_time += step_time 


    sweep_id = wandb.sweep(sweep_config, project="madelon-sweep-10ep")
    wandb.agent(sweep_id, function=run_exp, count=100)

    wandb.finish()