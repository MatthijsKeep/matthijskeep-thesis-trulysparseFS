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

from utils.load_data import load_fashion_mnist_data, load_cifar10_data, load_madelon_data, load_mnist_data
from wasap_sgd.train.monitor import Monitor

import copy
import datetime
import json
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import shutil
import sys
import time


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

    # get the sum of the absolute values of the weights for each neuron
    # print(weights.shape) # (784, )
    # print(weights)
    # print(weights)
    sum_weights = np.abs(weights).sum(axis=1)
    # sort the neurons by the sum of the absolute values of the weights
    # print(sum_weights)
    # print(f"The shape of the sum of the weights is: {sum_weights.shape}") # (784, 1)
    # get the indices of the k most important neurons
    # print(k)
    # print the index of the highest weight
    print(f"The input neuron with the highest weight is: {np.argmax(sum_weights)}")
    important_neurons_idx = np.argsort(sum_weights, axis=0)[::-1][:k]
    # print(important_neurons_idx)
    # print(f"The shape of the indices of the k most important neurons is: {important_neurons_idx.shape}")
    # print(f"The {len(important_neurons_idx)} most important neurons are: {important_neurons_idx}")

    # important_neurons = lil_matrix((weights.shape[0], weights.shape[1]))
    # # important_neurons[important_neurons_idx, :] = weights[important_neurons_idx, :]
    # # print(f"The weight matrix looks like: {weights}")

    # # for each neuron, remove it from the sparse matrix if the neuron is not in the list of important neurons
    # for i in range(weights.shape[0]):
    #     if i not in important_neurons_idx:
    #         weights[i, :] = 0
    
    
    # important_neurons = weights.tocsr()
    # # print(f"The important neurons matrix looks like: {important_neurons}")
    
    # # get the sum of the absolute values of the weights for each neuron, and include the indices
    # sum_weights = np.abs(important_neurons).sum(axis=1)
    # # sort the sum of the weights in descending order, and take the first k elements
    # sum_weights = (np.sort(sum_weights, axis=0)[-k:])[::-1]
    # # print(f"The sum of weights for each neuron is: {sum_weights.flatten()}")

    # zip the indices and the sum of the weights
    important_neurons = np.abs(set_mlp.w[1].copy()).sum(axis=1)
    # print(f"The important neurons are: {important_neurons}")

    return important_neurons_idx, important_neurons


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


class SET_MLP:
    def __init__(self, dimensions, activations, epsilon=20, weight_init='neuron_importance'):
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
        self.importance_pruning = True
        self.input_pruning = True
        self.lamda = lamda

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
                self.layer_importances[i] = np.zeros(self.w[i].shape[0], dtype='float32')
                # print(f"The shape of the input layer is: {self.w[i].shape[0]}, which is layer {i}")
            if i == self.n_layers: # Output layer
                # print(f"{i} == {self.n_layers}")
                # print(f"The shape of the weight matrix we are looking at is: {self.w[i-1].shape}")
                self.layer_importances[i] = np.zeros(self.w[i-1].shape[1], dtype='float32')
                # print(f"The shape of the output layer is: {self.w[i-1].shape[1]}, which is layer {i}")
            # for the hidden layers all neurons have the same importance
            if i not in [1, self.n_layers]: # All other layers (hidden layers)
                # print(f"{i} not in [1, {self.n_layers}]")
                # print(f"The shape of the weight matrix we are looking at is: {self.w[i].shape}")
                self.layer_importances[i] = np.ones(self.w[i].shape[0], dtype='float32')
                # print(f"The shape of the hidden layer is: {self.w[i].shape[0]}, which is layer {i}")

        # print(f"The shape of the layer importances is: {self.layer_importances}")          

    def _update_layer_importances(self):

        # Update the importances of the input layer
        # TODO: FIX
        start_time = time.time()
        lamda = self.lamda
        temp = np.array(self.input_sum.copy()).reshape(-1)
        # print(f"The shapes I add together are {self.layer_importances[1].shape} and {temp.shape}")
        # print(f"The left side of the equation has mean {(self.layer_importances[1] * bal).mean()}")
        # print(np.squeeze(self.input_sum))
        # print(f"The right side of the equation has mean {(temp * (1 - bal)).mean()}")
        # TODO: Determine balancing parameter
        
        self.layer_importances[1] = self.layer_importances[1] * lamda + temp * (1 - lamda) # NOTE (Matthijs): Only update input layer for now
        # print(f"The shape of the input layer importances is: {self.layer_importances[1].shape}")
        # print(f"The lowest neuron importance is: {self.layer_importances[1].min()}, which is neuron {np.argmin(self.layer_importances[1])}")
        # print(f"The highest neuron importance is: {self.layer_importances[1].max()}, which is neuron {np.argmax(self.layer_importances[1])}")
        # print(f"The mean neuron importance is: {self.layer_importances[1].mean()}")
        # Update the importance of the output layer?
        # NOTE (Matthijs): Not sure if I want to do anything with the importances of the output layer  
        # Print the runtime and round to 3 decimals 
        # print(f"Updating the layer importances took {round(time.time() - start_time, 5)} seconds")

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

            if zerow_before > self.input_sum.shape[0] - (args.K * 2):
                print(f"WARNING: No more neurons to prune, since {zerow_before} > {self.input_sum.shape[0] - (args.K * 2)}")
                self.input_pruning = False
                time.sleep(10)
            zerow_pct = zerow_before / self.input_sum.shape[0] * 100
            print(f"Before pruning in epoch {epoch} The amount of neurons without any incoming weights is: {zerow_before}, \
                    which is {zerow_pct}% of the total neurons.")

            start_input_pruning = datetime.datetime.now()
            # print(f"The shape of the layer is {self.w[i].shape}") # (input, nhidden)
            sum_incoming_weights = np.array(self.input_sum.copy())
            
            
            # print(f"The lowest value in the neuron weights is: {lowest_in_sum}")
            # Calculating the sum of the incoming weights for each node in the hidden layer.
            # print(sum_incoming_weights.shape) # (1, input))
            # print(f"The shape of the sum of the weights is {sum_incoming_weights.shape}") # (1, input))
            # get 20th percentile from (1, input)
            curr_percentile = 20 + ((epoch/no_training_epochs) * 60)
            t = np.percentile(sum_incoming_weights, 20)
            if t == 0:
                # Find a value for the percentile such that you slowly prune the neurons over the epochs until you reach 200 neurons
                # It should be a function of epoch/n_epochs
                curr_percentile = 20 + ((epoch/no_training_epochs) * 70)
                t_2 = np.percentile(sum_incoming_weights, curr_percentile)
                # prune the weights that are lower than 
                print(f"\n NOTE: t is 0, setting it to t_2 : {t_2}, which prunes the {curr_percentile}th percentile of the weights")
                t = t_2
            # print(t)
            # breakpoint()
            sum_incoming_weights = np.where(sum_incoming_weights <= t, 0, sum_incoming_weights)
            # print(sum_incoming_weights)
            ids = np.argwhere(sum_incoming_weights == 0)
            print("ids", ids.shape)
            # overlay the ids on a 28*28 matrix and set the values to 0
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

            # print(matrix2828.shape)
            # print(matrix2828)
            # print(ids)

            weights = self.w[i].tolil()
            pdw = self.pdw[i].tolil()

            weights[ids[:,0], :] = 0
            pdw[ids[:,0], :] = 0

            self.w[i] = weights.tocsr()
            self.pdw[i] = pdw.tocsr()

            print(f"Input pruning took {datetime.datetime.now() - start_input_pruning} seconds")

    def fit(self, x, y_true, x_test, y_test, loss, epochs, batch_size, learning_rate=1e-3, momentum=0.9,
            weight_decay=0.0002, zeta=0.3, dropoutrate=0., testing=True, save_filename="", monitor=False):
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
        self.input_layer_connections.append(self.get_core_input_connections())
        np.savez_compressed(self.save_filename + "_input_connections.npz",
                            inputLayerConnections=self.input_layer_connections)
        
        # 1-d sum of the absolute values of the input weights


        maximum_accuracy = 0
        metrics = np.zeros((epochs, 4))

        for i in range(epochs):
            # Shuffle the data
            self.input_weights = self.w[1].copy()
            _, self.input_sum = select_input_neurons(self.input_weights, args.K)
            # print(f"The shape of the input layer weights is: {self.input_sum.shape}")
            if i == 0 or i == 1 or i == 5 or i == 10 or i == 25 or i % 50 == 0 or i == epochs:
                start_time = time.time()
                print("In eval loop")
                # print(f"The shape of the input layer weights is: {set_mlp.w[1].shape}")
                # print(self.get_core_input_connections().shape)
                # print(set_mlp.w[1].copy().shape)
                # print(self.input_sum.shape) 
                # selected_features, importances = select_input_neurons(set_mlp.w[1].copy(), args.K)
                # print(importances)
                # print("Now saving the importances!")
                # print("==================================")
                importances_for_eval = self.input_sum.copy()
                # print(importances_for_eval.shape)
                importances_for_eval = pd.DataFrame(importances_for_eval).to_csv(header=False, index=False)

                importances_path = f"importances/importances_{str(args.data)}_{i}_{str(args.model)}.csv"
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
                # TODO: add topology update here 
                k = j * batch_size
                l = (j + 1) * batch_size
                z, a, masks = self._feed_forward(x_[k:l], True)

                self._back_prop(z, a, masks,  y_[k:l])

                t5 = datetime.datetime.now()
                self._update_layer_importances()
                if i < epochs - 1:  # do not change connectivity pattern after the last epoch
                    # self.weights_evolution_I() # this implementation is more didactic, but slow.
                    self.weights_evolution_II(i)  # this implementation has the same behaviour as the one above, but it is much faster.
                t6 = datetime.datetime.now()
                # print("Weights evolution time ", t6 - t5)

            t2 = datetime.datetime.now()

            if self.monitor:
                self.monitor.stop_monitor()

            print("\nSET-MLP Epoch ", i)
            print("Training time: ", t2 - t1)
            self.training_time += (t2 - t1).seconds

            # test model performance on the test data at each epoch
            # this part is useful to understand model performance and can be commented for production settings
            if testing and i % 10 == 0:
                t3 = datetime.datetime.now()
                accuracy_test, activations_test = self.predict(x_test, y_test)
                accuracy_train, activations_train = self.predict(x, y_true)

                t4 = datetime.datetime.now()
                maximum_accuracy = max(maximum_accuracy, accuracy_test)
                loss_test = self.loss.loss(y_test, activations_test)
                loss_train = self.loss.loss(y_true, activations_train)
                metrics[i, 0] = loss_train
                metrics[i, 1] = loss_test
                metrics[i, 2] = accuracy_train
                metrics[i, 3] = accuracy_test

                print(f"Testing time: {t4 - t3}; \n"
                      f"Loss test: {loss_test}; \n"
                      f"Accuracy test: {accuracy_test}; \n"
                      f"Maximum accuracy val: {maximum_accuracy} \n")
                self.testing_time += (t4 - t3).seconds




            # save performance metrics values in a file
            if self.save_filename != "":
                np.savetxt(self.save_filename +".txt", metrics)

            if self.save_filename != "" and self.monitor:
                with open(self.save_filename + "_monitor.json", 'w') as file:
                    file.write(json.dumps(self.monitor.get_stats(), indent=4, sort_keys=True, default=str))
        # print(self.get_core_input_connections())
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
        if epoch % 50 == 0:
            self._plot_input_distribution(epoch, self.input_sum.copy())
        for i in range(1, self.n_layers - 1):
            # uncomment line below to stop evolution of dense weights more than 80% non-zeros
            # if self.w[i].count_nonzero() / (self.w[i].get_shape()[0]*self.w[i].get_shape()[1]) < 0.8:
            t_ev_1 = datetime.datetime.now()
            # print(i) # layer number
            # print(self.w[i].copy().get_shape()) # shape of the layer

            # Importance Pruning (on the hidden layer(s)), currenly OFF in my implementation
            if self.importance_pruning and epoch % 10 == 0 and epoch > 100:
                # TODO: ship this to a separate function, and call it from here
                print("Importance Pruning")
                # print(self.w[i].shape) # (input, nhidden)
                sum_incoming_weights = np.abs(self.w[i]).sum(axis=0)
                # print(sum_incoming_weights.shape) # (1, nhidden))

                t = np.percentile(sum_incoming_weights, 20, axis=1)
                sum_incoming_weights = np.where(sum_incoming_weights <= t, 0, sum_incoming_weights)
                # print(sum_incoming_weights)
                ids = np.argwhere(sum_incoming_weights == 0)
                print("ids", ids.shape)

                weights = self.w[i].tolil()
                pdw = self.pdw[i].tolil()

                weights[:, ids[:,1]] = 0
                pdw[:, ids[:,1]] = 0

                self.w[i] = weights.tocsr()
                self.pdw[i] = pdw.tocsr()

            # Input neuron pruning (on the input layer)
            if self.input_pruning and epoch % 25 == 0 and epoch > 100 and i == 1:
                self._input_pruning(epoch=epoch, i=i)

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

            if i == 1:
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
                    # TODO: FIX
                    limit = np.sqrt(6. / float(self.dimensions[i - 1]))
                random_vals = np.random.uniform(-limit, limit, length_random)


            # adding  (wdok[ik,jk]!=0): condition
            # NOTE (Matthijs): I think here we should add the new connections in a non-random way
            while length_random > 0:
                if self.weight_init == 'neuron_importance':
                    # We need to add the connections differently for the three layer types (input, hidden, output)
                    # Check the neuron importance, bias new connections to the most important neurons
                    neuron_importance_i = self.layer_importances[i]
                    neuron_importance_j = self.layer_importances[i + 1]
                    # make neuron importance sum to 1
                    neuron_importance_i = neuron_importance_i / np.sum(neuron_importance_i)
                    neuron_importance_j = neuron_importance_j / np.sum(neuron_importance_j)

                    #TODO : Also add a completely random element to keep exploring?

                    ik = np.random.choice(self.dimensions[i - 1], size=length_random, p=neuron_importance_i).astype('int32')
                    # also bias the neurons it connects to towards the most important neurons
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
        accuracy = compute_accuracy(activations, y_test)
        return accuracy, activations


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    print("*******************************************************")
    setup_logger(args)
    print_and_log(args)
    print("*******************************************************")
    # print_and_log(torch.cuda.is_available())
    # print_and_log(torch.cuda.device_count())
    # use_cuda = not args.no_cuda and torch.cuda.is_available()
    # device = torch.device("cuda" if use_cuda else "cpu")
    # print("device", device)
    # print("*******************************************************")
    sum_training_time = 0
    runs = args.runs

    # load data
    no_training_samples = 50000  # max 60000 for Fashion MNIST
    no_testing_samples = 10000  # max 10000 for Fashion MNIST
    # set model parameters
    no_hidden_neurons_layer = args.nhidden
    # sparsity_level = args.sparsity
    # erdos renyi formula for sparsity level
    # 

    epsilon = 20  # set the sparsity level
    zeta = 0.3 # in [0..1]. It gives the percentage of unimportant connections which are removed and replaced with random ones after every epoch
    no_training_epochs = args.epochs
    batch_size = 128
    dropout_rate = 0.3
    learning_rate = args.lr
    lamda = args.lamda
    momentum = args.momentum
    weight_decay = 0.0002
    allrelu_slope = args.allrelu_slope
    # k = args.K

    # make a list of the layers

    for i in range(runs):
        print(args.data)
        if args.data == 'FashionMnist':
            x_train, y_train, x_test, y_test = load_fashion_mnist_data(no_training_samples, no_testing_samples)
        elif args.data == 'mnist':
            x_train, y_train, x_test, y_test = load_mnist_data(no_training_samples, no_testing_samples)
        elif args.data == 'madelon':
            x_train, y_train, x_test, y_test = load_madelon_data()
        np.random.seed(i)
        

        # create SET-MLP (MLP with adaptive sparse connectivity trained with Sparse Evolutionary Training)

        set_mlp = SET_MLP((x_train.shape[1], no_hidden_neurons_layer, y_train.shape[1]),
                          (AlternatedLeftReLU(-allrelu_slope), Softmax), epsilon=epsilon) # One-layer version
        # set_mlp = SET_MLP((x_train.shape[1], no_hidden_neurons_layer, no_hidden_neurons_layer, no_hidden_neurons_layer, y_train.shape[1]),
        #                   (AlternatedLeftReLU(-allrelu_slope), AlternatedLeftReLU(allrelu_slope), AlternatedLeftReLU(-allrelu_slope), Softmax), 
        #                    epsilon=epsilon) # Three-layer version                


        start_time = time.time()
        # train SET-MLP
        set_mlp.fit(
            x_train,
            y_train,
            x_test,
            y_test,
            loss=CrossEntropy,
            epochs=no_training_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
            zeta=zeta,
            dropoutrate=dropout_rate,
            testing=True,
            save_filename=f"results/set_mlp_sequential_{no_training_samples}_training_samples_e{epsilon}_rand{str(i)}",
            monitor=True,
        )

        step_time = time.time() - start_time
        print("\nTotal execution time: ", step_time)
        print("\nTotal training time: ", set_mlp.training_time)
        print("\nTotal testing time: ", set_mlp.testing_time)
        print("\nTotal evolution time: ", set_mlp.evolution_time)
        sum_training_time += step_time

        # test SET-MLP
        # select the 50 most important connections in the input layer
        # time the choosing of the weigths
        start_time = time.time()
        accuracy, _ = set_mlp.predict(x_test, y_test, batch_size=100)
        # print(f"The shape of the input layer weights is: {set_mlp.w[1].shape}")
        selected_features, importances = select_input_neurons(set_mlp.w[1].copy(), args.K)
        # print(set_mlp.w[1])
        print(f"The choosing of the {args.K} most important weights took {time.time() - start_time} seconds")
        
        # Print how many neurons in the input layer have a connection
        # print(f"The number of neurons in the input layer with a connection is: {np.count_nonzero(set_mlp.w[1].sum(axis=1))}")
        # Print which neurons have a connection

        # print(f" The selected features are {selected_features}")
        selected_features_for_eval = pd.DataFrame(selected_features)
        # convert to csv
        selected_features_for_eval = selected_features_for_eval.to_csv(header=False, index=False)

        selected_features_path = f"features/selected_features_{str(args.data)}_{no_training_epochs}_{str(args.model)}.csv"
        if not os.path.exists(os.path.dirname(selected_features_path)):
            os.makedirs(os.path.dirname(selected_features_path))

        with open(selected_features_path, 'w') as f:
            f.write(selected_features_for_eval)

        importances_for_eval = pd.DataFrame(importances)
        # convert to csv
        importances_for_eval = importances_for_eval.to_csv(header=False, index=False)

        importances_path = f"importances/importances_{str(args.data)}_{no_training_epochs}_{str(args.model)}.csv"
        if not os.path.exists(os.path.dirname(importances_path)):
            os.makedirs(os.path.dirname(importances_path))

        with open(importances_path, 'w') as f:
            f.write(importances_for_eval)

        # change x_train and x_test to only have the selected features
        # print(selected_features)
        x_train_new = np.squeeze(x_train[:, selected_features])
        x_test_new = np.squeeze(x_test[:, selected_features])
        # change y_train and y_test from one-hot to single label
        y_train = np.argmax(y_train, axis=1)
        y_test = np.argmax(y_test, axis=1)

        # print all shapes
        print(f"The shape of x_train is: {x_train.shape}")
        print(f"The shape of x_train_new is: {x_train_new.shape}")
        print(f"The shape of x_test is: {x_test.shape}")
        print(f"The shape of x_test_new is: {x_test_new.shape}")
        # print(f"The shape of y_train is: {y_train.shape}")
        # print(f"The shape of y_test is: {y_test.shape}")

        # time the tesitng
        start_time = time.time()
        accuracy_topk = svm_test(x_train_new, y_train, x_test_new, y_test)
        print("\n Accuracy of the last epoch on the testing data (with all features): ", accuracy)
        print(f"The testing of the {args.K} most important weights took {time.time() - start_time} seconds")
        print(f"Accuracy of the last epoch on the testing data (with {args.K} features): ", accuracy_topk)
        print_and_log(accuracy_topk)

        # plot the features
        if args.plot_features:
            plot_features(data=args.data)

        if args.plot_importances:
            plot_importances(importances, args.K)
    print(f"Average training time over {runs} runs is {sum_training_time/runs} seconds")
