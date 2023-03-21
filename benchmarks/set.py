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
from scipy.sparse import lil_matrix
from scipy.sparse import coo_matrix
from scipy.sparse import dok_matrix
from utils.nn_functions import *

from test import load_in_data, svm_test
from utils.load_data import load_fashion_mnist_data, load_madelon_data, load_mnist_data, load_usps, load_coil, load_isolet, load_har, load_pcmac, load_smk, load_gla


import copy
import datetime
import logging
import os
import time
import json
import sys
import numpy as np
from numba import njit, prange

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


def create_sparse_weights(epsilon, n_rows, n_cols):
    # He uniform initialization
    limit = np.sqrt(6. / float(n_rows))

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
    dtype = {'names': ['f{}'.format(i) for i in range(n_cols)], 'formats': n_cols * [a.dtype]}
    return np.in1d(a.view(dtype), b.view(dtype))  # boolean return

def setup_logger(args): 
    global logger
    if logger == None:
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


if not os.path.exists('./models'): os.mkdir('./models')
if not os.path.exists('./logs'): os.mkdir('./logs')
logger = None

class SET_MLP:
    def __init__(self, dimensions, activations, epsilon=20):
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

        self.save_filename = ""
        self.input_layer_connections = []
        self.monitor = None

        # Weights and biases are initiated by index. For a one hidden layer net you will have a w[1] and w[2]
        self.w = {}
        self.b = {}
        self.pdw = {}
        self.pdd = {}

        # Activations are also initiated by index. For the example we will have activations[2] and activations[3]
        self.activations = {}
        for i in range(len(dimensions) - 1):
            self.w[i + 1] = create_sparse_weights(self.epsilon, dimensions[i], dimensions[i + 1])  # create sparse weight matrices
            self.b[i + 1] = np.zeros(dimensions[i + 1], dtype='float32')
            self.activations[i + 2] = activations[i]

    def _feed_forward(self, x, drop=False):
        """
        Execute a forward feed through the network.
        :param x: (array) Batch of input data vectors.
        :return: (tpl) Node outputs and activations per layer. The numbering of the output is equivalent to the layer numbers.
        """
        # w(x) + b
        z = {}

        # activations: f(z)
        a = {1: x}  # First layer has no activations as input. The input x is the input.
        masks = {}

        for i in range(1, self.n_layers):
            z[i + 1] = a[i] @ self.w[i] + self.b[i]
            a[i + 1] = self.activations[i + 1].activation(z[i + 1])
            if drop:
                if i < self.n_layers - 1:
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
        # print("right before the _update_w_b function")
        for k, v in update_params.items():
            self._update_w_b(k, v[0], v[1])

    def _update_w_b(self, index, dw, delta):
        """
        Update weights and biases.

        :param index: (int) Number of the layer
        :param dw: (array) Partial derivatives
        :param delta: (array) Delta error.
        """
        # print("now in the _update_w_b function")
        # print(self.pdw)

        # perform the update with momentum
        if index not in self.pdw:
            self.pdw[index] = - self.learning_rate * dw
            self.pdd[index] = - self.learning_rate * delta
        else:
            self.pdw[index] = self.momentum * self.pdw[index] - self.learning_rate * dw
            self.pdd[index] = self.momentum * self.pdd[index] - self.learning_rate * delta

        self.w[index] += self.pdw[index] - self.weight_decay * self.w[index]
        self.b[index] += self.pdd[index] - self.weight_decay * self.b[index]

    def fit(self, x, y_true, x_test, y_test, loss, epochs, batch_size, learning_rate=1e-3, momentum=0.9,
            weight_decay=0.0002, zeta=0.3, dropoutrate=0., testing=True, save_filename="", monitor=False):
        """
        :param x: (array) Containing parameters
        :param y_true: (array) Containing one hot encoded labels.
        :return (array) A 2D array of metrics (epochs, 3).
        """
        if x.shape[0] != y_true.shape[0]:
            raise ValueError("Length of x and y arrays don't match")

        # self.monitor = Monitor(save_filename=save_filename) if monitor else None

        # Initiate the loss object with the final activation function
        self.loss = loss()
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.zeta = zeta
        self.dropout_rate = dropoutrate
        self.save_filename = save_filename
        self.input_layer_connections.append(self.get_core_input_connections())
        # np.savez_compressed(self.save_filename + "_input_connections.npz",
        #                     inputLayerConnections=self.input_layer_connections)

        maximum_accuracy = 0
        metrics = np.zeros((epochs, 4))

        for i in range(epochs):
            # Shuffle the data
            seed = np.arange(x.shape[0])
            np.random.shuffle(seed)
            x_ = x[seed]
            y_ = y_true[seed]

            if self.monitor:
                self.monitor.start_monitor()

            # training
            t1 = datetime.datetime.now()
            # if the shape is smaller than the batch size then the batch size is set to the shape of the data
            if x.shape[0] < batch_size:
                batch_size = x.shape[0]

            for j in range(x.shape[0] // batch_size):
                k = j * batch_size
                l = (j + 1) * batch_size
                # print("right before the _feed_forward function")
                z, a, masks = self._feed_forward(x_[k:l], True)
                # print("right before the _back_prop function")
                self._back_prop(z, a, masks,  y_[k:l])




            t2 = datetime.datetime.now()

            if self.monitor:
                self.monitor.stop_monitor()

            print("\nSET-MLP Epoch ", i)
            print("Training time: ", t2 - t1)

            # print(self.pdw)

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
                metrics[i, 0] = loss_train
                metrics[i, 1] = loss_test
                metrics[i, 2] = accuracy_train
                metrics[i, 3] = accuracy_test

                print(f"Testing time: {t4 - t3}\n; Loss test: {loss_test}; \n"
                                 f"Accuracy test: {accuracy_test}; \n"
                                 f"Maximum accuracy val: {maximum_accuracy}")

            t5 = datetime.datetime.now()
            if i < epochs - 1:# do not change connectivity pattern after the last epoch
                # self.weights_evolution_I() # this implementation is more didactic, but slow.
                self.weights_evolution_II()  # this implementation has the same behaviour as the one above, but it is much faster.
            t6 = datetime.datetime.now()
            print("Weights evolution time ", t6 - t5)

            # save performance metrics values in a file
            # if self.save_filename != "":
            #     np.savetxt(self.save_filename +".txt", metrics)

            # if self.save_filename != "" and self.monitor:
            #     with open(self.save_filename + "_monitor.json", 'w') as file:
            #         file.write(json.dumps(self.monitor.get_stats(), indent=4, sort_keys=True, default=str))

        return metrics

    def get_core_input_connections(self):
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

    def weights_evolution_II(self):
        # this represents the core of the SET procedure. It removes the weights closest to zero in each layer and add new random weights
        # improved running time using numpy routines - Amarsagar Reddy Ramapuram Matavalam (amar@iastate.edu)
        for i in range(1, self.n_layers - 1):
            # uncomment line below to stop evolution of dense weights more than 80% non-zeros
            # if self.w[i].count_nonzero() / (self.w[i].get_shape()[0]*self.w[i].get_shape()[1]) < 0.8:
                t_ev_1 = datetime.datetime.now()
                # converting to COO form - Added by Amar
                wcoo = self.w[i].tocoo()
                vals_w = wcoo.data
                rows_w = wcoo.row
                cols_w = wcoo.col

                print(i)
                print(self.pdw)
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

                #remove the weights (W) closest to zero and modify PD as well
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

                # if i == 1:
                #     self.input_layer_connections.append(coo_matrix((vals_w_new, (rows_w_new, cols_w_new)),
                #                                                    (self.dimensions[i - 1], self.dimensions[i])).getnnz(axis=1))
                #     np.savez_compressed(self.save_filename + "_input_connections.npz",
                #                         inputLayerConnections=self.input_layer_connections)

                # add new random connections
                keep_connections = np.size(rows_w_new)
                length_random = vals_w.shape[0]-keep_connections
                limit = np.sqrt(6. / float(self.dimensions[i - 1]))
                random_vals = np.random.uniform(-limit, limit, length_random)
                zero_vals = 0*random_vals  # explicit zeros

                # adding  (wdok[ik,jk]!=0): condition
                while length_random>0:
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

                    length_random = vals_w.shape[0]-np.size(rows_w_new) # this will constantly reduce lengthRandom

                # adding all the values along with corresponding row and column indices - Added by Amar
                vals_w_new = np.append(vals_w_new, random_vals) # be careful - we can add to an existing link ?
                # vals_pd_new = np.append(vals_pd_new, zero_vals) # be careful - adding explicit zeros - any reason??
                if vals_w_new.shape[0] != rows_w_new.shape[0]:
                    print("not good")
                self.w[i] = coo_matrix((vals_w_new, (rows_w_new, cols_w_new)), (self.dimensions[i-1], self.dimensions[i])).tocsr()

                # print("Number of non zeros in W and PD matrix after evolution in layer",i,[(self.w[i].data.shape[0]), (self.pdw[i].data.shape[0])])

                t_ev_2 = datetime.datetime.now()
                print("Weights evolution time for layer", i,"is", t_ev_2 - t_ev_1)

    def predict(self, x_test, y_test, batch_size=100):
        """
        :param x_test: (array) Test input
        :param y_test: (array) Correct test output
        :param batch_size:
        :return: (flt) Classification accuracy
        :return: (array) A 2D array of shape (n_cases, n_classes).
        """
        activations = np.zeros((y_test.shape[0], y_test.shape[1]))

        # also work with batch sizes that do not exactly divide the number of test cases
        
        # calculate number of batches (without the overflow) as variable j
        j_max = x_test.shape[0] // batch_size

        for j in range(x_test.shape[0] // batch_size):
            k = j * batch_size
            l = (j + 1) * batch_size
            _, a_test, _ = self._feed_forward(x_test[k:l], drop=False)
            activations[k:l] = a_test[self.n_layers]
        
        # add the remaining test cases (after the loop above has run j times)
        if x_test.shape[0] % batch_size != 0:
            k = j_max * batch_size
            l = x_test.shape[0]
            _, a_test, _ = self._feed_forward(x_test[k:l], drop=False)
            activations[k:l] = a_test[self.n_layers]

        accuracy = compute_accuracy(activations, y_test)
        return accuracy, activations


def load_fashion_mnist_data(no_training_samples, no_testing_samples):
    np.random.seed(0)

    data = np.load("data/FashionMNIST/fashion_mnist.npz")

    index_train = np.arange(data["X_train"].shape[0])
    np.random.shuffle(index_train)

    index_test = np.arange(data["X_test"].shape[0])
    np.random.shuffle(index_test)

    x_train = data["X_train"][index_train[0:no_training_samples], :]
    y_train = data["Y_train"][index_train[0:no_training_samples], :]
    x_test = data["X_test"][index_test[0:no_testing_samples], :]
    y_test = data["Y_test"][index_test[0:no_testing_samples], :]

    # normalize in 0..1
    x_train = x_train.astype('float64') / 255.
    x_test = x_test.astype('float64') / 255.

    return x_train, y_train, x_test, y_test


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    print("*******************************************************")
    setup_logger(args)
    print_and_log(args)

    sum_training_time = 0
    runs = 10
    svm_accs = np.zeros(runs)
    data = args.data
    K = 20 if data == 'madelon' else 50

    # load data
    no_training_samples = 50000  # max 60000 for Fashion MNIST
    no_testing_samples = 10000  # max 10000 for Fshion MNIST
    # set model parameters
    no_hidden_neurons_layer = 200
    epsilon = 20  # set the sparsity level
    zeta = 0.3  # in [0..1]. It gives the percentage of unimportant connections which are removed and replaced with random ones after every epoch
    no_training_epochs = 250
    batch_size = 256
    dropout_rate = 0.3
    learning_rate = 0.001
    momentum = 0.9
    weight_decay = 0.0002
    allrelu_slope = 0.6

    for i in range(runs):
        # TODO - standardize all data loading functions to 1 function

        if data == 'fashion_mnist':
            x_train, y_train, x_test, y_test = load_fashion_mnist_data(no_training_samples, no_testing_samples)
        elif data == 'mnist':
            x_train, y_train, x_test, y_test = load_mnist_data(no_training_samples, no_testing_samples)
        elif data == 'USPS':
            x_train, y_train, x_test, y_test = load_usps()
        elif data == 'coil':
            x_train, y_train, x_test, y_test = load_coil()
        elif data == 'isolet':
            x_train, y_train, x_test, y_test = load_isolet()
        elif data == 'har':
            x_train, y_train, x_test, y_test = load_har()
        elif data == 'pcmac':
            x_train, y_train, x_test, y_test = load_pcmac()
        elif data == 'smk':
            x_train, y_train, x_test, y_test = load_smk()
        elif data == 'gla':
            x_train, y_train, x_test, y_test = load_gla()
        elif data == 'madelon':
            x_train, y_train, x_test, y_test = load_madelon_data()

        np.random.seed(i)

        # create SET-MLP (MLP with adaptive sparse connectivity trained with Sparse Evolutionary Training)
        print(x_train.shape[1])
        set_mlp = SET_MLP((x_train.shape[1], no_hidden_neurons_layer, y_train.shape[1]),
                          (AlternatedLeftReLU(-allrelu_slope), Softmax), epsilon=epsilon)

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
            save_filename=f"Pretrained_results/set_mlp_{no_training_samples}_training_samples_e{epsilon}_rand{str(i)}",
            monitor=True,
        )

        step_time = time.time() - start_time
        print("\nTotal training time: ", step_time)
        sum_training_time += step_time

        sum_weights = np.abs(set_mlp.w[1]).sum(axis=1)
        selected_features = important_neurons_idx = np.argsort(sum_weights, axis=0)[::-1][:K]
        # test SET-MLP by selecting the top K features from the input layer
        train_X_new = np.squeeze(x_train[:, selected_features])
        test_X_new = np.squeeze(x_test[:, selected_features])
        # change y_train and y_test from one-hot to single label
        y_train = np.argmax(y_train, axis=1)
        y_test = np.argmax(y_test, axis=1)

        # TODO - Test if the shapes going into the SVM are the ones you expect so (n_samples, 20) and (n_samples,)
        print(f"Shape going into the SVM: {train_X_new.shape} and {y_train.shape} and {test_X_new.shape} and {y_test.shape}")

        svm_acc = svm_test(train_X_new, y_train, test_X_new, y_test)
        svm_accs[i] = svm_acc

        print("\nAccuracy of the last epoch on the testing data: ", svm_acc)
    print(f"Average training time over {runs} runs is {sum_training_time/runs} seconds")
    # Export the svm_accs to a csv file with run, accuracy, filename = {data}_{runs}.csv to the benchmark/results folder
    np.savetxt(f"benchmarks/results/set/{data}_{runs}.csv", svm_accs, delimiter=",")
