import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import torch

from argparser import get_parser

parser = get_parser()
args = parser.parse_args()

data = args.data

def plot_importances(data='MNIST', k=50):
    """
    This function plots the importances of the features, comparing my model to WAST

    Args:
        data (str, optional): The dataset to use. Defaults to 'MNIST'.
        k (int, optional): The number of features to plot. Defaults to 50.

    Returns:
        None
    """


    if data == 'MNIST':
        # retrieve 9 examples from data/MNIST/training.pt and compress into one image of size 28*28
        mnist_example = torch.load('data/MNIST/processed/training.pt')
        mnist_example = mnist_example[0][:25]


        # retrieve the importances from literature/WAST/importances for the MLP 
        ei_mlp = pd.read_csv('importances/importances_mnist_0_MLP.csv', header=None)
        e1_mlp = pd.read_csv('importances/importances_mnist_1_MLP.csv', header=None)
        e5_mlp = pd.read_csv('importances/importances_mnist_5_MLP.csv', header=None)
        e10_mlp = pd.read_csv('importances/importances_mnist_10_MLP.csv', header=None)

        # retrieve the importances from literature/WAST/importances for the WAST model
        ei_wast = pd.read_csv('importances/importances_mnist_0_WAST.csv', header=None)
        e1_wast = pd.read_csv('importances/importances_mnist_1_WAST.csv', header=None)
        e5_wast = pd.read_csv('importances/importances_mnist_5_WAST.csv', header=None)
        e10_wast = pd.read_csv('importances/importances_mnist_10_WAST.csv', header=None)

        # reshape the importances to 28x28 (the size of the MNIST images)
        importances_ei_mlp = ei_mlp.values.reshape(28, 28)
        importances_e1_mlp = e1_mlp.values.reshape(28, 28)
        importances_e5_mlp = e5_mlp.values.reshape(28, 28)
        importances_e10_mlp = e10_mlp.values.reshape(28, 28)

        importances_ei_wast = ei_wast.values.reshape(28, 28)
        importances_e1_wast = e1_wast.values.reshape(28, 28)
        importances_e5_wast = e5_wast.values.reshape(28, 28)
        importances_e10_wast = e10_wast.values.reshape(28, 28)


        # make a multiplot of the 9 example images 
        fig, axs = plt.subplots(5, 5, figsize=(10, 10))
        for i, j in itertools.product(range(5), range(5)):
            axs[i, j].imshow(mnist_example[i*5+j], cmap='gray')
            axs[i, j].axis('off')
        
        plt.tight_layout()
        plt.show()


        # make a 2x4 multiplot of the importances with MLP on the top row and WAST on the bottom row
        fig, axs = plt.subplots(2, 4, figsize=(20, 10))
        axs[0, 0].imshow(importances_ei_mlp)
        axs[0, 0].set_title('S-WAST initialization')
        # axs[0, 0].axis('off')
        axs[0, 1].imshow(importances_e1_mlp)
        axs[0, 1].set_title('S-WAST epoch 1')
        # axs[0, 1].axis('off')
        axs[0, 2].imshow(importances_e5_mlp)
        axs[0, 2].set_title('S-WAST epoch 5')
        # axs[0, 2].axis('off')
        axs[0, 3].imshow(importances_e10_mlp)
        axs[0, 3].set_title('S-WAST epoch 10')
        # axs[0, 3].axis('off')
        axs[1, 0].imshow(importances_ei_wast)
        axs[1, 0].set_title('WAST initialization')
        # axs[1, 0].axis('off')
        axs[1, 1].imshow(importances_e1_wast)
        axs[1, 1].set_title('WAST epoch 1')
        # axs[1, 1].axis('off')
        axs[1, 2].imshow(importances_e5_wast)
        axs[1, 2].set_title('WAST epoch 5')
        # axs[1, 2].axis('off')
        axs[1, 3].imshow(importances_e10_wast)
        axs[1, 3].set_title('WAST epoch 10')
        # axs[1, 3].axis('off')

        fig.suptitle('Importances of the pixels in the MNIST dataset')
        plt.tight_layout()
        plt.show()

    elif data == 'FashionMnist':
        # exactly the same idea but for fashionMNIST
        fashion_example = torch.load('data/fashionMNIST/processed/training.pt')
        fashion_example = fashion_example[0][:25]

        ei_mlp = pd.read_csv('importances/importances_FashionMnist_0_MLP.csv', header=None)
        e1_mlp = pd.read_csv('importances/importances_FashionMnist_1_MLP.csv', header=None)
        e5_mlp = pd.read_csv('importances/importances_FashionMnist_5_MLP.csv', header=None)
        e10_mlp = pd.read_csv('importances/importances_FashionMnist_10_MLP.csv', header=None)

        ei_wast = pd.read_csv('importances/importances_FashionMnist_0_WAST.csv', header=None)
        e1_wast = pd.read_csv('importances/importances_FashionMnist_1_WAST.csv', header=None)
        e5_wast = pd.read_csv('importances/importances_FashionMnist_5_WAST.csv', header=None)
        e10_wast = pd.read_csv('importances/importances_FashionMnist_10_WAST.csv', header=None)

        importances_ei_mlp = ei_mlp.values.reshape(28, 28)
        importances_e1_mlp = e1_mlp.values.reshape(28, 28)
        importances_e5_mlp = e5_mlp.values.reshape(28, 28)
        importances_e10_mlp = e10_mlp.values.reshape(28, 28)

        importances_ei_wast = ei_wast.values.reshape(28, 28)
        importances_e1_wast = e1_wast.values.reshape(28, 28)
        importances_e5_wast = e5_wast.values.reshape(28, 28)
        importances_e10_wast = e10_wast.values.reshape(28, 28)


        # plot the example images with one axis for the total plot

        fig, axs = plt.subplots(5, 5, figsize=(10, 10))
        for i, j in itertools.product(range(5), range(5)):
            axs[i, j].imshow(fashion_example[i*5+j], cmap='gray')
        
        # plt.tight_layout()
        plt.show()

        fig, axs = plt.subplots(2, 4, figsize=(20, 10))
        # axs[0, 0].imshow(fashion_example, cmap='gray')
        # axs[0, 0].axis('off')
        # axs[0, 0].set_title('9 Example images')
        axs[0, 0].imshow(importances_ei_mlp)
        axs[0, 0].set_title('S-WAST initialization')
        # axs[0, 1].axis('off')
        axs[0, 1].imshow(importances_e1_mlp)
        axs[0, 1].set_title('S-WAST epoch 1')
        # axs[0, 2].axis('off')
        axs[0, 2].imshow(importances_e5_mlp)
        axs[0, 2].set_title('S-WAST epoch 5')
        # axs[0, 3].axis('off')
        axs[0, 3].imshow(importances_e10_mlp)
        axs[0, 3].set_title('S-WAST epoch 10')
        # axs[0, 4].axis('off')


        # axs[1, 0].imshow(fashion_example, cmap='gray')
        # axs[1, 0].axis('off')
        # axs[1, 0].set_title('Example images')
        axs[1, 0].imshow(importances_ei_wast)
        axs[1, 0].set_title('WAST initialization')
        # axs[1, 1].axis('off')
        axs[1, 1].imshow(importances_e1_wast)
        axs[1, 1].set_title('WAST epoch 1')
        # axs[1, 2].axis('off')
        axs[1, 2].imshow(importances_e5_wast)
        axs[1, 2].set_title('WAST epoch 5')
        # axs[1, 3].axis('off')
        axs[1, 3].imshow(importances_e10_wast)
        axs[1, 3].set_title('WAST epoch 10')
        # axs[1, 4].axis('off')

        fig.suptitle('Importances of the pixels in the FashionMNIST dataset')
        plt.tight_layout()
        plt.show()


    
def plot_features(data='MNIST'):
    """
    The function plots the chosen features on top of an example image from the chosen dataset

    Args:
        data (str, optional): The dataset to use. Defaults to 'MNIST'. Other option is 'fashionMNIST' currently. 

    Returns:
        None
    """

    if data == 'MNIST':
        # Retrieve the example image from the MNIST dataset
        mnist_example = torch.load('data/MNIST/processed/training.pt')
        mnist_example = mnist_example[0][26]
        mnist_example = mnist_example.reshape(28, 28)

        # Retrieve the features from the MLP model, which is a list of the chosen pixels in the image 
        features_e1_mlp = pd.read_csv('features/selected_features_mnist_1_MLP.csv', header=None)
        features_e5_mlp = pd.read_csv('features/selected_features_mnist_5_MLP.csv', header=None)
        features_e10_mlp = pd.read_csv('features/selected_features_mnist_10_MLP.csv', header=None)

        
        # Retrieve the features from the WAST model, which are the chosen pixels in the image
        features_e1_wast = pd.read_csv('features/selected_features_mnist_1_WAST.csv', header=None)
        features_e5_wast = pd.read_csv('features/selected_features_mnist_5_WAST.csv', header=None)
        features_e10_wast = pd.read_csv('features/selected_features_mnist_10_WAST.csv', header=None)

        # convert into a list of tuples of length 28*28=784. The index of the tuple is the pixel number and the value is 1 if it is in the list above, otherwise 0. 
        # Then reshape the list into a 28x28 matrix for plotting
        features_e1_mlp = np.array([(1 if i in features_e1_mlp.values else 0) for i in range(784)]).reshape(28,28)
        features_e5_mlp = np.array([(1 if i in features_e5_mlp.values else 0) for i in range(784)]).reshape(28,28)
        features_e10_mlp = np.array([(1 if i in features_e10_mlp.values else 0) for i in range(784)]).reshape(28,28)

        features_e1_wast = np.array([(1 if i in features_e1_wast.values else 0) for i in range(784)]).reshape(28,28)
        features_e5_wast = np.array([(1 if i in features_e5_wast.values else 0) for i in range(784)]).reshape(28,28)
        features_e10_wast = np.array([(1 if i in features_e10_wast.values else 0) for i in range(784)]).reshape(28,28)

        # Make a 2x3 multiplot of the example image with the chosen features on top, using only markers of the chosen pixels
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        axs[0, 0].imshow(mnist_example, cmap='gray')
        axs[0, 0].scatter(np.where(features_e1_mlp==1)[1], np.where(features_e1_mlp==1)[0], marker='s', c='orangered')
        axs[0, 0].set_title('MLP epoch 1')
        axs[0, 0].axis('off')
        axs[0, 1].imshow(mnist_example, cmap='gray')
        axs[0, 1].scatter(np.where(features_e5_mlp==1)[1], np.where(features_e5_mlp==1)[0], marker='s', c='orangered')
        axs[0, 1].set_title('MLP epoch 5')
        axs[0, 1].axis('off')
        axs[0, 2].imshow(mnist_example, cmap='gray')
        axs[0, 2].scatter(np.where(features_e10_mlp==1)[1], np.where(features_e10_mlp==1)[0], marker='s', c='orangered')
        axs[0, 2].set_title('MLP epoch 10')
        axs[0, 2].axis('off')

        axs[1, 0].imshow(mnist_example, cmap='gray')
        axs[1, 0].scatter(np.where(features_e1_wast==1)[1], np.where(features_e1_wast==1)[0], marker='s', c='orangered')
        axs[1, 0].set_title('WAST epoch 1')
        axs[1, 0].axis('off')
        axs[1, 1].imshow(mnist_example, cmap='gray')
        axs[1, 1].scatter(np.where(features_e5_wast==1)[1], np.where(features_e5_wast==1)[0], marker='s', c='orangered')
        axs[1, 1].set_title('WAST epoch 5')
        axs[1, 1].axis('off')
        axs[1, 2].imshow(mnist_example, cmap='gray')
        axs[1, 2].scatter(np.where(features_e10_wast==1)[1], np.where(features_e10_wast==1)[0], marker='s', c='orangered')
        axs[1, 2].set_title('WAST epoch 10')
        axs[1, 2].axis('off')
        


        plt.tight_layout()
        plt.show()

    elif data == 'FashionMnist':

        # Retrieve the example image from the FashionMnist dataset
        fashionmnist_example = torch.load('data/FashionMNIST/processed/training.pt')
        fashionmnist_example = fashionmnist_example[0][42]
        fashionmnist_example = fashionmnist_example.reshape(28, 28)
        # print(fashionmnist_example.shape)

        # Retrieve the features from the MLP model, which is a list of the chosen pixels in the image
        features_e1_mlp = pd.read_csv('features/selected_features_FashionMnist_1_MLP.csv', header=None)
        features_e5_mlp = pd.read_csv('features/selected_features_FashionMnist_5_MLP.csv', header=None)
        features_e10_mlp = pd.read_csv('features/selected_features_FashionMnist_10_MLP.csv', header=None)
        

        # Retrieve the features from the WAST model, which are the chosen pixels in the image
        features_e1_wast = pd.read_csv('features/selected_features_FashionMnist_1_WAST.csv', header=None)
        features_e5_wast = pd.read_csv('features/selected_features_FashionMnist_5_WAST.csv', header=None)
        features_e10_wast = pd.read_csv('features/selected_features_FashionMnist_10_WAST.csv', header=None)

        # convert into a list of tuples of length 28*28=784. The index of the tuple is the pixel number and the value is 1 if it is in the list above, otherwise 0.
        # Then reshape the list into a 28x28 matrix for plotting
        features_e1_mlp = np.array([(1 if i in features_e1_mlp.values else 0) for i in range(784)]).reshape(28,28)
        features_e5_mlp = np.array([(1 if i in features_e5_mlp.values else 0) for i in range(784)]).reshape(28,28)
        features_e10_mlp = np.array([(1 if i in features_e10_mlp.values else 0) for i in range(784)]).reshape(28,28)

        features_e1_wast = np.array([(1 if i in features_e1_wast.values else 0) for i in range(784)]).reshape(28,28)
        features_e5_wast = np.array([(1 if i in features_e5_wast.values else 0) for i in range(784)]).reshape(28,28)
        features_e10_wast = np.array([(1 if i in features_e10_wast.values else 0) for i in range(784)]).reshape(28,28)

        # Make a 2x3 multiplot of the example image with the chosen features on top, using only markers of the chosen pixels
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        axs[0, 0].imshow(fashionmnist_example, cmap='gray')
        axs[0, 0].scatter(np.where(features_e1_mlp==1)[1], np.where(features_e1_mlp==1)[0], marker='s', c='orangered')
        axs[0, 0].set_title('MLP epoch 1')
        axs[0, 0].axis('off')
        axs[0, 1].imshow(fashionmnist_example, cmap='gray')
        axs[0, 1].scatter(np.where(features_e5_mlp==1)[1], np.where(features_e5_mlp==1)[0], marker='s', c='orangered')
        axs[0, 1].set_title('MLP epoch 5')
        axs[0, 1].axis('off')
        axs[0, 2].imshow(fashionmnist_example, cmap='gray')
        axs[0, 2].scatter(np.where(features_e10_mlp==1)[1], np.where(features_e10_mlp==1)[0], marker='s', c='orangered')
        axs[0, 2].set_title('MLP epoch 10')
        axs[0, 2].axis('off')
        
        axs[1, 0].imshow(fashionmnist_example, cmap='gray')
        axs[1, 0].scatter(np.where(features_e1_wast==1)[1], np.where(features_e1_wast==1)[0], marker='s', c='orangered')
        axs[1, 0].set_title('WAST epoch 1')
        axs[1, 0].axis('off')
        axs[1, 1].imshow(fashionmnist_example, cmap='gray')
        axs[1, 1].scatter(np.where(features_e5_wast==1)[1], np.where(features_e5_wast==1)[0], marker='s', c='orangered')
        axs[1, 1].set_title('WAST epoch 5')
        axs[1, 1].axis('off')
        axs[1, 2].imshow(fashionmnist_example, cmap='gray')
        axs[1, 2].scatter(np.where(features_e10_wast==1)[1], np.where(features_e10_wast==1)[0], marker='s', c='orangered')
        axs[1, 2].set_title('WAST epoch 10')
        axs[1, 2].axis('off')

        plt.tight_layout()
        plt.show()



    
    else:
        # unknown dataset chosen so raise an error
        raise ValueError('Unknown dataset chosen, valid options are: MNIST, fashionMNIST')

def plot_average(data='MNIST'):
    """
    Plot the average of a chosen dataset
    
    Args:
        data (str): The dataset to plot the average of. Valid options are: MNIST, fashionMNIST
    """

    if data == 'MNIST':
        # Retrieve all MNIST images from data/MNIST/processed/training.pt
        mnist_data = torch.load('data/MNIST/processed/training.pt')
        mnist_images = mnist_data[0]

        # print the shape of the images
        print(f'Shape of MNIST images: {mnist_images.shape}')
                

        # Calculate the average of all 60000 images (60000,28,28)
        mnist_average = np.average(mnist_images.numpy(), axis=0)



        # Plot the average
        plt.imshow(mnist_average)
        plt.title('Average of MNIST')
        # plt.axis('off')
        plt.show()

    elif data == 'FashionMnist':
        # Retrieve all FashionMNIST images from data/FashionMNIST/processed/training.pt
        fashionmnist_data = torch.load('data/FashionMNIST/processed/training.pt')
        fashionmnist_images = fashionmnist_data[0]

        # print the shape of the images
        print(f'Shape of FashionMNIST images: {fashionmnist_images.shape}')

        # Calculate the average of all 60000 images (60000,28,28)
        fashionmnist_average = np.average(fashionmnist_images.numpy(), axis=0)

        # Plot the average
        plt.imshow(fashionmnist_average)
        plt.title('Average of FashionMNIST')
        # plt.axis('off')
        plt.show()

    else:
        # unknown dataset chosen so raise an error
        raise ValueError('Unknown dataset chosen, valid options currently are: MNIST, fashionMNIST')





if __name__ == '__main__':
    # plot_importances(args.data)
    # 
    plot_features(args.data)
    # plot_average(args.data)


