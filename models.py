import torch
import torch.nn as nn

from set_mlp_sequential import *

# an autoencoder model with a single hidden layer
class AE(nn.Module):
    def __init__(self, dataset, input_size, nhidden):
        super(AE, self).__init__()
        self.fc1 = nn.Linear(input_size, nhidden, bias=True)
        self.fc2 = nn.Linear(nhidden, input_size, bias=True)
        self.input_size = input_size
        self.dataset = dataset 
    def forward(self, x):
        x0 = x.view(-1, self.input_size)
        x1 = torch.sigmoid(self.fc1(x0))
        return self.fc2(x1)


# a MLP model with a single hidden layer
class MLP(nn.Module):
    def __init__(self, dataset, input_size, nhidden, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, nhidden, bias=True)
        self.fc2 = nn.Linear(nhidden, output_size, bias=True)
        self.input_size = input_size
        self.output_size = output_size
        self.dataset = dataset 
    def forward(self, x):
        x0 = x.view(-1, self.input_size)
        x1 = torch.relu(self.fc1(x0))
        return self.fc2(x1)

# a MLP model with two hidden layers, for later experimentation
class MLP_2(nn.Module):
    def __init__(self, dataset, input_size, nhidden, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, nhidden, bias=True)
        self.fc2 = nn.Linear(nhidden, nhidden, bias=True)
        self.fc3 = nn.Linear(nhidden, output_size, bias=True)
        self.input_size = input_size
        self.output_size = output_size
        self.dataset = dataset 
    def forward(self, x):
        x0 = x.view(-1, self.input_size)
        x1 = torch.relu(self.fc1(x0))
        x2 = torch.relu(self.fc2(x1))
        return self.fc3(x2)

