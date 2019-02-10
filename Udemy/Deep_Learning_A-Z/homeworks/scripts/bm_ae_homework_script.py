import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

############################## Data preprocessing ##############################

base_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath('__file__')))), 
                          'ressources/Boltzmann_Machines/')

movies = pd.read_csv(base_path + 'ml-1m/movies.dat', sep='::', header=None, engine='python', encoding='latin-1')
users = pd.read_csv(base_path + 'ml-1m/users.dat', sep='::', header=None, engine='python', encoding='latin-1')
ratings = pd.read_csv(base_path + 'ml-1m/ratings.dat', sep='::', header=None, engine='python', encoding='latin-1')

# cols are the same as for ratings dataset
training_set = pd.read_csv(base_path + 'ml-100k/u1.base', delimiter='\t')
test_set = pd.read_csv(base_path + 'ml-100k/u1.test', delimiter='\t')

# convert data to array
training_set = np.array(training_set, dtype='int')
test_set = np.array(test_set, dtype='int')

# Getting the number of users and movies
nb_users = int(max(max(training_set[:,0]), max(test_set[:,0])))
nb_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))

# Converting the data into an array with users in lines and movies in columns
def convert(data):
    new_data = []
    for id_users in range(1, nb_users + 1):
        id_movies = data[:,1][data[:,0] == id_users]
        id_ratings = data[:,2][data[:,0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data
training_set = convert(training_set)
test_set = convert(test_set)

# Converting the data into Torch tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

############################## Building binary output RBM ##############################

training_set[training_set == 0] = -1 # replace 0 by -1 for non rated movies
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
training_set[training_set >= 3] = 1
test_set[test_set == 0] = -1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1

class RBM: 
    
    """
    Create a Bernouilli RBM that will predict if user liked of not a movie, using a binary output
    """
    
    def __init__(self, nv, nh):
        """
        Args:
        nv (int): number of visible nodes
        nh (int): number of hidden nodes
        """
        self.Weights = torch.randn(nh, nv) # initialize weights
        self.bias_hidden = torch.randn(1, nh) # initialize bias of hidden nodes (fist dimension is batch, second is bias)
        self.bias_visible = torch.randn(1, nv) # initialize bias of visible nodes
        
    def sample_hidden(self, x):
        """
        Function which will activate hidden nodes according to a certain probability given the input nodes. 
        This probability is the sigmoid function applied to (self.Weights * x + self.bias_hidden).
        """
        wx = torch.mm(x, self.Weights.t()) # make product of two tensors
        activation = wx + self.bias_hidden.expand_as(wx) # expand add a new dimention to make sure that bias is applied to each line of the mini batch (arg "1")
        p_hidden_given_visible = torch.sigmoid(activation) # represents the probability of the hidden node to active given visible node
        
        # Bernouilli function will create a random number between 0 and 1 and return one if 
        # this number is higher than p_hidden_given_visible, and 0 else.
        
        return p_hidden_given_visible, torch.bernoulli(p_hidden_given_visible)
    
    def sample_visible(self, y):
        """
        Function which will activate visible nodes according to a certain probability given the hidden nodes. 
        This probability is the sigmoid function applied to (self.Weights * y + self.bias_visible).
        """
        wy = torch.mm(y, self.Weights)
        activation = wy + self.bias_visible.expand_as(wy)
        p_visible_given_hidden = torch.sigmoid(activation) 
        
        return p_visible_given_hidden, torch.bernoulli(p_visible_given_hidden)
    
    def train(self, v0, vk, ph0, phk):
        """
        Use contrastive divergence to train our RBM

        Args:
        v0 : input vector containing ratings
        vk : visible nodes obtained after k steps of contrastive divergence
        ph0 : initial probability of hidden nodes
        phk : probability of hidden nodes after k gibbs samplings
        """
        self.Weights += (torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)).t()

        self.bias_visible += torch.sum((v0 - vk), 0)
        self.bias_hidden += torch.sum((ph0 - phk), 0)
            
     
nv = len(training_set[0]) # lenght of fisrt line of training set is the number of inputs
nh = 100 # 1682 movies so the model may detect many features, we can start by 100 of them
batch_size = 100 # Start with 100 for a fast training (increase diminushes precision)
rbm = RBM(nv, nh) # Create restricted Boltzmann machine
            

# Training the RBM
nb_epoch = 10
for epoch in range(1, nb_epoch + 1):
    train_loss = 0 # introduce a loss variable which will increase when we will find differences between predictions and answer
    counter = 0.
    for id_user in range(0, nb_users - batch_size, batch_size):
        # create batchs of users
        vk = training_set[id_user:id_user+batch_size] # vector that will be the input of gibbs chain and will be updated at each input
        v0 = training_set[id_user:id_user+batch_size] 
        ph0,_ = rbm.sample_hidden(v0)
        for k in range(10): # 10 is an hyperparameter, number of random walks
            _,hk = rbm.sample_hidden(vk) # do gibbs chain
            _,vk = rbm.sample_visible(hk) # do gibbs chain (update of visible nodes)
            vk[v0<0] = v0[v0<0] 
        phk,_ = rbm.sample_hidden(vk)
        rbm.train(v0, vk, ph0, phk)
        train_loss += torch.mean(torch.abs(v0[v0>=0] - vk[v0>=0])) # update train loss
        counter += 1.
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/counter))
    
# Testing the RBM
test_loss = 0
counter = 0.
for id_user in range(nb_users):
    v = training_set[id_user:id_user+1] # we use inputs of training set to activate neurons to predict output for test set
    vt = test_set[id_user:id_user+1]
    if len(vt[vt>=0]) > 0:
        _,h = rbm.sample_hidden(v)
        _,v = rbm.sample_visible(h)
        test_loss += torch.mean(torch.abs(vt[vt>=0] - v[vt>=0]))
        counter += 1.
print('test loss: '+str(test_loss/counter))