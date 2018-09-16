"""
This module implements training and evaluation of a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from convnet_pytorch import ConvNet
import cifar10_utils

import pickle
import torch
from torch import nn
from torch import functional as F
from torch.autograd import Variable

# Default constants
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 32
MAX_STEPS_DEFAULT = 5000
EVAL_FREQ_DEFAULT = 500
OPTIMIZER_DEFAULT = 'ADAM'

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None

def accuracy(predictions, targets):
  """
  Computes the prediction accuracy, i.e. the average of correct predictions
  of the network.
  
  Args:
    predictions: 2D float array of size [batch_size, n_classes]
    labels: 2D int array of size [batch_size, n_classes]
            with one-hot encoding. Ground truth labels for
            each sample in the batch
  Returns:
    accuracy: scalar float, the accuracy of predictions,
              i.e. the average correct predictions over the whole batch
  
  TODO:
  Implement accuracy computation.
  """

  ########################
  # PUT YOUR CODE HERE  #
  #######################
  # use predictions and targets to
  pred = np.argmax(predictions, axis=1)

  lab = np.argmax(targets, axis=1)

  count = 0
  for i in range(len(pred)):
    if pred[i] == lab[i]:
      count += 1

  accuracy = count / len(pred)
  ########################
  # END OF YOUR CODE    #
  #######################

  return accuracy

def train():
  """
  Performs training and evaluation of ConvNet model. 

  TODO:
  Implement training and evaluation of ConvNet model. Evaluate your model on the whole test set each eval_freq iterations.
  """

  ### DO NOT CHANGE SEEDS!
  # Set the random seeds for reproducibility
  np.random.seed(42)

  ########################
  # PUT YOUR CODE HERE  #
  #######################
  # loop through data
  cifar10 = cifar10_utils.get_cifar10('cifar10/cifar-10-batches-py')
  x, y = cifar10['train'].next_batch(FLAGS.batch_size)
  print(y.shape)
  print(x.shape)

  # get channels
  n_channels = np.size(x, 1)

  # create model
  net = ConvNet(n_channels, 10)

  # get loss function and optimizer
  crossEntropy = nn.CrossEntropyLoss()

  loss_list = []
  accuracy_list = []

  optimizer = torch.optim.Adam(net.parameters(), lr=FLAGS.learning_rate)

  for i in range(FLAGS.max_steps):

    x = Variable(torch.from_numpy(x), requires_grad=True)

    out = net(x)
    out_numpy = out.data[:].numpy()

    # apply cross entropy
    label_index = np.argmax(y, axis=1)
    label_index = torch.LongTensor(label_index)

    loss = crossEntropy(out, label_index)

    if i % FLAGS.eval_freq == 0:
      print(accuracy(out_numpy, y))
      print(loss)
      pickle.dump(loss_list, open("losses.p", "wb"))
      pickle.dump(accuracy_list, open("accuracies.p", "wb"))


    loss_list.append(loss)
    accuracy_list.append(accuracy(out_numpy, y))

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # insert data
    x, y = cifar10['train'].next_batch(FLAGS.batch_size)

  # test
  x, y = cifar10['test'].images, cifar10['test'].labels

  x = Variable(torch.from_numpy(x), requires_grad=False)
  out = net(x)
  out_numpy = out.data[:].numpy()
  print("The accuracy on the test set is:")
  print(accuracy(out_numpy, y))

  # save model
  torch.save(net, 'ConvNet.pt')

  ########################
  # END OF YOUR CODE    #
  #######################

def print_flags():
  """
  Prints all entries in FLAGS variable.
  """
  for key, value in vars(FLAGS).items():
    print(key + ' : ' + str(value))

def main():
  """
  Main function
  """
  # Print all Flags to confirm parameter settings
  print_flags()

  if not os.path.exists(FLAGS.data_dir):
    os.makedirs(FLAGS.data_dir)

  # Run the training operation
  train()

if __name__ == '__main__':
  # Command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
  parser.add_argument('--max_steps', type = int, default = MAX_STEPS_DEFAULT,
                      help='Number of steps to run trainer.')
  parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                      help='Batch size to run trainer.')
  parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
  parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()

  main()