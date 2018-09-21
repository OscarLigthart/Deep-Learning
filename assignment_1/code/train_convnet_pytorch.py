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
import matplotlib.pyplot as plt

# Default constants
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 32
MAX_STEPS_DEFAULT = 5000
EVAL_FREQ_DEFAULT = 100 # CHANGED THE EVAL FREQ IN ORDER TO CREATE MORE CHECKPOINTS, FOR NICER PLOTS
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

  # use predictions and targets to calculate accuracy
  pred = np.argmax(predictions, axis=1)

  lab = np.argmax(targets, axis=1)

  # check how many predictions are equal to the labels
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

  # get device
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  # loop through data
  cifar10 = cifar10_utils.get_cifar10('cifar10/cifar-10-batches-py', validation_size=2000)
  x, y = cifar10['train'].next_batch(FLAGS.batch_size)

  # get channels
  n_channels = np.size(x, 1)

  # create model
  net = ConvNet(n_channels, 10)
  net.to(device)

  # get loss function and optimizer
  crossEntropy = nn.CrossEntropyLoss()

  # keep track of loss and accuracy
  loss_list = []
  loss_val_list = []
  accuracy_train_list = []
  accuracy_val_list = []

  # set optimizer
  optimizer = torch.optim.Adam(net.parameters(), lr=FLAGS.learning_rate)

  # loop for the amount of steps
  for i in range(FLAGS.max_steps):

    # create torch compatible input
    x = Variable(torch.from_numpy(x), requires_grad=True)
    x = x.to(device)

    # perform forward pass
    out = net(x)

    # convert one hot to indices and create torch compatible input
    label_index = np.argmax(y, axis=1)
    label_index = torch.LongTensor(label_index)
    label_index = label_index.to(device)

    # apply cross entropy
    loss = crossEntropy(out, label_index)

    # show progress and run network on validation set
    if i % FLAGS.eval_freq == 0:

      # convert output to numpy array, differs when using cuda compared to cpu
      if torch.cuda.is_available():
        train_out = out.cpu()
        out_numpy = train_out.data[:].numpy()
      else:
        out_numpy = out.data[:].numpy()

      # calculate accuracy
      accuracy_train = accuracy(out_numpy, y)

      # don't track the gradients
      with torch.no_grad():

        # load validation data
        x_val, y_val = cifar10['validation'].next_batch(FLAGS.batch_size)

        # create torch compatible input
        x_val = Variable(torch.from_numpy(x_val), requires_grad=False)
        x_val = x_val.to(device)

        # run on validation set
        val_out = net.forward(x_val)

        # convert one hot to indices and create torch compatible input
        y_val_index = np.argmax(y_val, axis=1)
        y_val_index = torch.LongTensor(y_val_index)
        y_val_index = y_val_index.to(device)

        # apply cross entropy
        loss_val = crossEntropy.forward(val_out, y_val_index)

        # convert output to numpy array, differs when using cuda compared to cpu
        if torch.cuda.is_available():
          val_out = val_out.cpu()
          val_out_numpy = val_out.data[:].numpy()
          loss_val = loss_val.cpu()
          loss_val = loss_val.data.numpy()

          loss_train = loss.cpu()
          loss_train = loss_train.data.numpy()

        else:
          val_out_numpy = val_out.data[:].numpy()
          loss_val = loss_val.data.numpy()
          loss_train = loss.data.numpy()

        accuracy_val = accuracy(val_out_numpy, y_val)

      # save variables
      accuracy_train_list.append(accuracy_train)
      accuracy_val_list.append(accuracy_val)
      loss_list.append(loss_train)
      loss_val_list.append(loss_val)

      # print progress
      print("##############################################################")
      print("Epoch ", i)
      print("---------------------------------------------------------------")
      print("The ACCURACY on the TRAIN set is currently: ", accuracy_train)
      print("---------------------------------------------------------------")
      print("The ACCURACY on the VALIDATION set is currently:", accuracy_val)
      print("---------------------------------------------------------------")
      print("The LOSS on the TRAIN set is currently:", loss_train)
      print("---------------------------------------------------------------")
      print("The LOSS on the VALIDATION set is currently:", loss_val)
      print("---------------------------------------------------------------")
      print("###############################################################")
      print("\n")

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # insert new databatch for next loop
    x, y = cifar10['train'].next_batch(FLAGS.batch_size)

  # run test data through network without tracking the gradients
  with torch.no_grad():
    # test
    x, y = cifar10['test'].images, cifar10['test'].labels

    # convert variable to torch compatible input
    x = Variable(torch.from_numpy(x), requires_grad=False)
    x = x.to(device)

    # get output
    out = net(x)

    # convert output to numpy array, differs when using cuda compared to cpu
    if torch.cuda.is_available():
      out = out.cpu()
      out_numpy = out.data[:].numpy()
    else:
      out_numpy = out.data[:].numpy()

  # calculate accuracy
  test_accuracy = accuracy(out_numpy, y)
  print("The accuracy on the test set is:")
  print(test_accuracy)

  # save test, training and validation accuracies and losses to make a plot afterwards
  lists = [accuracy_train_list, accuracy_val_list, loss_list, loss_val_list, test_accuracy]
  pickle.dump(lists, open("lists.p", "wb"))

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