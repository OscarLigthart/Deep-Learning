"""
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from mlp_pytorch import MLP
import cifar10_utils
import torch
from torch import nn
from torch import functional as F
from torch.autograd import Variable

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '100,50,25'
LEARNING_RATE_DEFAULT = 5e-3
MAX_STEPS_DEFAULT = 3000
BATCH_SIZE_DEFAULT = 100
EVAL_FREQ_DEFAULT = 100

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
  Performs training and evaluation of MLP model. 

  TODO:
  Implement training and evaluation of MLP model. Evaluate your model on the whole test set each eval_freq iterations.
  """

  ### DO NOT CHANGE SEEDS!
  # Set the random seeds for reproducibility
  np.random.seed(42)
  torch.manual_seed(42)

  ## Prepare all functions
  # Get number of units in each hidden layer specified in the string such as 100,100
  if FLAGS.dnn_hidden_units:
    dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
    dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
  else:
    dnn_hidden_units = []

  ########################
  # PUT YOUR CODE HERE  #
  #######################

  # loop through data
  cifar10 = cifar10_utils.get_cifar10('cifar10/cifar-10-batches-py')
  x, y = cifar10['train'].next_batch(BATCH_SIZE_DEFAULT)
  print(y.shape)
  print(x.shape)
  x = x.reshape(np.size(x,0), -1)

  n_input = np.size(x, 1)

  # create model
  net = MLP(n_input, dnn_hidden_units, 10)

  # get loss function and optimizer
  crossEntropy = nn.CrossEntropyLoss()

  optimizer = torch.optim.SGD(net.parameters(), lr=FLAGS.learning_rate)

  for i in range(FLAGS.max_steps):

    x = Variable(torch.from_numpy(x), requires_grad = True)

    out = net(x)
    out_numpy = out.data[:].numpy()

    # apply cross entropy
    label_index = np.argmax(y, axis=1)
    label_index = torch.LongTensor(label_index)

    loss = crossEntropy(out, label_index)

    if i % FLAGS.eval_freq == 0:
      print(accuracy(out_numpy, y))
      print(loss)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # insert data
    x, y = cifar10['train'].next_batch(FLAGS.batch_size)
    x = x.reshape(np.size(x, 0), -1)



  # test
  x, y = cifar10['test'].images, cifar10['test'].labels
  x = x.reshape(np.size(x, 0), -1)

  x = Variable(torch.from_numpy(x), requires_grad = False)
  out = net(x)
  out_numpy = out.data[:].numpy()
  print("The accuracy on the test set is:")
  print(accuracy(out_numpy,y))

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
  parser.add_argument('--dnn_hidden_units', type = str, default = DNN_HIDDEN_UNITS_DEFAULT,
                      help='Comma separated list of number of units in each hidden layer')
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