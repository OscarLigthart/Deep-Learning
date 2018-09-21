"""
This module implements training and evaluation of a multi-layer perceptron in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from mlp_numpy import MLP
from modules import CrossEntropyModule
import cifar10_utils
import matplotlib
import matplotlib.pyplot as plt

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '100'
LEARNING_RATE_DEFAULT = 2e-3
MAX_STEPS_DEFAULT = 1500
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 100 # CHANGED IT TO 10 FOR THE SAKE OF MAKING PLOTS

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

  accuracy = count/len(pred)


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

  # initialize cross entropy module
  crossEntropy = CrossEntropyModule()

  # loop through data
  cifar10 = cifar10_utils.get_cifar10(data_dir=FLAGS.data_dir, validation_size=2000)
  x, y = cifar10['train'].next_batch(FLAGS.batch_size)
  x = x.reshape(np.size(x,0), -1)

  # get input size for network initialization
  n_input = np.size(x, 1)

  loss_list = []
  loss_val_list = []
  accuracy_train_list = []
  accuracy_val_list = []

  # create model
  net = MLP(n_input, dnn_hidden_units, 10)

  # loop for set amount of steps
  for i in range(FLAGS.max_steps):

    # get network output
    out = net.forward(x)

    # apply cross entropy
    loss = crossEntropy.forward(out, y)
    accuracy_train = accuracy(out, y)

    # get gradients of Cross Entropy
    dout = crossEntropy.backward(out, y)

    # backpropagate through the network
    net.backward(dout)

    # now upgrade weights
    for layer in net.layers:
      layer.params['weight'] = layer.params['weight'] - FLAGS.learning_rate * layer.grads['weight']
      layer.params['bias'] = layer.params['bias'] - FLAGS.learning_rate * layer.grads['bias']

    net.lastlayer.params['weight'] = net.lastlayer.params['weight'] - FLAGS.learning_rate * net.lastlayer.grads['weight']
    net.lastlayer.params['bias'] = net.lastlayer.params['bias'] - FLAGS.learning_rate * net.lastlayer.grads['bias']

    # load new batch for next epoch data
    x, y = cifar10['train'].next_batch(FLAGS.batch_size)
    x = x.reshape(np.size(x, 0), -1)

    # show progress
    if i % FLAGS.eval_freq == 0:
      # load validation data
      x_val, y_val = cifar10['validation'].next_batch(FLAGS.batch_size)
      x_val = x_val.reshape(np.size(x_val, 0), -1)

      # run on validation set
      val_out = net.forward(x_val)
      accuracy_val = accuracy(val_out, y_val)
      loss_val = crossEntropy.forward(val_out, y_val)

      # save variables
      accuracy_train_list.append(accuracy_train)
      accuracy_val_list.append(accuracy_val)
      loss_list.append(loss)
      loss_val_list.append(loss_val)
      print("##############################################################")
      print("Epoch ", i)
      print("---------------------------------------------------------------")
      print("The ACCURACY on the TRAIN set is currently: ", accuracy_train)
      print("---------------------------------------------------------------")
      print("The ACCURACY on the VALIDATION set is currently:", accuracy_val)
      print("---------------------------------------------------------------")
      print("The LOSS on the TRAIN set is currently:", loss)
      print("---------------------------------------------------------------")
      print("The LOSS on the VALIDATION set is currently:", loss_val)
      print("---------------------------------------------------------------")
      print("###############################################################")
      print("\n")


  # run model on test set and show accuracy
  x, y = cifar10['test'].images, cifar10['test'].labels
  x = x.reshape(np.size(x, 0), -1)
  out = net.forward(x)
  print("The accuracy on the test set is:")
  print(accuracy(out,y))

  # plot
  fig, ax1 = plt.subplots()

  ax1.plot(loss_list, label="Train loss", color = 'firebrick')
  ax1.plot(loss_val_list, label="Validation loss", color = 'darksalmon')
  ax1.set_title("Accuracy and Loss curves")
  ax1.set_ylabel("Loss")
  ax1.set_xlabel("Evaluation")
  ax1.tick_params(axis='y', labelcolor='red')
  ax1.legend(loc = 'upper left')

  ax2 = ax1.twinx()

  ax2.plot(accuracy_train_list, label='Train accuracy', color = 'royalblue')
  ax2.plot(accuracy_val_list, label='Validation accuracy', color = 'lightskyblue')
  ax2.tick_params(axis='y', labelcolor='blue')
  ax2.set_ylabel("Accuracy")

  fig.tight_layout()
  ax2.legend(loc = 'upper right')
  plt.show()

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