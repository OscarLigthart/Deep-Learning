################################################################################
# MIT License
# 
# Copyright (c) 2018
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2018
# Date Created: 2018-09-04
################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time
from datetime import datetime
import numpy as np

import torch
from torch.utils.data import DataLoader

from dataset import PalindromeDataset
from vanilla_rnn import VanillaRNN
from lstm import LSTM

import pickle

# You may want to look into tensorboardX for logging
# from tensorboardX import SummaryWriter

################################################################################

def train(config):

    assert config.model_type in ('RNN', 'LSTM')

    # Initialize the device which to run the model on
    device = torch.device(config.device)

    # Initialize the model that we are going to use
    if config.model_type == 'RNN':
        model = VanillaRNN(config.input_length, config.input_dim, config.num_hidden, config.num_classes, config.batch_size, config.device)
    elif config.model_type == 'LSTM':
        model = LSTM(config.input_length, config.input_dim, config.num_hidden, config.num_classes, config.batch_size, config.device)
    else:
        AssertionError('Models available: RNN, LSTM')

    # Initialize the dataset and data loader (note the +1)
    dataset = PalindromeDataset(config.input_length+1)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

    # Setup the loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), config.learning_rate)

    # keep track of variables
    accuracy_list = []
    loss_list = []

    # loop through data (get batches)
    for step, (batch_inputs, batch_targets) in enumerate(data_loader):

        # Only for time measurement of step through network
        t1 = time.time()

        # get model predictions (USE THE PLAIN INTEGERS FROM THE PALINDROME TO INSERT INTO THE MODEL AS INPUT)
        predictions = model(batch_inputs)

        ############################################################################
        # QUESTION: what happens here and why?
        #
        # it clips the gradient to a border value, to prevent exploding or vanishing
        # gradients.
        #
        ############################################################################
        torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=config.max_norm)
        ############################################################################

        # calculate loss
        loss = criterion(predictions, batch_targets)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        targets = batch_targets.data.numpy()
        predictions_np = predictions.data.numpy()

        # get amount of correct predictions
        correct = 0
        for i in range(len(targets)):
            if targets[i] == np.argmax(predictions_np[i]):
                correct += 1

        # get accuracy
        accuracy = correct / len(targets)

        # Just for time measurement
        t2 = time.time()
        examples_per_second = config.batch_size/float(t2-t1)

        if step % 10 == 0:

            print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
                  "Accuracy = {:.2f}, Loss = {:.3f}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                    config.train_steps, config.batch_size, examples_per_second,
                    accuracy, loss
            ))

            # keep track of losses
            loss_list.append(loss.data.numpy())

            # keep track of accuracies
            accuracy_list.append(accuracy)

        if step == config.train_steps:
            # If you receive a PyTorch data-loader error, check this bug report:
            # https://github.com/pytorch/pytorch/pull/9655
            break

    # save variables
    data = [loss_list, accuracy_list]
    filename = 'Result_' + config.model_type + '_inputlen_' + str(config.input_length) + '.p'
    pickle.dump(data, open(filename, 'wb'))

    print('Done training.')


 ################################################################################
 ################################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--model_type', type=str, default="RNN", help="Model type, should be 'RNN' or 'LSTM'")
    parser.add_argument('--input_length', type=int, default=10, help='Length of an input sequence')
    parser.add_argument('--input_dim', type=int, default=1, help='Dimensionality of input sequence')
    parser.add_argument('--num_classes', type=int, default=10, help='Dimensionality of output sequence')
    parser.add_argument('--num_hidden', type=int, default=128, help='Number of hidden units in the model')
    parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--train_steps', type=int, default=10000, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=10.0)
    parser.add_argument('--device', type=str, default="cpu", help="Training device 'cpu' or 'cuda:0'")

    config = parser.parse_args()

    train(config)