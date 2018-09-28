# MIT License
#
# Copyright (c) 2017 Tom Runia
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

import os
import time
from datetime import datetime
import argparse

import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import TextDataset
from model import TextGenerationModel

import pickle
import os

################################################################################

def accuracies(pred, target):

    pred = torch.argmax(pred, dim =1)
    n, d = target.shape[0], target.shape[1]
    # get amount of correct predictions
    correct = 0
    for i in range(n):
        for j in range(d):
            if target[i,j] == pred[i,j]:
                correct += 1

    acc = correct/ (n*d)

    return acc



def train(config):

    # Initialize the device which to run the model on
    device = torch.device(config.device)

    # Initialize the dataset and data loader (note the +1)
    dataset = TextDataset(config.txt_file, config.seq_length)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1, drop_last=True)
    vocab_size = dataset.vocab_size

    # Initialize the model that we are going to use
    model = TextGenerationModel(config.batch_size, config.seq_length, vocab_size, config.lstm_num_hidden, config.lstm_num_layers, config.device)
    model.to(device)

    # Setup the loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    criterion = criterion.to(device)


    # if pickle file is available, load steps and use index -1 to get last step + get lists of values
    if os.path.isfile("steps.p"):
        print('Pre-trained model available...')
        print('Resuming training...')
        # load lists
        step_intervals = pickle.load(open("steps.p", "rb"))
        all_sentences = pickle.load(open("sentences.p", "rb"))
        accuracy_list = pickle.load(open("accuracies.p", "rb"))
        loss_list = pickle.load(open("loss.p", "rb"))

        # start where we left off
        all_steps = step_intervals[-1]


        # load model
        model = torch.load('TrainIntervalModel.pt')
        model.to(device)

        # load previous learning rate
        learning_rate = pickle.load(open("lr.p", "rb"))

        # initialize optimizer with
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)

    else:
        print('No pre-trained model available...')
        print('Initializing training...')

        # create lists to keep track of data while training
        all_sentences = []
        step_intervals = []
        accuracy_list = []
        loss_list = []

        # initialize total step counter
        all_steps = 0

        # initialize optimizer with starting learning rate
        optimizer = torch.optim.RMSprop(model.parameters(), lr=config.learning_rate)

    # initialize optimizer with previous learning rate. (extract from pickle then use scheduler)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.learning_rate_step, gamma=config.learning_rate_decay)

    # since the nested for loop stops looping after a complete iteration through the data_loader, add for loop for epochs
    for epoch in range(config.epochs):
        for step, (batch_inputs, batch_targets) in enumerate(data_loader):

            # Only for time measurement of step through network
            t1 = time.time()

            # apply scheduler
            #scheduler.step()

            # check if loaded data is of batch_size (input is not of that size at the end of the .txt file)
            if batch_inputs[0].shape[0] < config.batch_size:
                continue

            # create 2D tensor instead of list of 1D tensors
            #batch_inputs = torch.stack(batch_inputs)
            batch_inputs = batch_inputs.to(device)

            out = model(batch_inputs)

            # transpose to match cross entropy input dimensions
            out.transpose_(1, 2)

            batch_targets = torch.stack(batch_targets)
            batch_targets = batch_targets.to(device)


            #######################################################
            # Add more code here ...
            #######################################################

            loss = criterion(out, batch_targets)
            accuracy = accuracies(out, batch_targets)


            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Just for time measurement
            t2 = time.time()
            examples_per_second = config.batch_size/float(t2-t1)

            if step % config.print_every == 0:

                print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
                      "Accuracy = {:.2f}, Loss = {:.3f}".format(
                        datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                        int(config.train_steps), config.batch_size, examples_per_second,
                        accuracy, loss
                ))

            if all_steps % config.sample_every == 0:

                ###############################
                # Generate generated sequence #
                ###############################

                model.eval()

                random_input = torch.randint(0, vocab_size, (config.batch_size,)).view(1,-1)

                sentences = model(random_input)

                # pick a random sentence (from the batch)
                index = np.random.randint(0, config.batch_size, 1)
                sentence = sentences[:,index,:]

                # get predictions
                sentence = torch.argmax(sentence, dim=2)

                # squeeze sentence into 1D
                sentence = sentence.view(-1).cpu()

                sentence = torch.cat((random_input[0][index].long(), sentence), dim=0)

                # print sentence
                print(dataset.convert_to_string(sentence.data.numpy()))

                # save sentence
                all_sentences.append(sentence.data.numpy())

                model.train()

                ##########################
                # Save loss and accuracy #
                ##########################

                # save loss value
                loss = loss.cpu()
                loss_list.append(loss.data.numpy())

                # save accuracy value
                accuracy_list.append(accuracy)

                # save step interval
                step_intervals.append(all_steps)


            if step == config.train_steps:
                # If you receive a PyTorch data-loader error, check this bug report:
                # https://github.com/pytorch/pytorch/pull/9655
                break

            # counter of total amounts of steps (keep track over multiple trainingsessions)
            all_steps += 1

        # pickle sentences and steps
        pickle.dump(all_sentences, open('sentences.p', 'wb'))
        pickle.dump(step_intervals, open('steps.p', 'wb'))

        # pickle accuracy and loss
        pickle.dump(accuracy_list, open('accuracies.p', 'wb'))
        pickle.dump(loss_list, open('loss.p', 'wb'))

        # save model
        Modelname = 'TrainIntervalModel' + str(epoch) + 'acc:' + str(accuracy) + '.pt'
        torch.save(model, Modelname)

        # save learning rate
        pickle.dump(optimizer.param_groups[0]['lr'], open('lr.p', 'wb'))

    print('Done training.')


 ################################################################################
 ################################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--txt_file', type=str, default='assets/book_EN_grimms_fairy_tails.txt', help="Path to a .txt file to train on")
    #parser.add_argument('--txt_file', type=str, default='assets/book_NL_darwin_reis_om_de_wereld.txt', help="Path to a .txt file to train on")
    parser.add_argument('--seq_length', type=int, default=30, help='Length of an input sequence')
    parser.add_argument('--lstm_num_hidden', type=int, default=128, help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=2, help='Number of LSTM layers in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=64, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=2e-3, help='Learning rate')

    # It is not necessary to implement the following three params, but it may help training.
    parser.add_argument('--learning_rate_decay', type=float, default=0.96, help='Learning rate decay fraction')
    parser.add_argument('--learning_rate_step', type=int, default=5000, help='Learning rate step')
    parser.add_argument('--dropout_keep_prob', type=float, default=1.0, help='Dropout keep probability')

    parser.add_argument('--train_steps', type=int, default=1e-6, help='Number of training steps') # default 1e6
    parser.add_argument('--max_norm', type=float, default=5.0, help='--')

    # Misc params
    parser.add_argument('--summary_path', type=str, default="./summaries/", help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=50, help='How often to print training progress')
    parser.add_argument('--sample_every', type=int, default=1000, help='How often to sample from the model')

    parser.add_argument('--device', type=str, default="cpu", help="Training device 'cpu' or 'cuda:0'")

    # own params
    parser.add_argument('--epochs', type=int, default=30, help="Amount of epochs to train")

    config = parser.parse_args()

    # Train the model
    train(config)
