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

import torch.nn.functional as F

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import TextDataset
from model import TextGenerationModel

import pickle
import os

################################################################################



def train(config):

    # Initialize the device which to run the model on
    device = torch.device(config.device)

    # Initialize the dataset and data loader (note the +1)
    dataset = TextDataset(config.txt_file, config.seq_length)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1, drop_last=True)
    vocab_size = dataset.vocab_size


    # Initialize the model that we are going to use
    model = TextGenerationModel(config.batch_size, config.seq_length, vocab_size, config.lstm_num_hidden, config.lstm_num_layers, config.device)
    model = model.to(device)

    print(model)

    # Setup the loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()

    # if pickle file is available, load steps and use index -1 to get last step + get lists of values, to continue training
    # where we left off
    if os.path.isfile("steps.p"):
        print('Pre-trained model available...')
        print('Resuming training...')

        # load lists
        step_intervals = pickle.load(open("steps.p", "rb"))
        all_sentences = pickle.load(open("sentences.p", "rb"))
        accuracy_list = pickle.load(open("accuracies.p", "rb"))
        loss_list = pickle.load(open("loss.p", "rb"))
        model_info = pickle.load(open("model_info.p", "rb"))

        # start where we left off
        all_steps = step_intervals[-1]

        # load model
        Modelname = 'TrainIntervalModel' + model_info[0] + 'acc:' + model_info[1] + '.pt'
        model = torch.load(Modelname)
        model = model.to(device)

    # otherwise start training from a clean slate
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
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.learning_rate_step, gamma=config.learning_rate_decay)

    # since the nested for loop stops looping after a complete iteration through the data_loader, add for loop for epochs
    for epoch in range(config.epochs):
        print(model)
        for step, (batch_inputs, batch_targets) in enumerate(data_loader):

            # Only for time measurement of step through network
            t1 = time.time()

            # apply scheduler
            scheduler.step()

            # create 2D tensor instead of list of 1D tensors
            #batch_inputs = torch.stack(batch_inputs)
            batch_inputs = batch_inputs.to(device)

            h,c = model.init_hidden()
            out, (h,c) = model(batch_inputs, h, c)

            # transpose to match cross entropy input dimensions
            out.transpose_(1, 2)

            batch_targets = batch_targets.to(device)

            #######################################################
            # Add more code here ...
            #######################################################

            loss = criterion(out, batch_targets)
            
            max = torch.argmax(out, dim=1)
            correct = (max == batch_targets)
            accuracy = torch.sum(correct).item()/correct.size()[0]/correct.size()[1]

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

                # do not keep track of gradients during model evaluation
                with torch.no_grad():

                    # create random character to start sentence with
                    random_input = torch.randint(0, vocab_size, (config.batch_size,), dtype=torch.long).view(-1,1)
                    x_input = random_input.to(device)

                    # initialize hidden state and cell state
                    h,c = model.init_hidden()
                    h = h.to(device)
                    c = c.to(device)

                    sentences = x_input

                    # loop through sequence length to set generated output as input for next sequence
                    for i in range(config.seq_length):

                        # get randomly generated sentence
                        out, (h,c) = model(x_input, h, c)

                        ####################
                        # Temperature here #
                        ####################

                        # check whether user wants to apply temperature sampling
                        if config.temperature:

                            # apply temperature sampling
                            out = out / config.tempvalue
                            out = F.softmax(out, dim=2)

                            # create a torch distribution of the calculated softmax probabilities and sample from that distribution
                            distribution = torch.distributions.categorical.Categorical(out.view(config.batch_size, vocab_size))
                            out = distribution.sample().view(-1, 1)

                        # check whether user wants to apply greedy sampling
                        else:
                            # load new datapoint by taking the predicted previous letter using greedy approach
                            out = torch.argmax(out, dim=2)

                        # append generated character to total sentence
                        sentences = torch.cat((sentences, out), 1)
                        x_input = out


                    # pick a random sentence (from the batch of created sentences)
                    index = np.random.randint(0, config.batch_size, 1)
                    sentence = sentences[index, :]

                    # squeeze sentence into 1D
                    sentence = sentence.view(-1).cpu()

                    # print sentence
                    print(dataset.convert_to_string(sentence.data.numpy()))

                    # save sentence
                    all_sentences.append(sentence.data.numpy())

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

            # counter of total amounts of steps (keep track over multiple training sessions)
            all_steps += 1

        if config.savefiles:
            # pickle sentences and steps
            pickle.dump(all_sentences, open('sentences.p', 'wb'))
            pickle.dump(step_intervals, open('steps.p', 'wb'))

            # pickle accuracy and loss
            pickle.dump(accuracy_list, open('accuracies.p', 'wb'))
            pickle.dump(loss_list, open('loss.p', 'wb'))



            # save model

            Modelname = 'TrainIntervalModel' + str(epoch) + 'acc:' + str(accuracy) + '.pt'
            torch.save(model, Modelname)

            model_info = [str(epoch), str(accuracy)]
            pickle.dump(model_info, open('model_info.p', 'wb'))

    print('Done training.')


 ################################################################################
 ################################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--txt_file', type=str, default='assets/book_EN_grimms_fairy_tails.txt', help="Path to a .txt file to train on")
    parser.add_argument('--seq_length', type=int, default=50, help='Length of an input sequence')
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
    parser.add_argument('--sample_every', type=int, default=100, help='How often to sample from the model')

    parser.add_argument('--device', type=str, default="cpu", help="Training device 'cpu' or 'cuda:0'")

    # own params
    parser.add_argument('--epochs', type=int, default=50, help="Amount of epochs to train")
    parser.add_argument('--temperature', type=bool, default=False, help="Choose whether temperature sampling should be applied")
    parser.add_argument('--tempvalue', type=float, default=1, help="Choose temperature value")
    parser.add_argument('--savefiles', type=bool, default=False, help="Choose whether to save variables and model")

    config = parser.parse_args()

    # Train the model
    train(config)
