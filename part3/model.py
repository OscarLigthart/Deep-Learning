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

import torch.nn as nn
import torch


class TextGenerationModel(nn.Module):

    def __init__(self, batch_size, seq_length, vocabulary_size,
                 lstm_num_hidden=256, lstm_num_layers=2, device='cuda:0'):

        super(TextGenerationModel, self).__init__()
        # Initialization here...

        #input = vocabulary size?
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.lstm_num_hidden = lstm_num_hidden
        self.lstm_num_layers = lstm_num_layers
        self.device = device


        # linear nodig?

        self.lstm = nn.LSTM(1, lstm_num_hidden, lstm_num_layers)
        self.linear = nn.Linear(lstm_num_hidden, vocabulary_size)


    def forward(self, x):
        # Implementation here...

        h = torch.zeros(self.lstm_num_layers, self.batch_size, self.lstm_num_hidden).to(self.device)
        c = torch.zeros(self.lstm_num_layers, self.batch_size, self.lstm_num_hidden).to(self.device)

        out = torch.FloatTensor().to(self.device)

        # if input is a whole sentence, use following code (length will thus be longer than 1)
        if len(x) > 1:
            # run data through LSTM
            x_input = x.view(self.seq_length, -1, 1).float().to(self.device)
            out, (h, c) = self.lstm(x_input, (h, c))
            out = self.linear(out)

        # load randomly generated letter, to generate sentences based on that letter (length of input will then be 1)
        elif len(x) == 1:

            # get batch of randomly generated letters
            x_input = x.view(1, -1, 1).float().to(self.device)

            # loop through sequence length to set generated output as input for next sequence
            for i in range(self.seq_length):
                # if input is one letter (to generate a sentence), use following code
                out_i, (h, c) = self.lstm(x_input, (h, c))
                out_i = self.linear(out_i)
                out = torch.cat((out, out_i), 0)

                # load new datapoint by taking the predicted previous letter
                out_i = torch.argmax(out_i, dim=2)
                x_input = out_i.view(1, -1, 1).float().to(self.device)

        return out

