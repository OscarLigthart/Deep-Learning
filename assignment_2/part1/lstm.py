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

import torch
import torch.nn as nn
import numpy as np

################################################################################

class LSTM(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, batch_size, device='cpu'):
        super(LSTM, self).__init__()
        # Initialization here ...

        mean = 0
        std_dev = 0.001

        self.num_hidden = num_hidden
        self.batch_size = batch_size
        self.seq_length = seq_length

        # g
        self.W_gx = torch.nn.Parameter(torch.Tensor(np.random.normal(mean, std_dev, (num_hidden, input_dim)))).to(device)
        self.bias_g = torch.nn.Parameter(torch.Tensor(np.zeros(batch_size))).to(device)

        # h
        self.W_gh = torch.nn.Parameter(torch.Tensor(np.random.normal(mean, std_dev, (num_hidden, num_hidden)))).to(device)
        self.bias_h = torch.nn.Parameter(torch.Tensor(np.zeros(batch_size))).to(device)


        # i
        self.W_ix = torch.nn.Parameter(torch.Tensor(np.random.normal(mean, std_dev, (num_hidden, input_dim)))).to(device)
        self.W_ih = torch.nn.Parameter(torch.Tensor(np.random.normal(mean, std_dev, (num_hidden, num_hidden)))).to(
            device)
        self.bias_i = torch.nn.Parameter(torch.Tensor(np.zeros(batch_size))).to(device)

        # f
        self.W_fx = torch.nn.Parameter(torch.Tensor(np.random.normal(mean, std_dev, (num_hidden, input_dim)))).to(
            device)
        self.W_fh = torch.nn.Parameter(torch.Tensor(np.random.normal(mean, std_dev, (num_hidden, num_hidden)))).to(
            device)
        self.bias_f = torch.nn.Parameter(torch.Tensor(np.zeros(batch_size))).to(device)

        # o
        self.W_ox = torch.nn.Parameter(torch.Tensor(np.random.normal(mean, std_dev, (num_hidden, input_dim)))).to(
            device)
        self.W_oh = torch.nn.Parameter(torch.Tensor(np.random.normal(mean, std_dev, (num_hidden, num_hidden)))).to(
            device)
        self.bias_o = torch.nn.Parameter(torch.Tensor(np.zeros(batch_size))).to(device)

        #p
        self.W_ph = torch.nn.Parameter(torch.Tensor(np.random.normal(mean, std_dev, (num_classes, num_hidden)))).to(
            device)
        self.bias_p = torch.nn.Parameter(torch.Tensor(np.zeros(batch_size))).to(device)


        # todo kijk of die functies in 1 kunnen
        self.tanh = torch.nn.Tanh()
        self.sigmoid = torch.nn.Sigmoid()
        self.sigmoid2 = torch.nn.Sigmoid()
        self.sigmoid3 = torch.nn.Sigmoid()

        self.tanh2 = torch.nn.Tanh()
        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        # Implementation here ...
        h = torch.Tensor(np.zeros((self.num_hidden, self.batch_size)))
        c = torch.Tensor(np.zeros((self.num_hidden, self.batch_size)))

        for j in range(self.seq_length):

            g = self.tanh(torch.mm(self.W_gx, x[:, j].view(1, -1)) + torch.mm(self.W_gh, h) + self.bias_g)

            i = self.sigmoid(torch.mm(self.W_ix, x[:, j].view(1, -1)) + torch.mm(self.W_ih, h) + self.bias_i)

            f = self.sigmoid2(torch.mm(self.W_fx, x[:, j].view(1, -1)) + torch.mm(self.W_fh, h) + self.bias_f)

            o = self.sigmoid3(torch.mm(self.W_ox, x[:, j].view(1, -1)) + torch.mm(self.W_oh, h) + self.bias_o)

            c = g * i + c * f

            h = self.tanh2(c) * o

        p = torch.mm(self.W_ph, h) + self.bias_p

        p = torch.t(p)

        return p