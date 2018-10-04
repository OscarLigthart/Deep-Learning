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

class VanillaRNN(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, batch_size, device='cpu'):
        super(VanillaRNN, self).__init__()
        # Initialization here ...

        # input_dim = batch_size x 1
        # num_hidden = 128
        # num_classes = 10
        # batch_size = 128

        mean = 0
        std_dev = 0.001

        self.num_hidden = num_hidden
        self.batch_size = batch_size
        self.seq_length = seq_length

        self.W_hx = torch.nn.Parameter(torch.Tensor(np.random.normal(mean, std_dev, (num_hidden, input_dim)))).to(device)

        self.W_hh = torch.nn.Parameter(torch.Tensor(np.random.normal(mean, std_dev, (num_hidden, num_hidden)))).to(device)

        self.W_ph = torch.nn.Parameter(torch.Tensor(np.random.normal(mean, std_dev, (num_classes, num_hidden)))).to(device)

        self.bias_h = torch.nn.Parameter(torch.Tensor(np.zeros((num_hidden, 1)))).to(device)

        self.bias_p = torch.nn.Parameter(torch.Tensor(np.zeros((num_classes, 1)))).to(device)

        self.tanh = torch.nn.Tanh()



    def forward(self, x):
        # Implementation here ...


        h = torch.Tensor(np.zeros((self.num_hidden, self.batch_size)))

        for i in range(self.seq_length):

            h = self.tanh(torch.mm(self.W_hx, x[:,i].view(1,-1)) + torch.mm(self.W_hh, h) + self.bias_h)


        p = torch.mm(self.W_ph, h) + self.bias_p

        p = torch.t(p)

        return p