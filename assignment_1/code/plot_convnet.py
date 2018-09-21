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

import pickle
import torch
from torch.autograd import Variable

import matplotlib.pyplot as plt

# load data
lists = pickle.load(open("lists.p", "rb"))


accuracy_train_list, accuracy_val_list, loss_list, loss_val_list = lists

# plot
fig, ax1 = plt.subplots()

ax1.plot(loss_list, label="Train loss", color='firebrick')
ax1.plot(loss_val_list, label="Validation loss", color='darksalmon')
ax1.set_title("Accuracy and Loss curves")
ax1.set_ylabel("Loss")
ax1.set_xlabel("Evaluation")
ax1.tick_params(axis='y', labelcolor='red')
ax1.legend(loc='upper left')

ax2 = ax1.twinx()

ax2.plot(accuracy_train_list, label='Train accuracy', color='royalblue')
ax2.plot(accuracy_val_list, label='Validation accuracy', color='lightskyblue')
ax2.tick_params(axis='y', labelcolor='blue')
ax2.set_ylabel("Accuracy")

fig.tight_layout()
ax2.legend(loc='upper right')
plt.show()
