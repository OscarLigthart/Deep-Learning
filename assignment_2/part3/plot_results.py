from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pickle
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from dataset import TextDataset
from model import TextGenerationModel
import matplotlib.pyplot as plt

# load saved data

losses = pickle.load(open('results/loss.p', 'rb'))

steps = pickle.load(open('results/steps.p', 'rb'))

accuracies = pickle.load(open('results/accuracies.p', 'rb'))

# plot saved data

# plot
fig, ax1 = plt.subplots()

skip = 10
ax1.plot(steps[::skip], losses[::skip], label="Loss", color='red')

ax1.set_title("Accuracy and Loss curves")
ax1.set_ylabel("Loss")
ax1.set_xlabel("Step")
ax1.tick_params(axis='y', labelcolor='red')
ax1.legend(loc='upper left')

ax2 = ax1.twinx()

ax2.plot(steps[::skip], accuracies[::skip], label='Accuracy', color='blue')
ax2.tick_params(axis='y', labelcolor='blue')
ax2.set_ylabel("Accuracy")

fig.tight_layout()
ax2.legend(loc='upper right')
plt.show()
