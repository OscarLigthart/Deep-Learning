import argparse

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision.utils import make_grid
import numpy as np

from datasets.bmnist import bmnist


class Encoder(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()

        self.fc1 = nn.Linear(784, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, z_dim)
        self.fc22 = nn.Linear(hidden_dim, z_dim)

    def forward(self, input):
        """
        Perform forward pass of encoder.

        Returns mean and std with shape [batch_size, z_dim]. Make sure
        that any constraints are enforced.
        """

        hidden = torch.relu(self.fc1(input.view(-1,784)))
        mean = self.fc21(hidden)
        logvar = self.fc22(hidden)

        return mean, logvar


class Decoder(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()

        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 784)


    def forward(self, input):
        """
        Perform forward pass of encoder.

        Returns mean with shape [batch_size, 784].
        """

        hidden = torch.relu(self.fc1(input))
        mean = torch.sigmoid(self.fc2(hidden))

        return mean


class VAE(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()

        self.z_dim = z_dim
        self.encoder = Encoder(hidden_dim, z_dim)
        self.decoder = Decoder(hidden_dim, z_dim)

    def forward(self, input):
        """
        Given input, perform an encoding and decoding step and return the
        negative average elbo for the given batch.
        """

        # encode
        mean, logvar = self.encoder(input)

        # reparametrize
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        #z = eps.mul(std).add_(mean)

        z = mean + eps.mul(std)

        # decode
        recon = self.decoder(z)

        recon_loss = F.binary_cross_entropy(recon, input.view(-1, 784), reduction='sum')

        reg_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())

        average_negative_elbo = recon_loss + reg_loss

        return average_negative_elbo

    def sample(self, n_samples):
        """
        Sample n_samples from the model. Return both the sampled images
        (from bernoulli) and the means for these bernoullis (as these are
        used to plot the data manifold).
        """

        # sample from multivariate gaussian from z dimension (self.z_dim)
        z = torch.randn(n_samples, self.z_dim)

        # run through decoder
        im_means = self.decoder(z)

        # sample
        sampled_ims = torch.bernoulli(im_means)

        return sampled_ims, im_means


def epoch_iter(model, data, optimizer):
    """
    Perform a single epoch for either the training or validation.
    use model.training to determine if in 'training mode' or not.

    Returns the average elbo for the complete epoch.
    """
    average_epoch_elbo = None

    total_loss = 0
    for i, batch in enumerate(data):

        optimizer.zero_grad()
        loss = model(batch)

        if model.training:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() / len(batch)

    average_epoch_elbo = total_loss / len(data)

    return average_epoch_elbo


def run_epoch(model, data, optimizer):
    """
    Run a train and validation epoch and return average elbo for each.
    """
    traindata, valdata = data

    model.train()
    train_elbo = epoch_iter(model, traindata, optimizer)

    model.eval()
    val_elbo = epoch_iter(model, valdata, optimizer)

    return train_elbo, val_elbo


def save_elbo_plot(train_curve, val_curve, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(train_curve, label='train elbo')
    plt.plot(val_curve, label='validation elbo')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('ELBO')
    plt.tight_layout()
    plt.savefig(filename)


def main():
    data = bmnist()[:2]  # ignore test split
    model = VAE(z_dim=ARGS.zdim)
    optimizer = torch.optim.Adam(model.parameters())
    sample_size = 10

    train_curve, val_curve = [], []
    for epoch in range(ARGS.epochs):
        elbos = run_epoch(model, data, optimizer)
        train_elbo, val_elbo = elbos
        train_curve.append(train_elbo)
        val_curve.append(val_elbo)
        print(f"[Epoch {epoch}] train elbo: {train_elbo} val_elbo: {val_elbo}")

        # ----------------------------------------------------------    ----------
        #  Add functionality to plot samples from model during training.
        #  You can use the make_grid functionality that is already imported.
        # --------------------------------------------------------------------
        if epoch % 20 == 0:
            plt.figure()
            sampled_ims, im_means = model.sample(sample_size)
            print('Making grid')
            grid = make_grid(im_means.view(-1,1,28,28), nrow=5)

            npimg = grid.detach().numpy()

            plt.imshow(np.transpose(npimg, (1, 2, 0)))
            plt.show()
    # --------------------------------------------------------------------
    #  Add functionality to plot plot the learned data manifold after
    #  if required (i.e., if zdim == 2). You can use the make_grid
    #  functionality that is already imported.
    # --------------------------------------------------------------------
    # plot hier waar in 2d space de datapunten liggen.


    save_elbo_plot(train_curve, val_curve, 'elbo.pdf')

    plt.show()
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=40, type=int,
                        help='max number of epochs')
    parser.add_argument('--zdim', default=2, type=int,
                        help='dimensionality of latent space')

    ARGS = parser.parse_args()

    main()