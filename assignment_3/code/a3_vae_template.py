import argparse

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision.utils import make_grid
import numpy as np
from scipy.stats import norm

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
        z = mean + eps.mul(std)

        # decode
        recon = self.decoder(z)

        # calculate reconstruction loss
        recon_loss = F.binary_cross_entropy(recon, input.view(-1, 784), reduction='sum')

        # calculte regularization loss
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

    total_loss = 0
    for i, batch in enumerate(data):

        # run batch through model
        optimizer.zero_grad()
        loss = model(batch)

        # backpropagate through model if training
        if model.training:
            loss.backward()
            optimizer.step()

        # normalize loss
        total_loss += loss.item() / len(batch)

    # get average elbo per epoch
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
        if epoch % int(ARGS.epochs/2) == 0 or epoch == (ARGS.epochs - 1):
            sampled_ims, im_means = model.sample(sample_size)

            # plot means
            plt.figure()
            grid = make_grid(im_means.view(-1,1,28,28), nrow=5)
            npimg = grid.detach().numpy()
            plt.imshow(np.transpose(npimg, (1, 2, 0)))
            titlename = 'Generated VAE images at epoch ' + str(epoch) + ' using the means'
            plt.title(titlename)

            # plot sampled images
            plt.figure()
            grid = make_grid(sampled_ims.view(-1, 1, 28, 28), nrow=5)
            npimg = grid.detach().numpy()
            plt.imshow(np.transpose(npimg, (1, 2, 0)))
            titlename = 'Generated VAE images at epoch ' + str(epoch) + ' using sampled images'
            plt.title(titlename)


    # --------------------------------------------------------------------
    #  Add functionality to plot plot the learned data manifold after
    #  if required (i.e., if zdim == 2). You can use the make_grid
    #  functionality that is already imported.
    # --------------------------------------------------------------------
    if ARGS.zdim == 2:

        # create unit square
        x = np.linspace(0.01, 0.99, 20)
        y = np.linspace(0.01, 0.99, 20)
        xv, yv = np.meshgrid(x, y)
        unit_square = np.stack((np.ravel(xv), np.ravel(yv)), axis=-1)

        # transform to gaussian using ppf
        z = norm.ppf((unit_square))
        z = torch.Tensor(z)

        # run through decoder
        imgs = model.decoder(z)

        # plot manifold
        plt.figure()
        grid = make_grid(imgs.view(-1, 1, 28, 28), nrow=20)
        npimg = grid.detach().numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        titlename = 'Data manifold'
        plt.title(titlename)

    # save elbo plot and show plots
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
