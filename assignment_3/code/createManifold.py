import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from scipy.stats import norm
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

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


##### THIS CODE CAN BE USED TO QUICKLY CREATE THE MANIFOLD #######

# create coordinates on unit square
nx, ny = (3, 2)
x = np.linspace(0.01, 0.99, 20)
y = np.linspace(0.01, 0.99, 20)
xv, yv = np.meshgrid(x, y)
unit_square = np.stack((np.ravel(xv), np.ravel(yv)), axis=-1)

# use iCDF to get gaussian values
z = norm.ppf((unit_square))
z = torch.Tensor(z)

# load model and generate samples
model = torch.load('VAE.pt')
imgs = model.decoder(z)

# plot manifold
plt.figure()
grid = make_grid(imgs.view(-1,1,28,28), nrow=20)
npimg = grid.detach().numpy()
plt.imshow(np.transpose(npimg, (1, 2, 0)))
titlename = 'Data manifold'
plt.show()
plt.title(titlename)