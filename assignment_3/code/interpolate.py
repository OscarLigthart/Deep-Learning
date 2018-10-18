import argparse
import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets
import pickle
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

import numpy as np
from torch import Tensor
from torch.autograd import Variable


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # Construct generator. You are free to experiment with your model,
        # but the following is a good start:
        #   Linear args.latent_dim -> 128
        #   LeakyReLU(0.2)
        #   Linear 128 -> 256
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 256 -> 512
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 512 -> 1024
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 1024 -> 768
        #   Output non-linearity

        # create list of modules
        self.modules = []

        self.modules.append(nn.Linear(args.latent_dim, 128))
        self.modules.append(nn.LeakyReLU(0.2))

        dims = [(128,256),(256,512),(512,1024)]

        for dim in dims:
            self.modules.append(nn.Linear(dim[0], dim[1]))
            self.modules.append(nn.BatchNorm1d(dim[1]))
            self.modules.append(nn.LeakyReLU(0.2))

        self.modules.append(nn.Linear(1024,784))
        self.modules.append(nn.Tanh())

        # turn list into model
        self.model = nn.Sequential(*self.modules)


    def forward(self, z):
        # Generate images from z

        out = self.model(z)

        return out



class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # Construct distriminator. You are free to experiment with your model,
        # but the following is a good start:
        #   Linear 784 -> 512
        #   LeakyReLU(0.2)
        #   Linear 512 -> 256
        #   LeakyReLU(0.2)
        #   Linear 256 -> 1
        #   Output non-linearity

        self.modules = []
        dims = [(784, 512), (512,256)]
        for dim in dims:
            self.modules.append(nn.Linear(dim[0], dim[1]))
            self.modules.append(nn.LeakyReLU(0.2))

        self.modules.append(nn.Linear(256,1))
        self.modules.append(nn.Sigmoid())

        self.model = nn.Sequential(*self.modules)

    def forward(self, img):

        # return discriminator score for img
        x = self.model(img)

        return x


def interpolate(generator, device):

    # take 2 image samples (make sure they are of different classes) and interpolate in latent space
    x = np.random.normal(size=(args.latent_dim))
    y = np.random.normal(size=(args.latent_dim))
    z = np.empty((9, args.latent_dim))
    for i, a in enumerate(np.linspace(0.0, 1.0, 9)):
        z[i] = a * x + (1.0 - a) * y

    z = torch.Tensor(z)
    z = z.to(device)

    # run through generator
    generator.eval()
    gen = generator(z)
    gen_img = gen.view(-1, 1, 28, 28)

    # show interpolation
    plt.figure()
    grid = make_grid(gen_img.view(-1, 1, 28, 28), nrow=9)
    npimg = grid.detach().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    titlename = 'Interpolation'
    plt.show()
    plt.title(titlename)




def main():
    # Create output image directory
    os.makedirs('images', exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initialize models and optimizers
    gen_model = Generator()
    gen_model.load_state_dict(torch.load('ResultGANZeroSum/final_mnist_generator.pt'))

    # interpolate
    interpolate(gen_model, device)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=200,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='learning rate')
    parser.add_argument('--latent_dim', type=int, default=100,
                        help='dimensionality of the latent space')
    parser.add_argument('--save_interval', type=int, default=500,
                        help='save every SAVE_INTERVAL iterations')
    parser.add_argument('--save_model', type=int, default=100)
    args = parser.parse_args()

    main()
