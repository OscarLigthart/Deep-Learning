import argparse
import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets

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

        self.modules = []

        # first linear

        self.modules.append(nn.Linear(args.latent_dim, 128))
        self.modules.append(nn.LeakyReLU(0.2))

        dims = [(128,256),(256,512),(512,1024)]

        for dim in dims:
            self.modules.append(nn.Linear(dim[0], dim[1]))
            self.modules.append(nn.BatchNorm1d(dim[1]))
            self.modules.append(nn.LeakyReLU(0.2))

        self.modules.append(nn.Linear(1024,784))
        self.modules.append(nn.Tanh())

        # hier evt sequential
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


def train(dataloader, discriminator, generator, optimizer_G, optimizer_D):

    BCE = torch.nn.BCELoss()

    for epoch in range(args.n_epochs):
        for i, (imgs, _) in enumerate(dataloader):

            #imgs.cuda()

            valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
            real = torch.ones(imgs.size(0),1)
            #fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)
            fake = torch.zeros(imgs.size(0),1)

            # Train Generator
            # ---------------

            optimizer_G.zero_grad()

            z = torch.randn(imgs.shape[0], args.latent_dim)

            #z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], args.latent_dim))))

            gen = generator(z)
            gen_img = gen.view(-1, 1, 28, 28)

            loss_g = BCE(discriminator(gen), real)

            loss_g.backward()
            optimizer_G.step()

            # Train Discriminator
            # -------------------
            optimizer_D.zero_grad()

            # get real loss
            img_vec = imgs.view(imgs.shape[0], -1)
            loss_d_real = BCE(discriminator(img_vec), real)

            # get fake loss
            loss_d_fake = BCE(discriminator(gen.detach()), fake)

            # average
            loss_d = (loss_d_real + loss_d_fake)/2

            loss_d.backward()
            optimizer_D.step()

            # Save Images
            # -----------
            batches_done = epoch * len(dataloader) + i
            if batches_done % args.save_interval == 0:
                # You can use the function save_image(Tensor (shape Bx1x28x28),
                # filename, number of rows, normalize) to save the generated
                # images, e.g.:
                save_image(gen_img[:25],
                            'images/{}.png'.format(batches_done),
                            nrow=5, normalize=True)

            # Show Progress
            # ------------

            if i % 100 == 0:
                print(f"[Epoch {epoch}] Batch: {i}/{len(dataloader)} Generator Loss: {loss_g} Discriminator Loss: {loss_d}")



def main():
    # Create output image directory
    os.makedirs('images', exist_ok=True)

    # load data
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST('./data/mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))])),
        batch_size=args.batch_size, shuffle=True)

    # Initialize models and optimizers
    generator = Generator()
    discriminator = Discriminator()
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr)

    # Start training
    train(dataloader, discriminator, generator, optimizer_G, optimizer_D)

    # You can save your generator here to re-use it to generate images for your
    # report, e.g.:
    # torch.save(generator.state_dict(), "mnist_generator.pt")


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
    args = parser.parse_args()

    main()
