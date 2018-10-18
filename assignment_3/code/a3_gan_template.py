import argparse
import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets
import pickle

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

def loss_function(real,fake):

    loss = -0.5 * torch.log(real) - 0.5 * torch.log(1-fake)

    loss = torch.mean(loss)

    return loss

def train(dataloader, discriminator, generator, optimizer_G, optimizer_D, device):

    BCE = torch.nn.BCELoss()

    # keep track of loss
    loss_list_d = []
    loss_list_g = []

    for epoch in range(args.n_epochs):
        for i, (imgs, _) in enumerate(dataloader):

            imgs = imgs.to(device)

            ##### BCE #####
            if args.BCE:

                # create real labels
                real = np.random.uniform(0.7, 1.2, size=(imgs.size(0),1))
                real = torch.Tensor(real)
                real_gen = torch.ones(imgs.size(0),1)
                real = real.to(device)

                # create fake labels
                fake = np.random.uniform(0, 0.3, size=(imgs.size(0), 1))
                fake = torch.Tensor(fake)
                fake = fake.to(device)

                # Train Generator
                # ---------------

                optimizer_G.zero_grad()

                z = torch.randn(imgs.shape[0], args.latent_dim)


                z = z.to(device)

                gen = generator(z)
                gen_img = gen.view(-1, 1, 28, 28)

                loss_g = BCE(discriminator(gen), real_gen)

                loss_g.backward()
                optimizer_G.step()

                # Train Discriminator
                # -------------------

                optimizer_D.zero_grad()

                # get discriminator loss
                img_vec = imgs.view(imgs.shape[0], -1)

                # get predictions on fake and real images
                real_img_pred = discriminator(img_vec)
                fake_img_pred = discriminator(gen.detach())

                # update on only fake or only real samples at random
                if torch.rand(1).item() > 0.5:
                    loss_d = BCE(real_img_pred, real)
                else:
                    loss_d = BCE(fake_img_pred, fake)

                loss_d.backward()
                optimizer_D.step()


            else:
                ##### MinMax #####

                # sample from latent space
                z = torch.randn(imgs.shape[0], args.latent_dim)
                z = z.to(device)

                #
                gen = generator(z)
                gen_img = gen.view(-1, 1, 28, 28)

                # get discriminator loss
                img_vec = imgs.view(imgs.shape[0], -1)

                real_img_pred = discriminator(img_vec)
                fake_img_pred = discriminator(gen)

                loss_d = loss_function(real_img_pred, fake_img_pred)
                loss_g = -loss_d

                optimizer_D.zero_grad()
                loss_d.backward(retain_graph=True)
                optimizer_D.step()

                optimizer_G.zero_grad()
                loss_g.backward()
                optimizer_G.step()


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
            # -------------

            if i % 100 == 0:
                print(f"[Epoch {epoch}] Batch: {i}/{len(dataloader)} Generator Loss: {loss_g} Discriminator Loss: {loss_d}")

                # keep track of generator and discriminator loss
                loss_d = loss_d.cpu()
                loss_g = loss_g.cpu()
                loss_list_d.append(loss_d.data.numpy())
                loss_list_g.append(loss_g.data.numpy())


            # Save Model during training
            # --------------------------
            if epoch % args.save_model == 0:
                modelname = 'mnist_generator_epoch_' + str(epoch) + '.pt'
                torch.save(generator.state_dict(), modelname)

    # Save loss curves
    pickle.dump(loss_list_g, open('GeneratorLoss.p', 'wb'))
    pickle.dump(loss_list_d, open('DiscriminatorLoss.p', 'wb'))






def main():
    # Create output image directory
    os.makedirs('images', exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
    generator = generator.to(device)
    discriminator = Discriminator()
    discriminator = discriminator.to(device)

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr)

    # Start training
    train(dataloader, discriminator, generator, optimizer_G, optimizer_D, device)

    # You can save your generator here to re-use it to generate images for your
    # report, e.g.:
    torch.save(generator.state_dict(), "final_mnist_generator.pt")


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
    parser.add_argument('--BCE', type=bool, default=True,
                        help='Choose to train the model using BCE (True) or ZeroSum Minimax game (False)')
    args = parser.parse_args()

    main()
