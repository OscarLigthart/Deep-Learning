import pickle
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from dataset import TextDataset
from model import TextGenerationModel
import matplotlib.pyplot as plt
import numpy as np

sentences = pickle.load(open('results/sentences.p', 'rb'))

# Initialize the dataset and data loader (note the +1)
# dataset = TextDataset('~/home/oscar/Documents/AI/Deep-Learning/assignment_2/part3/assets/book_EN_democracy_in_the_US.txt', 30)
dataset = TextDataset('assets/book_EN_grimms_fairy_tails.txt', 30)
data_loader = DataLoader(dataset, 64, num_workers=1, drop_last=True)
vocab_size = dataset.vocab_size
show = [0, 2, 100, 2001, 4001]

###################
# Hyperparameters #
###################

temperature = True
tempvalue = [0.5,1,2]
batch_size = 64

# show five sentences created during training
#for i in show:
#    print(dataset.convert_to_string(sentences[i]))


# load 3 models

model1 = torch.load('results/TrainIntervalModel0acc:0.6307291666666667.pt', map_location={'cuda:0': 'cpu'})

model2 = torch.load('results/TrainIntervalModel20acc:0.7119791666666667.pt', map_location={'cuda:0': 'cpu'})

model3 = torch.load('results/TrainIntervalModel48acc:0.7526041666666666.pt', map_location={'cuda:0': 'cpu'})

models = [model1, model2, model3]
device = torch.device('cpu')


for h, model in enumerate(models):
    print("RESULTS FOR MODEL " + str(h))
    print("------------------------------------")
    for i in range(4):

        # create a sentence without using temperature sampling in the first loop
        if i == 0:
            temperature = False
            print('Greedy sampling:')
        else:
            temperature = True
            print('Temperature sampling with a temperature of ' + str(tempvalue[i-1]) + ':')

        with torch.no_grad():
            random_input = torch.randint(0, vocab_size, (batch_size,), dtype=torch.long).view(-1, 1)
            x_input = random_input.to(device)

            h, c = model.init_hidden()

            h = h.to(device)
            c = c.to(device)

            sentences = x_input


            # loop through sequence length to set generated output as input for next sequence
            for j in range(30):

                # get randomly generated sentence
                out, (h, c) = model(x_input, h, c)

                ####################
                # Temperature here #
                ####################

                if temperature:
                    out = out / tempvalue[i-1]
                    out = F.softmax(out, dim=2)

                    m = torch.distributions.categorical.Categorical(out.view(batch_size, vocab_size))
                    out = m.sample().view(-1,1)

                else:
                    # load new datapoint by taking the predicted previous letter using greedy approach
                    out = torch.argmax(out, dim=2)


                sentences = torch.cat((sentences, out), 1)
                x_input = out

            # pick a random sentence (from the batch of created sentences)
            index = np.random.randint(0, batch_size, 1)
            sentence = sentences[index, :]

            # squeeze sentence into 1D
            sentence = sentence.view(-1).cpu()

            # print sentence
            print(dataset.convert_to_string(sentence.data.numpy()))
            print()