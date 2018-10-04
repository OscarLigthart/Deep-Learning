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
dataset = TextDataset('assets/book_EN_grimms_fairy_tails.txt', 30)
data_loader = DataLoader(dataset, 64, num_workers=1, drop_last=True)
vocab_size = dataset.vocab_size

###################
# Hyperparameters #
###################

temperature = True
tempvalue = 0.5
batch_size = 64
start_of_sentence = "My name is"

###################

# load a model
model = torch.load('results/TrainIntervalModel48acc:0.7526041666666666.pt', map_location={'cuda:0': 'cpu'})
device = torch.device('cpu')

with torch.no_grad():
    random_input = torch.randint(0, vocab_size, (batch_size,), dtype=torch.long).view(-1, 1)
    x_input = random_input.to(device)

    h, c = model.init_hidden()

    h = h.to(device)
    c = c.to(device)

    sent_int = dataset.convert_to_int(start_of_sentence)

    input = torch.LongTensor(np.repeat(np.expand_dims(np.array(sent_int), 0), batch_size, 0)).to(device)

    for i in range(len(start_of_sentence)-1):

        out, (h, c) = model(input[:,i].view(-1, 1), h, c)


    x_input = input[:,-1].view(-1,1)


    sentences = input

    # loop through sequence length to set generated output as input for next sequence
    for j in range(30):

        # get randomly generated sentence
        out, (h, c) = model(x_input, h, c)

        ####################
        # Temperature here #
        ####################

        if temperature:
            out = out / tempvalue
            out = F.softmax(out, dim=2)

            m = torch.distributions.categorical.Categorical(out.view(batch_size, vocab_size))
            out = m.sample().view(-1, 1)

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