import numpy as np
import pickle

import matplotlib.pyplot as plt


# load saved data

# loop for every sequence length
for i in range(6):

    # plot saved data
    filename = "Result_LSTM_inputlen_" + str(5+i) + '.p'
    LSTM_data = pickle.load(open(filename, 'rb'))
    LSTM_loss = LSTM_data[0]
    LSTM_acc = LSTM_data[1]

    filename = "Result_RNN_inputlen_" + str(5+i) + '.p'
    RNN_data = pickle.load(open(filename, 'rb'))
    RNN_loss = RNN_data[0]
    RNN_acc = RNN_data[1]

    # data was saved every 100 steps, so create ticks for x-axis
    x = np.linspace(0, 10000, 1001, dtype=np.int64)


    # plot
    fig, ax1 = plt.subplots()

    ax1.plot(x, LSTM_loss, label="LSTM loss", color='firebrick')
    ax1.plot(x, RNN_loss, label="RNN loss", color='darksalmon')
    ax1.set_title("Accuracy and Loss curves for input length = " + str(5+i))
    ax1.set_ylabel("Loss")
    ax1.set_xlabel("Evaluation")
    ax1.tick_params(axis='y', labelcolor='red')
    ax1.legend(loc='upper left', bbox_to_anchor=(0, -0.1))

    ax2 = ax1.twinx()

    ax2.plot(x, LSTM_acc, label='LSTM accuracy', color='royalblue')
    ax2.plot(x, RNN_acc, label='RNN accuracy', color='lightskyblue')
    ax2.tick_params(axis='y', labelcolor='blue')
    ax2.set_ylabel("Accuracy")

    fig.tight_layout()
    ax2.legend(loc='upper right', bbox_to_anchor=(1, -0.1))
    plt.show()