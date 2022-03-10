# Implementing, and playing with, the GRU model
# from here: https://medium.com/swlh/stock-price-prediction-with-pytorch-37f52ae84632

import numpy as np
import torch as t
import torch.nn as nn


def split_data(stock, lookback):
    data_raw = stock.to_numpy()

    # create all possible sequences of length seq_len
    data = np.array(
        [data_raw[i: i + lookback]
         for i in range(data_raw.shape[0] - lookback)]
    )

    test_set_size = int(np.round(0.2 * data.shape[0]))
    train_set_size = data.shape[0] - (test_set_size)

    x_train = data[:train_set_size, :-1, :]
    y_train = data[:train_set_size, -1, :]

    x_test = data[train_set_size:, :-1, :]
    y_test = data[train_set_size:, -1, :]

    return (x_train, y_train, x_test, y_test)


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim,
                            num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = t.zeros(self.num_layers, x.size(
            0), self.hidden_dim).requires_grad_()
        c0 = t.zeros(self.num_layers, x.size(
            0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out


class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(GRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = t.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn) = self.gru(x, (h0.detach()))
        out = self.fc(out[:, -1, :])
        return out
