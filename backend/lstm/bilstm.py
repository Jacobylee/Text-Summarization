import torch.nn as nn
import torch
import torch.nn.functional as F

EMBED_DIM = 50
HIDDEN_DIM = 200
NUM_LAYERS = 1  # todo
DROPOUT = 0.2


class EncoderLSTM(nn.Module):
    def __init__(self, input_dim, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_dim, hidden_size)
        self.lstm_encoder = nn.LSTM(hidden_size, hidden_size, num_layers=NUM_LAYERS, batch_first=True, bidirectional=False)
        self.dropout = nn.Dropout(DROPOUT)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(NUM_LAYERS, 1, self.hidden_size),  # h0
                torch.randn(NUM_LAYERS, 1, self.hidden_size))  # c0

    def forward(self, x):
        embed = self.embedding(x.unsqueeze(0))
        embed = self.dropout(embed)
        outputs, (h_n, c_n) = self.lstm_encoder(embed, self.hidden)
        return outputs.squeeze(dim=0), (h_n, c_n)


class DecoderLSTM(nn.Module):
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.decoder = nn.LSTM(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, hidden):
        output = self.embedding(x).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.decoder(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size)
