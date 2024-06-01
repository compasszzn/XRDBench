import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.hidden_size = 64
        self.num_layers = 4
        self.gru = nn.GRU(1, self.hidden_size, self.num_layers, batch_first=True, bidirectional=True)

        if args.task == 'spg':
            self.fc = nn.Linear(self.hidden_size*2, 230)
        elif args.task == 'crysystem':
            self.fc = nn.Linear(self.hidden_size*2, 7)

    def forward(self, x):

        x = x.squeeze(1).unsqueeze(-1)
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out
