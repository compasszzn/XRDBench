import torch
import torch.nn as nn
import torch.nn.functional as F

class FFN(nn.Module):
    def __init__(self, drop_rate=0.2, drop_rate_2=0.4):
        super(FFN, self).__init__()
        initializer = torch.nn.init.xavier_uniform_
        self.encoder = nn.Linear(3501,230)


    def forward(self, x):
        x = self.encoder(x)
        x = F.leaky_relu(x)
        x = x.squeeze(1)
        x = F.softmax(x, dim=1)
        return x
