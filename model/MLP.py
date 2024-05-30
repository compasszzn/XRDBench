import torch
import torch.nn as nn
import torch.nn.functional as F
## Rapid Identification of X-ray Diffraction Patterns Based on Very Limited Data by interpretable Convolutional Neural Networks.
##MOF
class Model(nn.Module):
    def __init__(self,args):
        super(Model, self).__init__()
        self.mlp_layers = nn.Sequential(
            nn.Linear(3501, 7000),
            nn.ReLU(),
            nn.Linear(7000, 7000),
            nn.ReLU(),
        )
        self.flatten = nn.Flatten()
        if args.task=='spg':
            self.fc_layers = nn.Sequential(
                nn.Linear(7000, 3500),
                nn.ReLU(),
                nn.Linear(3500, 230),
                nn.Softmax(dim=1)
            )
        elif args.task=='crysystem':
            self.fc_layers = nn.Sequential(
                nn.Linear(7000, 3500),
                nn.ReLU(),
                nn.Linear(3500, 7),
                nn.Softmax(dim=1)
            )

    def forward(self, x):
        x = self.mlp_layers(x)
        x = self.flatten(x)
        x = self.fc_layers(x)
        x = F.softmax(x, dim=1)
        return x