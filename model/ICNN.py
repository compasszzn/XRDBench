import torch
import torch.nn as nn
import torch.nn.functional as F
## Rapid Identification of X-ray Diffraction Patterns Based on Very Limited Data by interpretable Convolutional Neural Networks.
##MOF
class Model(nn.Module):
    def __init__(self,args):
        super(Model, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.2),
            
            nn.Conv1d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.2),
            
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.2),
            
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.2)
        )
        self.flatten = nn.Flatten()
        if args.task=='spg':
            self.fc_layers = nn.Sequential(
                nn.Linear(8512, 120),
                nn.ReLU(),
                nn.Linear(120, 84),
                nn.ReLU(),
                nn.Linear(84, 230),
                nn.Softmax(dim=1)
            )
        elif args.task=='crysystem':
            self.fc_layers = nn.Sequential(
                nn.Linear(8512, 120),
                nn.ReLU(),
                nn.Linear(120, 84),
                nn.ReLU(),
                nn.Linear(84, 7),
                nn.Softmax(dim=1)
            )

    def forward(self, x):
        x = F.interpolate(x, size=2251,mode='linear', align_corners=False)
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.fc_layers(x)
        x = F.softmax(x, dim=1)
        return x