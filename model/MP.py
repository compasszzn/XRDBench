import torch
import torch.nn as nn
import torch.nn.functional as F
## Deep Learning Models to Identify Common Phases across Material Systems from X‑ray Diffraction
class Model(nn.Module):
    def __init__(self,args):
        super(Model, self).__init__()
        self.conv1 = nn.Conv1d(1, 27, kernel_size=11, stride=1)
        self.conv2 = nn.Conv1d(27, 27, kernel_size=11, stride=1)
        self.conv3 = nn.Conv1d(27, 27, kernel_size=11, stride=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.21)

        self.fcl1 = nn.Linear(11556,2000)
        if args.task=='spg':
            self.fcl2 = nn.Linear(2000,230)
        elif args.task=='crysystem':
            self.fcl2 = nn.Linear(2000,7)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.dropout(self.pool(F.relu(self.conv1(x))))
        x = self.dropout(self.pool(F.relu(self.conv2(x))))
        x = self.dropout(self.pool(F.relu(self.conv3(x))))
        x = self.flatten(x)
        x = self.dropout(F.relu(self.fcl1(x)))
        x = self.fcl2(x)
        return x
