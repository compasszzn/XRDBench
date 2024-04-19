import torch
import torch.nn as nn
import torch.nn.functional as F
## a deep-learning technique for phase identification in multiphase inorganic compounds using syhthetic xrd powder patterns
class Model(nn.Module):
    def __init__(self,args):
        super(Model, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=50, stride=2,padding=25)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=25, stride=3, padding=12)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=3, padding=1)
        self.fcl1 = nn.Linear(8064,2000)
        self.fcl2 = nn.Linear(2000,500)
        if args.task=='spg':
            self.fcl3 = nn.Linear(500,230)
        elif args.task=='crysystem':
            self.fcl3 = nn.Linear(500,7)
        
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = F.interpolate(x, size=4501,mode='linear', align_corners=False)
        x = self.pool1(F.leaky_relu(self.conv1(x)))
        x = self.pool2(F.leaky_relu(self.conv2(x)))
        x = self.flatten(x)
        # x = x.view(x.size(0), -1)
        x = self.fcl1(F.leaky_relu(x))
        x = self.fcl2(F.leaky_relu(x))
        x = self.fcl3(F.leaky_relu(x))
        x = F.softmax(x, dim=1)
        return x
