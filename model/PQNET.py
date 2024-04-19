import torch
import torch.nn as nn
import torch.nn.functional as F
## A deep convolutional neural network for real-time full profile analysis of big powder diffraction data
class Model(nn.Module):
    def __init__(self,args):
        super(Model, self).__init__()
        self.conv1 = nn.Conv1d(1, 128, kernel_size=35, stride=1,padding=17)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=35, stride=1, padding=17)
        self.conv3 = nn.Conv1d(128, 128, kernel_size=35, stride=1, padding=17)
        self.conv4 = nn.Conv1d(128, 64, kernel_size=30, stride=1)
        self.conv5 = nn.Conv1d(64, 64, kernel_size=25, stride=1)
        self.conv6 = nn.Conv1d(64, 64, kernel_size=25, stride=1)
        self.conv7 = nn.Conv1d(64, 64, kernel_size=25, stride=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fcl1 = nn.Linear(5504,500)
        self.fcl2 = nn.Linear(500,250)
        if args.task=='spg':
            self.fcl3 = nn.Linear(250,230)
        elif args.task=="crysystem":
            self.fcl3 = nn.Linear(250,7)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = self.pool1(F.leaky_relu(self.conv3(x)))
        x = self.pool1(F.leaky_relu(self.conv4(x)))
        x = self.pool1(F.leaky_relu(self.conv5(x)))
        x = self.pool1(F.leaky_relu(self.conv6(x)))
        x = self.pool1(F.leaky_relu(self.conv7(x)))
        x = self.flatten(x)
        x = self.fcl1(F.leaky_relu(x))
        x = self.dropout(x)
        x = self.fcl2(F.leaky_relu(x))
        x = self.dropout(x)
        x = self.fcl3(F.leaky_relu(x))
        x = F.softmax(x, dim=1)
        return x
