import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F

class AUTOANALYZER(nn.Module):
    def __init__(self, dropout_rate,task):
        super(AUTOANALYZER, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=35, stride=1, padding=17),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=30, stride=1, padding=15),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=25, stride=1, padding=12),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=20, stride=1, padding=10),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=1, stride=2, padding=0),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=15, stride=1, padding=7),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=1, stride=2, padding=0),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=10, stride=1, padding=5),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=1, stride=2, padding=0)
        )
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout_rate)
        if task=='spg':
            self.dense_layers = nn.Sequential(
                nn.Linear(4544, 1200),
                nn.ReLU(),
                nn.BatchNorm1d(1200),
                self.dropout,
                nn.Linear(1200, 230),
                nn.Softmax(dim=1)
            )
        elif task=='crysystem':
            self.dense_layers = nn.Sequential(
                nn.Linear(4544, 1200),
                nn.ReLU(),
                nn.BatchNorm1d(1200),
                self.dropout,
                nn.Linear(1200, 7),
                nn.Softmax(dim=1)
            )

    def forward(self, x):
        x = F.interpolate(x, size=4501,mode='linear', align_corners=False)
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.dense_layers(x)
        return x