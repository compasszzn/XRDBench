import torch
import torch.nn as nn
import torch.nn.functional as F

class XCA(nn.Module):
    def __init__(self,task):
        super(XCA, self).__init__()

        # Define Convolutional Layers
        self.conv = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=5, stride=2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.AvgPool1d(kernel_size=1),
            nn.Conv1d(8, 8, kernel_size=5, stride=2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.AvgPool1d(kernel_size=1),
            nn.Conv1d(8, 4, kernel_size=5, stride=2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.AvgPool1d(kernel_size=1)
        )
        self.dropout = nn.Dropout(0.4)
        # Define Dense Layers
        if task=='spg':
            self.fc = nn.Sequential(
                nn.Linear(1740, 230)
            )
        elif task=="crysystem":
            self.fc = nn.Sequential(
                nn.Linear(1740, 7)
            )
    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)  # Flatten the output
        x = self.dropout(x)
        x = self.fc(x)
        return F.softmax(x, dim=1)